"""Model-agnostic inference layer used by the API *and* by the evaluation script.

Loads the shared vocabulary plus whichever trained models are available and exposes
``predict`` (top-k next word / word completion) and ``complete`` (bounded sentence
continuation).  Missing artifacts never crash start-up: the model is reported as
``unavailable`` together with a human-readable reason.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ml_models import config
from ml_models.preprocessing.text_processing import (
    EOS,
    EOS_ID,
    PAD_ID,
    SENTENCE_END,
    UNK_ID,
    TextPreprocessor,
    Vocabulary,
    capitalise_after,
    detokenize,
)

_SENTENCE_START = re.compile(r'([.!?]["\')]?\s+)([a-z])')

MODEL_INFO = {
    "ensemble": {
        "name": "PredictAI Blend",
        "description": "Recommended. Mixes the N-gram, LSTM and Transformer predictions with weights tuned on validation data, "
        "so it suggests the most relevant words overall.",
    },
    "ngram": {
        "name": "N-gram (trigram)",
        "description": "Statistical baseline: unigram/bigram/trigram counts with interpolated "
        "absolute-discount smoothing and back-off.",
    },
    "lstm": {
        "name": "LSTM neural network",
        "description": "Embedding + LSTM + Dense softmax language model trained with TensorFlow/Keras.",
    },
    "transformer": {
        "name": "Transformer",
        "description": "Decoder-only Transformer (causal multi-head self-attention) language model trained with TensorFlow/Keras.",
    },
}

# Keras models: id -> (module, class, artifact folder, artifact file, metadata file, training script, env switch)
KERAS_MODELS = {
    "lstm": ("ml_models.lstm.lstm_model", "LSTMModel", "lstm_model.keras", "lstm_meta.json", "train_lstm.py", "ENABLE_LSTM"),
    "transformer": (
        "ml_models.transformer.transformer_model", "TransformerModel",
        "transformer_model.keras", "transformer_meta.json", "train_transformer.py", "ENABLE_TRANSFORMER",
    ),
}


class ModelError(Exception):
    """Base class for inference errors that map to a clean API response."""


class UnknownModelError(ModelError):
    pass


class ModelUnavailableError(ModelError):
    pass


@dataclass
class PredictionResult:
    model: str
    model_version: str
    mode: str
    prefix: str | None
    context_tokens: list[str]
    predictions: list[tuple[str, float]]
    latency_ms: float


@dataclass
class CompletionResult:
    model: str
    model_version: str
    continuation: str
    tokens: list[str]
    stopped_because: str
    latency_ms: float


@dataclass
class ModelStatus:
    id: str
    name: str
    description: str
    status: str                  # "ready" | "unavailable"
    version: str | None = None
    reason: str | None = None
    vocab_size: int | None = None
    details: dict = field(default_factory=dict)


class PredictionEngine:
    def __init__(self, saved_dir: Path | str | None = None, enable_lstm: bool = True, enable_transformer: bool = True):
        self.saved_dir = Path(saved_dir) if saved_dir else config.SAVED_DIR
        self.vocab: Vocabulary | None = None
        self.preprocessor: TextPreprocessor | None = None
        self._models: dict[str, object] = {}
        self._status: dict[str, ModelStatus] = {}
        self._load({"lstm": enable_lstm, "transformer": enable_transformer})

    # ---------------------------------------------------------------- loading
    def _set_status(self, model_id: str, status: str, **kw) -> None:
        info = MODEL_INFO[model_id]
        self._status[model_id] = ModelStatus(
            id=model_id, name=info["name"], description=info["description"], status=status, **kw
        )

    def _load(self, enabled: dict) -> None:
        vocab_path = self.saved_dir / "vocab.json"
        if not vocab_path.exists():
            reason = "vocab.json not found - run the dataset preparation and training scripts"
            for mid in MODEL_INFO:
                self._set_status(mid, "unavailable", reason=reason)
            return
        try:
            self.vocab = Vocabulary.load(vocab_path)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            for mid in MODEL_INFO:
                self._set_status(mid, "unavailable", reason=f"vocabulary could not be loaded: {exc}")
            return
        self.preprocessor = TextPreprocessor(
            self.vocab, max_context=config.MAX_CONTEXT_TOKENS, max_chars=config.MAX_INPUT_CHARS
        )
        self._load_ngram()
        for model_id, (*_, env_switch) in KERAS_MODELS.items():
            if enabled.get(model_id, True):
                self._load_keras(model_id)
            else:
                self._set_status(model_id, "unavailable", reason=f"disabled by configuration ({env_switch}=false)")
        self._load_ensemble()

    def _load_ngram(self) -> None:
        path = self.saved_dir / "ngram" / "ngram_model.npz"
        if not path.exists():
            self._set_status("ngram", "unavailable", reason="not trained yet - run training/train_ngram.py")
            return
        try:
            from ml_models.ngram.ngram_model import NGramModel

            model = NGramModel.load(path)
            if model.vocab_size != len(self.vocab):
                raise ValueError("vocabulary size mismatch between N-gram artifact and vocab.json")
            self._models["ngram"] = model
            self._set_status("ngram", "ready", version=model.version, vocab_size=model.vocab_size, details=model.meta())
        except Exception as exc:  # noqa: BLE001
            self._set_status("ngram", "unavailable", reason=f"failed to load: {exc}")

    def _load_ensemble(self) -> None:
        meta_path = self.saved_dir / "ensemble" / "ensemble_meta.json"
        if not meta_path.exists():
            self._set_status("ensemble", "unavailable", reason="not tuned yet - run training/train_ensemble.py")
            return
        try:
            from ml_models.ensemble.ensemble_model import EnsembleModel

            meta = json.loads(meta_path.read_text())
            model = EnsembleModel(dict(self._models), meta["weights"], len(self.vocab), meta)
            self._models["ensemble"] = model
            self._set_status("ensemble", "ready", version=model.version, vocab_size=model.vocab_size, details=model.meta)
        except Exception as exc:  # noqa: BLE001
            self._set_status("ensemble", "unavailable", reason=f"could not be built: {exc}")

    def _load_keras(self, model_id: str) -> None:
        module_name, class_name, artifact, meta_file, script, _ = KERAS_MODELS[model_id]
        label = MODEL_INFO[model_id]["name"]
        path = self.saved_dir / model_id / artifact
        if not path.exists():
            self._set_status(model_id, "unavailable", reason=f"not trained yet - run training/{script}")
            return
        try:
            import importlib

            model_cls = getattr(importlib.import_module(module_name), class_name)
        except Exception as exc:  # noqa: BLE001
            self._set_status(model_id, "unavailable", reason=f"TensorFlow could not be imported: {exc}")
            return
        try:
            model = model_cls.load(path, self.saved_dir / model_id / meta_file)
            if model.vocab_size != len(self.vocab):
                raise ValueError(f"vocabulary size mismatch between {label} artifact and vocab.json")
            model.next_token_probs([EOS_ID])  # warm-up: traces the graph once
            self._models[model_id] = model
            self._set_status(model_id, "ready", version=model.version, vocab_size=model.vocab_size, details=model.meta)
        except Exception as exc:  # noqa: BLE001
            self._set_status(model_id, "unavailable", reason=f"failed to load: {exc}")

    # ----------------------------------------------------------------- status
    def statuses(self) -> list[ModelStatus]:
        return [self._status[m] for m in MODEL_INFO]

    def is_ready(self, model_id: str) -> bool:
        return model_id in self._models

    def any_ready(self) -> bool:
        return bool(self._models)

    def _get(self, model_id: str):
        if model_id not in MODEL_INFO:
            raise UnknownModelError(f"unknown model {model_id!r}; choose one of {sorted(MODEL_INFO)}")
        model = self._models.get(model_id)
        if model is None:
            reason = self._status[model_id].reason or "not loaded"
            raise ModelUnavailableError(f"model {model_id!r} is not available: {reason}")
        return model

    # ---------------------------------------------------------------- predict
    def _rank(self, probs: np.ndarray, prefix: str | None, top_k: int) -> list[tuple[str, float]]:
        mask = self.vocab.prefix_mask(prefix) if prefix else self.vocab.word_mask
        scores = np.where(mask, probs, 0.0)
        k = min(top_k, int(np.count_nonzero(scores)))
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self.vocab.itos[i], float(scores[i])) for i in idx if scores[i] > 0]

    def predict(self, text: str, model_id: str = "ngram", top_k: int = 5) -> PredictionResult:
        t0 = time.perf_counter()
        model = self._get(model_id)
        parsed = self.preprocessor.parse(text)
        probs = model.next_token_probs(parsed.context_ids)
        preds = self._rank(probs, parsed.prefix, top_k)
        return PredictionResult(
            model=model_id,
            model_version=model.version,
            mode=parsed.mode,
            prefix=parsed.prefix,
            context_tokens=parsed.context_tokens[-8:],
            predictions=preds,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # --------------------------------------------------------------- complete
    def complete(
        self,
        text: str,
        model_id: str = "ngram",
        max_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 8,
        stop_at_sentence: bool = True,
        seed: int | None = None,
    ) -> CompletionResult:
        """Bounded continuation by repeated sampling from the model's top-k words."""
        t0 = time.perf_counter()
        model = self._get(model_id)
        parsed = self.preprocessor.parse(text, whole_word=True)
        rng = np.random.default_rng(seed)
        # generated text never opens quotes/brackets: a sampler cannot keep them balanced
        blocked = [self.vocab.stoi[t] for t in ('"', "(", ")") if t in self.vocab.stoi]
        ctx = list(parsed.context_ids)
        prefix = parsed.prefix
        generated: list[str] = []
        stopped = "max_tokens"

        for step in range(max_tokens):
            probs = model.next_token_probs(ctx).copy()
            probs[PAD_ID] = 0.0
            probs[UNK_ID] = 0.0
            probs[blocked] = 0.0
            if step == 0:
                probs[EOS_ID] = 0.0   # an explicit "continue" request must produce something
            if step == 0 and prefix:
                probs = np.where(self.vocab.prefix_mask(prefix), probs, 0.0)
            for tok in generated[-10:]:                       # mild repetition penalty
                tid = self.vocab.stoi.get(tok)
                if tid is not None and self.vocab.word_mask[tid]:
                    probs[tid] *= 0.5
            k = min(top_k, int(np.count_nonzero(probs)))
            if k == 0:
                stopped = "no_candidates"
                break
            idx = np.argpartition(-probs, k - 1)[:k]
            w = probs[idx] ** (1.0 / max(temperature, 1e-3))
            nxt = int(rng.choice(idx, p=w / w.sum()))
            if nxt == EOS_ID:
                stopped = "end_of_text"
                break
            token = self.vocab.itos[nxt]
            generated.append(token)
            ctx.append(nxt)
            ctx = ctx[-config.MAX_CONTEXT_TOKENS :]
            if stop_at_sentence and token in SENTENCE_END and len(generated) >= 3:
                stopped = "sentence_end"
                break

        continuation = self._join_continuation(text, parsed, generated)
        model_obj = self._models[model_id]
        return CompletionResult(
            model=model_id,
            model_version=model_obj.version,
            continuation=continuation,
            tokens=generated,
            stopped_because=stopped,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    @staticmethod
    def _join_continuation(text: str, parsed, generated: list[str]) -> str:
        if not generated:
            return ""
        if parsed.prefix:
            # the first token completes the word the user is typing: only add the remainder
            first = generated[0]
            remainder = first[len(parsed.prefix):] if first.startswith(parsed.prefix) else first
            rest = detokenize(generated[1:], quote_open=parsed.quote_open)
            rest = (" " + rest) if rest and not rest[0] in ".,;:!?)'" else rest
            return remainder + rest
        body = detokenize(generated, quote_open=parsed.quote_open)
        body = _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), body)   # "year. the" -> "year. The"
        body = capitalise_after(text, body)
        glue = body[:1] in (".", ",", ";", ":", "!", "?", ")", "'")
        needs_space = bool(text) and not text[-1].isspace() and text[-1] != "(" and not glue
        return (" " if needs_space else "") + body
