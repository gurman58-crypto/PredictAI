"""Evaluate every available model on the SAME held-out test split and write the report.

Protocol (identical for all models, see documentation/EVALUATION.md):
  * target set: test-stream positions EVAL_MIN_CONTEXT, EVAL_MIN_CONTEXT+1, ... in blocks of
    EVAL_BLOCK - exactly the same positions for the N-gram variants and for the LSTM;
  * perplexity  = exp(mean negative log-likelihood) over ALL targets (``<unk>`` included),
    each model's distribution is over the same 10k-token vocabulary;
  * top-1 / top-5 accuracy = share of targets whose true token is ranked first / in the top five,
    ranking among all tokens except <pad> and <unk>; targets that are <unk> are skipped;
  * latency = end-to-end ``PredictionEngine.predict`` time (preprocessing + model + ranking, no
    HTTP) on the same 300 sampled contexts for every model, after a warm-up;
  * model size = artifact size on disk.

Usage:  python -m ml_models.evaluation.evaluate        # evaluates every model that has been trained
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml_models import config
from ml_models.inference import KERAS_MODELS, MODEL_INFO, PredictionEngine
from ml_models.preprocessing.text_processing import PAD_ID, UNK_ID, detokenize


class Scorer:
    """Accumulates NLL / top-k hits from batches of probability rows."""

    def __init__(self):
        self.n = 0
        self.nll = 0.0
        self.n_acc = 0
        self.top1 = 0
        self.top5 = 0

    def add(self, probs: np.ndarray, targets: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=np.float64)
        rows = np.arange(len(targets))
        self.nll += float(-np.log(np.maximum(probs[rows, targets], 1e-12)).sum())
        self.n += len(targets)

        ranked = probs.copy()
        ranked[:, PAD_ID] = 0.0
        ranked[:, UNK_ID] = 0.0
        valid = targets != UNK_ID
        if not valid.any():
            return
        ranked, tg = ranked[valid], targets[valid]
        top5 = np.argpartition(-ranked, 4, axis=1)[:, :5]
        self.top5 += int((top5 == tg[:, None]).any(axis=1).sum())
        self.top1 += int((ranked.argmax(axis=1) == tg).sum())
        self.n_acc += len(tg)

    def result(self) -> dict:
        return {
            "perplexity": round(float(np.exp(self.nll / self.n)), 2),
            "top1_accuracy": round(100.0 * self.top1 / self.n_acc, 2),
            "top5_accuracy": round(100.0 * self.top5 / self.n_acc, 2),
            "targets_scored": self.n,
            "targets_in_accuracy": self.n_acc,
        }


def target_blocks(n_tokens: int) -> np.ndarray:
    first = config.EVAL_MIN_CONTEXT
    return np.arange(first, n_tokens - config.EVAL_BLOCK + 1, config.EVAL_BLOCK)


def eval_ngram(model, test: np.ndarray, t0s: np.ndarray, order: int) -> dict:
    scorer = Scorer()
    targets_all = (t0s[:, None] + np.arange(config.EVAL_BLOCK)[None, :]).ravel()
    chunk = 512
    for i in range(0, len(targets_all), chunk):
        ts = targets_all[i : i + chunk]
        probs = np.stack([model.next_token_probs(test[max(0, t - 2) : t], order=order) for t in ts])
        scorer.add(probs, test[ts])
    return scorer.result()


def eval_keras(model, test: np.ndarray, t0s: np.ndarray, batch: int = 128) -> dict:
    """LSTM and Transformer: one forward pass over a window gives the prediction at every position."""
    scorer = Scorer()
    min_ctx, block = config.EVAL_MIN_CONTEXT, config.EVAL_BLOCK
    width = min_ctx + block - 1
    for i in range(0, len(t0s), batch):
        starts = t0s[i : i + batch]
        windows = test[(starts - min_ctx)[:, None] + np.arange(width)[None, :]].astype(np.int32)
        probs = model.keras_model.predict(windows, batch_size=batch, verbose=0)[:, -block:, :]
        probs = probs.reshape(-1, probs.shape[-1]).astype(np.float64)
        probs[:, PAD_ID] = 0.0
        probs /= probs.sum(axis=1, keepdims=True)
        targets = test[(starts[:, None] + np.arange(block)[None, :]).ravel()]
        scorer.add(probs, targets)
    return scorer.result()


def eval_ensemble(model, test: np.ndarray, t0s: np.ndarray, batch: int = 64) -> dict:
    """Blend: weighted sum of each member's distribution at exactly the same target positions."""
    scorer = Scorer()
    min_ctx, block = config.EVAL_MIN_CONTEXT, config.EVAL_BLOCK
    width = min_ctx + block - 1
    for i in range(0, len(t0s), batch):
        starts = t0s[i : i + batch]
        targets_idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
        mixed = 0.0
        for mid, member in model.members.items():
            if mid == "ngram":
                p = np.stack([member.next_token_probs(test[max(0, t - 2) : t]) for t in targets_idx])
            else:
                windows = test[(starts - min_ctx)[:, None] + np.arange(width)[None, :]].astype(np.int32)
                p = member.keras_model.predict(windows, batch_size=batch, verbose=0)[:, -block:, :]
                p = p.reshape(-1, p.shape[-1]).astype(np.float64)
                p[:, PAD_ID] = 0.0
                p /= p.sum(axis=1, keepdims=True)
            mixed = mixed + model.weights[mid] * p
        scorer.add(mixed / mixed.sum(axis=1, keepdims=True), test[targets_idx])
    return scorer.result()


def measure_latency(engine: PredictionEngine, model_id: str, texts: list[str]) -> dict:
    for text in texts[: config.LATENCY_WARMUP]:
        engine.predict(text, model_id, 5)
    times = []
    for text in texts:
        t0 = time.perf_counter()
        engine.predict(text, model_id, 5)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "samples": len(arr),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--saved", type=Path, default=config.SAVED_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    test = np.load(args.data / "test.npy")
    engine = PredictionEngine(args.saved)
    if not engine.any_ready():
        print("no trained model could be loaded - nothing to evaluate")
        for s in engine.statuses():
            print(f"  {s.id}: {s.reason}")
        return 1

    t0s = target_blocks(len(test))
    n_targets = len(t0s) * config.EVAL_BLOCK
    print(f"test tokens: {len(test):,}; evaluated targets: {n_targets:,} (same for every model)")

    rng = np.random.default_rng(args.seed)
    positions = rng.integers(40, len(test), size=config.LATENCY_SAMPLES + config.LATENCY_WARMUP)
    texts = [detokenize(engine.vocab.decode(test[p - 12 : p])) + " " for p in positions]

    models, variants = {}, {}
    for status in engine.statuses():
        mid = status.id
        if not engine.is_ready(mid):
            models[mid] = {"status": "pending", "reason": status.reason}
            print(f"{mid}: pending ({status.reason})")
            continue
        model = engine._models[mid]
        t = time.time()
        if mid == "ngram":
            metrics = eval_ngram(model, test, t0s, order=model.order)
            for order, label in ((1, "unigram"), (2, "bigram"), (3, "trigram")):
                variants[label] = eval_ngram(model, test, t0s, order=order)
                print(f"  ngram {label}: {variants[label]}")
            artifact = args.saved / "ngram" / "ngram_model.npz"
            extra = {"n_bigrams": len(model.bigram), "n_trigrams": len(model.trigram), "discount": model.discount}
        elif mid == "ensemble":
            metrics = eval_ensemble(model, test, t0s)
            sizes = [args.saved / m / ("ngram_model.npz" if m == "ngram" else KERAS_MODELS[m][2]) for m in model.members]
            artifact = args.saved / "ensemble" / "ensemble_meta.json"
            extra = {"weights": model.weights, "members": list(model.members),
                     "members_size_mb": round(sum(p.stat().st_size for p in sizes) / 1e6, 2)}
        else:
            metrics = eval_keras(model, test, t0s)
            artifact = args.saved / mid / KERAS_MODELS[mid][2]
            keep = ("epochs_run", "best_epoch", "training_seconds", "seq_len", "embed_dim", "lstm_units",
                    "d_model", "heads", "layers", "ff_dim")
            extra = {"parameters": int(model.keras_model.count_params()),
                     **{k: model.meta[k] for k in keep if k in model.meta}}
        metrics.update(
            {
                "status": "evaluated",
                "latency": measure_latency(engine, mid, texts),
                "model_size_mb": extra.get("members_size_mb", round(artifact.stat().st_size / 1e6, 2)),
                "version": model.version,
                **extra,
            }
        )
        models[mid] = metrics
        print(f"{mid}: {metrics}  ({time.time() - t:.0f}s)")

    vocab_meta = json.loads((args.saved / "vocab.json").read_text())
    report = {
        "experiment_version": config.MODEL_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": config.DATASET_NAME,
            "licence": config.DATASET_LICENSE,
            "vocab_size": vocab_meta["size"],
            "test_tokens": int(len(test)),
            "test_targets_evaluated": int(n_targets),
            "test_unk_rate": round(float((test == UNK_ID).mean()), 4),
        },
        "protocol": {
            "target_set": f"test positions from {config.EVAL_MIN_CONTEXT}, blocks of {config.EVAL_BLOCK} (same for all models)",
            "perplexity": "exp(mean NLL) over all targets including <unk>; full-vocabulary softmax / smoothed distribution",
            "accuracy": "top-1 / top-5 over all tokens except <pad>/<unk>; <unk> targets skipped",
            "latency": f"PredictionEngine.predict end-to-end (no HTTP), {config.LATENCY_SAMPLES} sampled contexts after {config.LATENCY_WARMUP} warm-up calls, 1 thread of requests",
            "context": "LSTM and Transformer see 20-29 left-context tokens per target; N-gram sees its own order-1 tokens",
        },
        "environment": {"python": platform.python_version(), "machine": platform.platform(), "numpy": np.__version__},
        "models": models,
        "ngram_variants": variants,
    }
    (args.saved / "evaluation_results.json").write_text(json.dumps(report, indent=2))

    rows = []
    for mid, m in models.items():
        if m.get("status") == "evaluated":
            rows.append(
                {
                    "model": mid, "top1_%": m["top1_accuracy"], "top5_%": m["top5_accuracy"],
                    "perplexity": m["perplexity"], "median_ms": m["latency"]["median_ms"],
                    "p95_ms": m["latency"]["p95_ms"], "size_MB": m["model_size_mb"],
                }
            )
    for label, m in variants.items():
        rows.append({"model": f"ngram-{label}", "top1_%": m["top1_accuracy"], "top5_%": m["top5_accuracy"], "perplexity": m["perplexity"]})
    df = pd.DataFrame(rows)
    df.to_csv(args.saved / "evaluation_summary.csv", index=False)
    print("\n" + df.to_string(index=False))

    # qualitative examples (same prompts for every model)
    ex_rows = []
    for p in rng.integers(60, len(test), size=12):
        text = detokenize(engine.vocab.decode(test[p - 10 : p])) + " "
        actual = engine.vocab.itos[int(test[p])]
        for mid in MODEL_INFO:
            if engine.is_ready(mid):
                res = engine.predict(text, mid, 5)
                ex_rows.append({"context": text, "actual_next": actual, "model": mid,
                                "top5": ", ".join(w for w, _ in res.predictions)})
    pd.DataFrame(ex_rows).to_csv(args.saved / "evaluation_examples.csv", index=False)
    print("wrote", args.saved / "evaluation_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
