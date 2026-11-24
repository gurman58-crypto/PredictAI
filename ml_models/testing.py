"""Helpers that build tiny *real* artifacts (vocabulary, N-gram, optional LSTM) for unit tests.

The toy models are trained on a few dozen sentences, so the tests (ML and API) run in seconds and
never depend on the full WikiText-2 training run.  They are NOT used by the application itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_models.ngram.ngram_model import NGramModel
from ml_models.preprocessing.text_processing import EOS, Vocabulary, tokenize

TOY_SENTENCES = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat chased the dog.",
    "The dog chased the cat.",
    "A bird sang a song in the morning.",
    "The sun rose in the morning.",
    "Students write notes in the library.",
    "Students read books in the library.",
    "I think that artificial intelligence is changing the world.",
    "Artificial intelligence is transforming education.",
    "Machine learning is a part of artificial intelligence.",
    "We don't know what the future will bring.",
    "She said, \"Hello, how are you?\"",
]
TOY_REPEATS = 6


def toy_tokens() -> list[str]:
    tokens: list[str] = []
    for _ in range(TOY_REPEATS):
        for sentence in TOY_SENTENCES:
            tokens.extend(tokenize(sentence))
            tokens.append(EOS)
    return tokens


def build_toy_saved_dir(path: Path | str, with_lstm: bool = False, with_transformer: bool = False,
                         with_ensemble: bool = False) -> Path:
    """Create ``vocab.json`` + trained N-gram (+ tiny LSTM / Transformer) under ``path``."""
    path = Path(path)
    tokens = toy_tokens()
    vocab = Vocabulary.build(tokens, max_size=200, min_freq=1)
    ids = np.asarray(vocab.encode(tokens), dtype=np.int64)
    vocab.save(path / "vocab.json")

    ngram = NGramModel(len(vocab), min_count_trigram=1, discount=0.75).fit(ids)
    ngram.save(path / "ngram" / "ngram_model.npz")

    if with_lstm:
        from tensorflow import keras

        from ml_models.lstm.lstm_model import build_lstm

        seq = 10
        model = build_lstm(len(vocab), embed_dim=8, lstm_units=16, dropout=0.0)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        starts = np.arange(0, len(ids) - seq - 1, 3)
        idx = starts[:, None] + np.arange(seq)[None, :]
        model.fit(ids[idx].astype("int32"), ids[idx + 1].astype("int32"), epochs=2, batch_size=16, verbose=0)
        (path / "lstm").mkdir(parents=True, exist_ok=True)
        model.save(path / "lstm" / "lstm_model.keras")
        (path / "lstm" / "lstm_meta.json").write_text(json.dumps({"version": "toy", "seq_len": seq}))

    if with_transformer:
        from ml_models.transformer.transformer_model import build_transformer

        seq = 10
        model = build_transformer(len(vocab), max_len=seq, d_model=16, num_heads=2, num_layers=1, ff_dim=32, dropout=0.0)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        starts = np.arange(0, len(ids) - seq - 1, 3)
        idx = starts[:, None] + np.arange(seq)[None, :]
        model.fit(ids[idx].astype("int32"), ids[idx + 1].astype("int32"), epochs=2, batch_size=16, verbose=0)
        (path / "transformer").mkdir(parents=True, exist_ok=True)
        model.save(path / "transformer" / "transformer_model.keras")
        (path / "transformer" / "transformer_meta.json").write_text(json.dumps({"version": "toy", "seq_len": seq}))

    if with_ensemble:
        (path / "ensemble").mkdir(parents=True, exist_ok=True)
        weights = {"ngram": 0.4, "lstm": 0.2, "transformer": 0.4}
        (path / "ensemble" / "ensemble_meta.json").write_text(json.dumps({"version": "toy", "weights": weights}))
    return path
