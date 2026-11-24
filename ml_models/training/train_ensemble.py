"""Fit the blend weights of the trained models on the VALIDATION split (takes about a minute).

    python -m ml_models.training.train_ensemble

Writes ``saved_models/ensemble/ensemble_meta.json``.  Run it after the individual models are trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ml_models import config
from ml_models.ensemble.ensemble_model import fit_weights
from ml_models.evaluation.evaluate import target_blocks
from ml_models.inference import PredictionEngine
from ml_models.preprocessing.text_processing import PAD_ID


def target_probs(engine: PredictionEngine, mid: str, ids: np.ndarray, t0s: np.ndarray) -> np.ndarray:
    """Probability each model gives to the true next token, at the evaluation target positions."""
    model = engine._models[mid]
    block, min_ctx = config.EVAL_BLOCK, config.EVAL_MIN_CONTEXT
    targets = (t0s[:, None] + np.arange(block)[None, :]).ravel()
    if mid == "ngram":
        return np.array([model.next_token_probs(ids[max(0, t - 2): t])[ids[t]] for t in targets])
    out, width, batch = [], min_ctx + block - 1, 128
    for i in range(0, len(t0s), batch):
        starts = t0s[i: i + batch]
        windows = ids[(starts - min_ctx)[:, None] + np.arange(width)[None, :]].astype(np.int32)
        p = model.keras_model.predict(windows, batch_size=batch, verbose=0)[:, -block:, :].astype(np.float64)
        p[:, :, PAD_ID] = 0.0
        p /= p.sum(axis=2, keepdims=True)
        tg = ids[(starts[:, None] + np.arange(block)[None, :])]
        out.append(np.take_along_axis(p, tg[:, :, None], axis=2)[:, :, 0].ravel())
    return np.concatenate(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--saved", type=Path, default=config.SAVED_DIR)
    ap.add_argument("--every", type=int, default=2, help="use every N-th evaluation block of the validation split")
    args = ap.parse_args(argv)

    valid = np.load(args.data / "valid.npy")
    engine = PredictionEngine(args.saved)
    members = [m for m in ("ngram", "lstm", "transformer") if engine.is_ready(m)]
    if len(members) < 2:
        print("need at least two trained models (train_ngram, train_lstm, train_transformer) first")
        return 1
    t0s = target_blocks(len(valid))[:: args.every]
    print(f"validation targets: {len(t0s) * config.EVAL_BLOCK:,}; members: {members}")
    P = np.stack([target_probs(engine, m, valid, t0s) for m in members], axis=1)
    w = fit_weights(P)
    ppl = lambda col: float(np.exp(-np.log(np.maximum(col, 1e-12)).mean()))
    single = {m: round(ppl(P[:, i]), 2) for i, m in enumerate(members)}
    blend = round(ppl(P @ w), 2)
    print("validation perplexity, single models:", single)
    print("weights:", dict(zip(members, np.round(w, 4))), "-> blend perplexity", blend)
    out = args.saved / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ensemble_meta.json").write_text(json.dumps({
        "version": "v1",
        "method": "linear mixture of probabilities, weights fitted by EM on the validation split",
        "weights": {m: float(x) for m, x in zip(members, w)},
        "validation_targets": int(P.shape[0]),
        "validation_perplexity_single": single,
        "validation_perplexity_blend": blend,
    }, indent=2))
    print("saved", out / "ensemble_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
