"""PredictAI Blend: a weighted mixture of the trained models' next-word distributions.

    p(w | context) = sum_i  weight_i * p_i(w | context)

The weights are fitted on the VALIDATION split only (see ``training/train_ensemble.py``); the test split is never
used.  A mixture of probability distributions is itself a valid distribution, so perplexity and top-k are
measured exactly like for the single models.  If a member model is unavailable the remaining weights are
re-normalised, so the blend degrades gracefully.
"""
from __future__ import annotations

import numpy as np


class EnsembleModel:
    model_id = "ensemble"

    def __init__(self, members: dict, weights: dict, vocab_size: int, meta: dict | None = None):
        usable = {m: float(weights.get(m, 0.0)) for m in members if weights.get(m, 0.0) > 0}
        if len(usable) < 2:
            raise ValueError("the blend needs at least two available member models with positive weight")
        total = sum(usable.values())
        self.members = {m: members[m] for m in usable}
        self.weights = {m: w / total for m, w in usable.items()}
        self.vocab_size = vocab_size
        self.meta = dict(meta or {}, weights={m: round(w, 4) for m, w in self.weights.items()})
        self.version = self.meta.get("version", "v1")

    def next_token_probs(self, context_ids) -> np.ndarray:
        probs = None
        for mid, model in self.members.items():
            p = np.asarray(model.next_token_probs(context_ids), dtype=np.float64) * self.weights[mid]
            probs = p if probs is None else probs + p
        return probs / probs.sum()


def fit_weights(target_probs: np.ndarray, iters: int = 300) -> np.ndarray:
    """Maximum-likelihood mixture weights by EM.  ``target_probs[n, i]`` = p_i(true word of example n)."""
    n, k = target_probs.shape
    w = np.full(k, 1.0 / k)
    p = np.maximum(target_probs, 1e-12)
    for _ in range(iters):
        r = p * w
        r /= r.sum(axis=1, keepdims=True)
        w = r.mean(axis=0)
    return w
