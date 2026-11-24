"""Unigram / bigram / trigram language model with interpolated absolute discounting.

p3(w | a b) = max(c(a b w) - D, 0) / c(a b .)  +  D * N1+(a b .) / c(a b .) * p2(w | b)
p2(w | b)   = max(c(b w) - D, 0)   / c(b .)    +  D * N1+(b .)   / c(b .)   * p1(w)
p1(w)       = c(w) / N

Unseen contexts simply back off to the next lower order, so a distribution over
the whole vocabulary exists for every possible context (unknown words are mapped
to ``<unk>`` by the preprocessor and handled like any other token).

Counts are stored in sorted numpy arrays (CSR layout) so the artifact is small and
look-ups are a binary search.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_models.preprocessing.text_processing import PAD_ID


class _Level:
    """Counts of ``next`` tokens grouped by an integer context key (CSR layout)."""

    FIELDS = ("ctx", "start", "n", "nxt", "cnt", "total")

    def __init__(self, ctx, start, n, nxt, cnt, total):
        self.ctx, self.start, self.n = ctx, start, n
        self.nxt, self.cnt, self.total = nxt, cnt, total

    @classmethod
    def empty(cls) -> "_Level":
        return cls(
            np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int32),
            np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64),
        )

    @classmethod
    def build(cls, keys: np.ndarray, counts: np.ndarray, vocab_size: int) -> "_Level":
        """``keys`` = context * V + next, sorted ascending, with matching ``counts``."""
        if len(keys) == 0:
            return cls.empty()
        ctx = keys // vocab_size
        nxt = (keys % vocab_size).astype(np.int32)
        ctx_u, start, n = np.unique(ctx, return_index=True, return_counts=True)
        total = np.add.reduceat(counts, start)
        return cls(
            ctx_u.astype(np.int64), start.astype(np.int64), n.astype(np.int32),
            nxt, counts.astype(np.int32), total.astype(np.int64),
        )

    def lookup(self, key: int):
        i = int(np.searchsorted(self.ctx, key))
        if i < len(self.ctx) and self.ctx[i] == key:
            s = int(self.start[i])
            e = s + int(self.n[i])
            return self.nxt[s:e], self.cnt[s:e], int(self.total[i]), int(self.n[i])
        return None

    def __len__(self) -> int:
        return len(self.nxt)


class NGramModel:
    model_id = "ngram"

    def __init__(
        self,
        vocab_size: int,
        order: int = 3,
        discount: float = 0.75,
        min_count_bigram: int = 1,
        min_count_trigram: int = 2,
        version: str = "v1",
    ):
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if not 0 < discount < 1:
            raise ValueError("discount must be in (0, 1)")
        self.vocab_size = vocab_size
        self.order = order
        self.discount = discount
        self.min_count_bigram = min_count_bigram
        self.min_count_trigram = min_count_trigram
        self.version = version
        self.unigram_counts = np.zeros(vocab_size, np.int64)
        self.p_uni = np.zeros(vocab_size, np.float64)
        self.bigram = _Level.empty()
        self.trigram = _Level.empty()
        self.n_tokens = 0

    # ------------------------------------------------------------------ fit
    def fit(self, ids: np.ndarray) -> "NGramModel":
        ids = np.asarray(ids, dtype=np.int64)
        V = self.vocab_size
        if len(ids) < 3:
            raise ValueError("need at least 3 tokens to train")
        if ids.min() < 0 or ids.max() >= V:
            raise ValueError("token id outside vocabulary range")
        self.n_tokens = int(len(ids))
        self.unigram_counts = np.bincount(ids, minlength=V).astype(np.int64)
        self.unigram_counts[PAD_ID] = 0
        self.p_uni = self.unigram_counts / self.unigram_counts.sum()

        keys, counts = np.unique(ids[:-1] * V + ids[1:], return_counts=True)
        keep = counts >= self.min_count_bigram
        self.bigram = _Level.build(keys[keep], counts[keep], V)

        keys, counts = np.unique((ids[:-2] * V + ids[1:-1]) * V + ids[2:], return_counts=True)
        keep = counts >= self.min_count_trigram
        self.trigram = _Level.build(keys[keep], counts[keep], V)
        return self

    # ------------------------------------------------------------ inference
    def next_token_probs(self, context_ids, order: int | None = None) -> np.ndarray:
        """Full probability distribution over the vocabulary for the next token."""
        order = self.order if order is None else order
        D = self.discount
        ctx = list(context_ids)
        p = self.p_uni.copy()
        if order >= 2 and len(ctx) >= 1:
            hit = self.bigram.lookup(int(ctx[-1]))
            if hit is not None:
                nxt, cnt, total, types = hit
                p *= D * types / total
                p[nxt] += np.maximum(cnt - D, 0.0) / total
        if order >= 3 and len(ctx) >= 2:
            hit = self.trigram.lookup(int(ctx[-2]) * self.vocab_size + int(ctx[-1]))
            if hit is not None:
                nxt, cnt, total, types = hit
                p *= D * types / total
                p[nxt] += np.maximum(cnt - D, 0.0) / total
        return p

    # ---------------------------------------------------------------- io
    def meta(self) -> dict:
        return {
            "model": "ngram",
            "version": self.version,
            "vocab_size": self.vocab_size,
            "order": self.order,
            "discount": self.discount,
            "smoothing": "interpolated absolute discounting (backoff to lower orders)",
            "min_count_bigram": self.min_count_bigram,
            "min_count_trigram": self.min_count_trigram,
            "n_training_tokens": self.n_tokens,
            "n_bigrams": len(self.bigram),
            "n_trigrams": len(self.trigram),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"unigram": self.unigram_counts}
        for name, level in (("bi", self.bigram), ("tri", self.trigram)):
            for f in _Level.FIELDS:
                arrays[f"{name}_{f}"] = getattr(level, f)
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta())), **arrays)

    @classmethod
    def load(cls, path: Path) -> "NGramModel":
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            model = cls(
                vocab_size=meta["vocab_size"],
                order=meta["order"],
                discount=meta["discount"],
                min_count_bigram=meta["min_count_bigram"],
                min_count_trigram=meta["min_count_trigram"],
                version=meta["version"],
            )
            model.n_tokens = meta["n_training_tokens"]
            model.unigram_counts = z["unigram"].astype(np.int64)
            model.p_uni = model.unigram_counts / model.unigram_counts.sum()
            model.bigram = _Level(*[z[f"bi_{f}"] for f in _Level.FIELDS])
            model.trigram = _Level(*[z[f"tri_{f}"] for f in _Level.FIELDS])
        return model
