"""Train the N-gram model on the prepared training split.

The absolute-discount constant D is chosen on the VALIDATION split (never on test).

Usage:  python -m ml_models.training.train_ngram
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml_models import config
from ml_models.ngram.ngram_model import NGramModel


def validation_perplexity(model: NGramModel, ids: np.ndarray, n: int, order: int) -> float:
    """Perplexity on the first ``n`` validation tokens (all tokens incl. <unk>)."""
    nll = 0.0
    count = 0
    for t in range(2, min(len(ids), n + 2)):
        p = model.next_token_probs(ids[max(0, t - 2) : t], order=order)
        nll -= np.log(max(p[ids[t]], 1e-12))
        count += 1
    return float(np.exp(nll / count))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--saved", type=Path, default=config.SAVED_DIR)
    ap.add_argument("--val-tokens", type=int, default=30_000, help="validation tokens used to pick the discount")
    args = ap.parse_args(argv)

    train = np.load(args.data / "train.npy")
    valid = np.load(args.data / "valid.npy")
    vocab_size = len(json.loads((args.saved / "vocab.json").read_text())["itos"])

    t0 = time.time()
    model = NGramModel(
        vocab_size=vocab_size,
        order=config.NGRAM_ORDER,
        min_count_bigram=config.NGRAM_MIN_COUNT_BIGRAM,
        min_count_trigram=config.NGRAM_MIN_COUNT_TRIGRAM,
        version=config.MODEL_VERSION,
    ).fit(train)
    fit_seconds = time.time() - t0
    print(f"counted {len(train):,} tokens in {fit_seconds:.1f}s: "
          f"{len(model.bigram):,} bigrams, {len(model.trigram):,} trigrams stored")

    grid = {}
    for d in config.NGRAM_DISCOUNT_GRID:
        model.discount = d
        grid[d] = validation_perplexity(model, valid, args.val_tokens, order=3)
        print(f"  discount {d:.2f}: validation perplexity {grid[d]:.1f}")
    model.discount = min(grid, key=grid.get)
    print(f"selected discount {model.discount} (validation perplexity {grid[model.discount]:.1f})")

    out = args.saved / "ngram" / "ngram_model.npz"
    model.save(out)
    meta = model.meta()
    meta.update(
        {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fit_seconds": round(fit_seconds, 2),
            "validation_perplexity_by_discount": {str(k): round(v, 2) for k, v in grid.items()},
            "validation_tokens_used": args.val_tokens,
            "artifact": out.name,
            "size_bytes": out.stat().st_size,
        }
    )
    (args.saved / "ngram" / "ngram_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
