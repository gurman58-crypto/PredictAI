"""Turn the raw WikiText-2 files into integer token streams + the shared vocabulary.

Order of operations (important for a fair evaluation):
  1. use the official train / valid / test split (article-level, so no text overlaps);
  2. clean WikiText-specific artefacts and tokenise with the SHARED tokenizer;
  3. fit the vocabulary on the TRAINING split only;
  4. encode all splits (unknown tokens -> <unk>), save as .npy;
  5. save vocab.json (+ preprocessing config) next to the model artifacts.

Usage:  python -m ml_models.dataset.prepare_dataset
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ml_models import config
from ml_models.preprocessing.text_processing import (
    EOS,
    EOS_ID,
    UNK_ID,
    Vocabulary,
    tokenize,
)

# WikiText stores "1 @,@ 000", "role @-@ playing", "did n't", "game 's"
_WT_JOIN = re.compile(r"\s@([-,.])@\s")
_WT_CLITIC = re.compile(r"\s+(?=n't\b|'(?:s|re|ve|ll|d|m)\b)")


def clean_wikitext_line(line: str) -> str:
    """Source-specific clean-up (undo WikiText's pre-tokenisation). Not applied to user text."""
    line = _WT_JOIN.sub(r"\1", line)
    return _WT_CLITIC.sub("", line)


def text_to_tokens(raw: str) -> list[str]:
    """Paragraph per line; headings ("= Title =") are skipped; ``<eos>`` ends each paragraph."""
    tokens: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        toks = tokenize(clean_wikitext_line(line))
        if toks:
            tokens.extend(toks)
            tokens.append(EOS)
    return tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--saved", type=Path, default=config.SAVED_DIR)
    ap.add_argument("--max-vocab", type=int, default=config.MAX_VOCAB)
    ap.add_argument("--min-freq", type=int, default=config.MIN_FREQ)
    args = ap.parse_args(argv)

    splits = {}
    for name in ("train", "valid", "test"):
        path = args.raw / f"{name}.txt"
        if not path.exists():
            print(f"missing {path} - run: python -m ml_models.dataset.download_dataset")
            return 1
        splits[name] = text_to_tokens(path.read_text(encoding="utf-8"))
        print(f"{name:5s}: {len(splits[name]):>9,} tokens")

    vocab = Vocabulary.build(splits["train"], max_size=args.max_vocab, min_freq=args.min_freq)
    print(f"vocabulary: {len(vocab):,} tokens (fitted on the training split only)")

    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, toks in splits.items():
        ids = np.asarray(vocab.encode(toks), dtype=np.int32)
        np.save(args.out / f"{name}.npy", ids)
        stats[name] = {
            "tokens": int(len(ids)),
            "unk_rate": round(float((ids == UNK_ID).mean()), 5),
            "paragraphs": int((ids == EOS_ID).sum()),
        }
        print(f"  {name}: unk rate {stats[name]['unk_rate']:.2%}")

    source_file = args.raw / "SOURCE.json"
    source = json.loads(source_file.read_text()) if source_file.exists() else {}
    dataset_meta = {
        "dataset": config.DATASET_NAME,
        "licence": config.DATASET_LICENSE,
        "source": source,
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_vocab": args.max_vocab,
        "min_freq": args.min_freq,
        "vocab_size": len(vocab),
        "splits": stats,
    }
    (args.out / "dataset_meta.json").write_text(json.dumps(dataset_meta, indent=2))
    vocab.save(args.saved / "vocab.json", extra={"dataset": dataset_meta})
    print("saved vocab ->", args.saved / "vocab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
