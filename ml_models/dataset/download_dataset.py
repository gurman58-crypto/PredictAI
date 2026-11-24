"""Download WikiText-2 into ml_models/dataset/data/raw/{train,valid,test}.txt

WikiText-2 (Merity et al., 2016) is distributed under CC BY-SA 3.0 (it is Wikipedia text).
Check the licence/terms before redistributing the data; this repository does NOT bundle it.

Sources
  github       (default) raw word-level WikiText-2 files from the pytorch/examples repository
  huggingface  the official Salesforce/wikitext dataset (parquet) - needs ``pyarrow``
  local        copy train.txt / valid.txt / test.txt from a folder you already have

Usage (from the project root):
    python -m ml_models.dataset.download_dataset
    python -m ml_models.dataset.download_dataset --source huggingface
    python -m ml_models.dataset.download_dataset --source local --path /path/to/wikitext-2
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ml_models.config import DATASET_LICENSE, DATASET_NAME, RAW_DIR

GITHUB_BASE = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
HF_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-v1"
SPLITS = {"train": "train", "valid": "validation", "test": "test"}


def _fetch(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URLs
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  attempt {attempt}/{retries} failed: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def from_github(out: Path) -> str:
    for split in SPLITS:
        url = f"{GITHUB_BASE}/{split}.txt"
        print(f"downloading {url}")
        (out / f"{split}.txt").write_bytes(_fetch(url))
    return GITHUB_BASE


def from_huggingface(out: Path) -> str:
    import pandas as pd  # parquet reading needs pyarrow

    for split, hf_name in SPLITS.items():
        url = f"{HF_BASE}/{hf_name}-00000-of-00001.parquet"
        print(f"downloading {url}")
        df = pd.read_parquet(io.BytesIO(_fetch(url)))
        (out / f"{split}.txt").write_text("".join(df["text"].tolist()), encoding="utf-8")
    return HF_BASE


def from_local(out: Path, src: Path) -> str:
    for split in SPLITS:
        candidate = src / f"{split}.txt"
        if not candidate.exists() and split == "valid":
            candidate = src / "validation.txt"
        if not candidate.exists():
            raise FileNotFoundError(f"{src} must contain {split}.txt")
        shutil.copyfile(candidate, out / f"{split}.txt")
    return str(src)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["github", "huggingface", "local"], default="github")
    ap.add_argument("--path", type=Path, help="folder with train/valid/test .txt (for --source local)")
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == "github":
        origin = from_github(args.out)
    elif args.source == "huggingface":
        origin = from_huggingface(args.out)
    else:
        if not args.path:
            ap.error("--path is required with --source local")
        origin = from_local(args.out, args.path)

    for split in SPLITS:
        size = (args.out / f"{split}.txt").stat().st_size
        if size < 10_000:
            print(f"ERROR: {split}.txt is suspiciously small ({size} bytes)")
            return 1
        print(f"  {split}.txt  {size / 1e6:.2f} MB")

    (args.out / "SOURCE.json").write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "source": args.source,
                "origin": origin,
                "licence": DATASET_LICENSE,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    print("done ->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
