"""One-time download of the transformer weights, then a quick self-test.

Usage (from the project root, with the virtual environment active):
    python -m ml_models.transformer.download
    python -m ml_models.transformer.download --model gpt2      # bigger / better, slower
"""
from __future__ import annotations

import argparse
import sys

from ml_models.transformer.transformer_model import DEFAULT_MODEL, TransformerModel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers / torch are not installed. Run: pip install transformers torch")
        return 1

    print(f"downloading {args.model} (first time only, a few hundred MB) ...")
    AutoTokenizer.from_pretrained(args.model)
    AutoModelForCausalLM.from_pretrained(args.model)

    print("loading from the local cache and testing ...")
    model = TransformerModel(args.model, local_files_only=True)
    for prompt in ("Dear Diary, Today its raining outside", "Thank you for your", "I am writing to inform you that"):
        words, _ = model.predict(prompt + " ", top_k=5)
        gen = model.generate(prompt, max_tokens=20, temperature=0.8, stop_at_sentence=True, seed=1)
        print(f"\n{prompt!r}\n  next words : {[w for w, _ in words]}\n  continues  : {gen.text!r}")
    print("\nOK - start the backend again; the Transformer model will show as ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
