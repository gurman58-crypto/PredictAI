"""Train the decoder-only Transformer language model.

Same data, vocabulary, window sampling, validation-based best-epoch selection and wall-clock budget as the
LSTM (see ``keras_common.py``), so the comparison is fair.  Differences: a short learning-rate warm-up (Transformers
are unstable without one) and a smaller dropout.  Training is resumable: re-run the same command after an interruption.

Usage:  python -m ml_models.training.train_transformer
        python -m ml_models.training.train_transformer --quick            # 1-minute smoke test
        python -m ml_models.training.train_transformer --epochs 15 --max-minutes 90
"""
from __future__ import annotations

import argparse
import sys

from ml_models import config


def main(argv: list[str] | None = None) -> int:
    from ml_models.training import keras_common as kc  # imports TensorFlow

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kc.add_common_args(ap, epochs=8, batch_size=128, lr=1.5e-3, max_minutes=45.0, saved_name="transformer_model.keras")
    ap.add_argument("--seq-len", type=int, default=config.TRANSFORMER_SEQ_LEN)
    ap.add_argument("--d-model", type=int, default=config.TRANSFORMER_D_MODEL)
    ap.add_argument("--heads", type=int, default=config.TRANSFORMER_HEADS)
    ap.add_argument("--layers", type=int, default=config.TRANSFORMER_LAYERS)
    ap.add_argument("--ff", type=int, default=config.TRANSFORMER_FF)
    ap.add_argument("--dropout", type=float, default=config.TRANSFORMER_DROPOUT)
    ap.add_argument("--warmup-steps", type=int, default=400)
    args = ap.parse_args(argv)

    import tensorflow as tf
    from tensorflow import keras

    from ml_models.transformer.transformer_model import build_transformer

    tf.keras.utils.set_random_seed(args.seed)
    train, valid, vocab_size = kc.load_splits(args)

    model = build_transformer(vocab_size, args.seq_len, args.d_model, args.heads, args.layers, args.ff, args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0), loss="sparse_categorical_crossentropy")
    kc.run_training(
        model, name="transformer", args=args, train=train, valid=valid, seq_len=args.seq_len,
        extra_callbacks=[kc.LinearWarmup(args.lr, args.warmup_steps)],
        extra_meta={
            "architecture": "Embedding + learned positions -> N x [LayerNorm, causal self-attention, LayerNorm, GELU feed-forward] "
                            "(pre-LN, residual) -> LayerNorm -> Dense(softmax)",
            "d_model": args.d_model, "heads": args.heads, "layers": args.layers, "ff_dim": args.ff,
            "dropout": args.dropout, "warmup_steps": args.warmup_steps,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
