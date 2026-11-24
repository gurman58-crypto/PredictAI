"""Train the LSTM language model (Embedding -> LSTM -> Dense softmax).

Uses the training loop shared with the Transformer (``keras_common.py``): window sampling, validation-based
best-epoch selection, early stopping, a wall-clock budget and resume support.

Usage:  python -m ml_models.training.train_lstm
        python -m ml_models.training.train_lstm --quick          # 1-minute smoke test
        python -m ml_models.training.train_lstm --max-minutes 60 --epochs 15
"""
from __future__ import annotations

import argparse
import sys

from ml_models import config


def main(argv: list[str] | None = None) -> int:
    from ml_models.training import keras_common as kc  # imports TensorFlow

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kc.add_common_args(ap, epochs=12, batch_size=128, lr=2e-3, max_minutes=30.0, saved_name="lstm_model.keras")
    ap.add_argument("--seq-len", type=int, default=config.LSTM_SEQ_LEN)
    ap.add_argument("--embed-dim", type=int, default=config.LSTM_EMBED_DIM)
    ap.add_argument("--units", type=int, default=config.LSTM_UNITS)
    ap.add_argument("--dropout", type=float, default=config.LSTM_DROPOUT)
    args = ap.parse_args(argv)

    import tensorflow as tf
    from tensorflow import keras

    from ml_models.lstm.lstm_model import build_lstm

    tf.keras.utils.set_random_seed(args.seed)
    train, valid, vocab_size = kc.load_splits(args)

    model = build_lstm(vocab_size, args.embed_dim, args.units, args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0), loss="sparse_categorical_crossentropy")
    kc.run_training(
        model, name="lstm", args=args, train=train, valid=valid, seq_len=args.seq_len,
        extra_meta={
            "architecture": "Embedding -> LSTM -> Dropout -> Dense(softmax), return_sequences=True",
            "embed_dim": args.embed_dim, "lstm_units": args.units, "dropout": args.dropout,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
