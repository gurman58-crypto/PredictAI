"""Training loop shared by the LSTM and the Transformer, so both are trained and validated identically.

Training is sequence-to-sequence: the corpus is cut into windows of ``seq_len`` tokens and every time step
predicts the following token.  A new random window offset is used every epoch.  Validation loss selects the
best epoch; a wall-clock budget keeps training feasible on a student laptop; training is resumable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tensorflow import keras

from ml_models import config


class WindowSequence(keras.utils.PyDataset):
    """Windows of L tokens; inputs = tokens[i:i+L], targets = tokens[i+1:i+L+1]."""

    def __init__(self, ids, seq_len, batch_size, shuffle, seed=0, **kw):
        super().__init__(**kw)
        self.ids, self.L, self.bs, self.shuffle = ids, seq_len, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self._reindex()

    def _reindex(self):
        offset = int(self.rng.integers(0, self.L)) if self.shuffle else 0
        n = (len(self.ids) - 1 - offset) // self.L
        self.starts = offset + np.arange(n) * self.L
        if self.shuffle:
            self.rng.shuffle(self.starts)

    def __len__(self):
        return int(np.ceil(len(self.starts) / self.bs))

    def __getitem__(self, i):
        s = self.starts[i * self.bs : (i + 1) * self.bs]
        idx = s[:, None] + np.arange(self.L)[None, :]
        return self.ids[idx], self.ids[idx + 1]

    def on_epoch_end(self):
        if self.shuffle:
            self._reindex()


class TimeBudget(keras.callbacks.Callback):
    """Stop starting new epochs once the wall-clock budget would be exceeded."""

    def __init__(self, minutes):
        super().__init__()
        self.limit, self.t0, self.stopped_early, self.done = minutes * 60, time.time(), False, 0

    def on_epoch_end(self, epoch, logs=None):
        self.done += 1
        used = time.time() - self.t0
        avg = used / self.done
        print(f"  [epoch {epoch + 1}] {used / 60:.1f} min used, val perplexity {np.exp(logs['val_loss']):.1f}")
        if used + avg > self.limit and epoch + 1 < self.params["epochs"]:
            print("  time budget reached - stopping after this epoch")
            self.model.stop_training = True
            self.stopped_early = True


class LinearWarmup(keras.callbacks.Callback):
    """Ramp the learning rate from ~0 to ``peak`` over the first ``steps`` optimiser steps."""

    def __init__(self, peak, steps):
        super().__init__()
        self.peak, self.steps = peak, steps

    def on_train_batch_begin(self, batch, logs=None):
        step = int(self.model.optimizer.iterations.numpy())
        if step < self.steps:
            self.model.optimizer.learning_rate.assign(self.peak * (step + 1) / self.steps)


def add_common_args(ap, *, epochs, batch_size, lr, max_minutes, saved_name):
    ap.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    ap.add_argument("--saved", type=Path, default=config.SAVED_DIR)
    ap.add_argument("--epochs", type=int, default=epochs)
    ap.add_argument("--batch-size", type=int, default=batch_size)
    ap.add_argument("--lr", type=float, default=lr)
    ap.add_argument("--max-minutes", type=float, default=max_minutes, help="stop starting new epochs after this budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="tiny run on 150k tokens to check the pipeline")
    ap.add_argument("--out-name", default=saved_name)


def load_splits(args):
    train = np.load(args.data / "train.npy").astype(np.int32)
    valid = np.load(args.data / "valid.npy").astype(np.int32)
    vocab_size = len(json.loads((args.saved / "vocab.json").read_text())["itos"])
    if args.quick:
        train, valid = train[:150_000], valid[:20_000]
        args.epochs = 1
    return train, valid, vocab_size


def run_training(model, *, name, args, train, valid, seq_len, extra_meta, extra_callbacks=()):
    """Fit ``model`` (already compiled), keep the best epoch, write meta + history, return the meta dict."""
    import tensorflow as tf

    out_dir = args.saved / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / args.out_name

    train_seq = WindowSequence(train, seq_len, args.batch_size, True, seed=args.seed)
    val_seq = WindowSequence(valid, seq_len, args.batch_size, False)
    model.summary()
    print(f"{len(train_seq)} steps/epoch, batch {args.batch_size} x {seq_len} tokens")

    budget = TimeBudget(args.max_minutes)
    t0 = time.time()
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        verbose=2,
        callbacks=[
            *extra_callbacks,
            keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-4),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            budget,
            # resume support: re-running the same command continues from the last finished epoch
            keras.callbacks.BackupAndRestore(str(out_dir / "_backup"), save_freq="epoch", delete_checkpoint=True),
        ],
    )
    train_seconds = time.time() - t0

    # The checkpoint holds the best epoch incl. optimizer state. Re-save it slim (inference only).
    best_model = keras.models.load_model(model_path, compile=False)
    best_model.save(model_path)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best = int(np.argmin(hist["val_loss"]))
    meta = {
        "model": name,
        "version": config.MODEL_VERSION,
        "framework": f"tensorflow {tf.__version__} / keras {keras.__version__}",
        "vocab_size": int(model.output_shape[-1]),
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs_requested": args.epochs,
        "epochs_run": len(hist["loss"]),
        "best_epoch": best + 1,
        "best_val_loss": round(hist["val_loss"][best], 4),
        "best_val_perplexity_windowed": round(float(np.exp(hist["val_loss"][best])), 2),
        "stopped_by_time_budget": budget.stopped_early,
        "training_seconds": round(train_seconds, 1),
        "training_tokens": int(len(train)),
        "parameters": int(model.count_params()),
        "seed": args.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": model_path.name,
        "size_bytes": model_path.stat().st_size,
        "quick_run": bool(args.quick),
        **extra_meta,
    }
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / f"{name}_history.json").write_text(json.dumps(hist, indent=2))
    print(f"saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB); best epoch {best + 1}")
    return meta
