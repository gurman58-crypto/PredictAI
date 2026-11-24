"""Generic inference wrapper for Keras language models (LSTM and Transformer).

Both models map a batch of token ids ``[B, T]`` to a next-token distribution at every position
``[B, T, V]``.  For prediction only the last position is needed.

TensorFlow is imported lazily so the rest of the application keeps working on machines where it is not
installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_models.preprocessing.text_processing import EOS_ID, PAD_ID


class KerasLanguageModel:
    model_id = "keras"

    def __init__(self, keras_model, vocab_size: int, max_context: int, meta: dict | None = None):
        import tensorflow as tf

        self._tf = tf
        self.keras_model = keras_model
        self.vocab_size = vocab_size
        self.max_context = max_context
        self.meta = meta or {}
        self.version = self.meta.get("version", "v1")
        # fixed signature => the graph is traced once, even for varying context lengths
        self._fn = tf.function(
            lambda x: keras_model(x, training=False)[:, -1, :],
            input_signature=[tf.TensorSpec([1, None], tf.int32)],
        )

    def next_token_probs(self, context_ids) -> np.ndarray:
        ctx = list(context_ids)[-self.max_context :] or [EOS_ID]
        out = self._fn(self._tf.constant([ctx], dtype=self._tf.int32)).numpy()[0].astype(np.float64)
        out[PAD_ID] = 0.0
        return out / out.sum()

    @classmethod
    def load(cls, model_path: Path, meta_path: Path | None = None):
        from tensorflow import keras

        cls._register_custom_objects()
        model = keras.models.load_model(Path(model_path), compile=False)
        meta = {}
        if meta_path and Path(meta_path).exists():
            meta = json.loads(Path(meta_path).read_text())
        vocab_size = int(model.output_shape[-1])
        max_context = int(meta.get("seq_len", 30))
        return cls(model, vocab_size, max_context, meta)

    @classmethod
    def _register_custom_objects(cls) -> None:
        """Hook: subclasses import the module that registers their custom Keras layers."""
