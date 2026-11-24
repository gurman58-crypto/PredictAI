"""Decoder-only Transformer language model (GPT-style) built with TensorFlow/Keras.

    token ids -> Embedding -> + learned positional embedding -> N x [ LayerNorm -> causal multi-head
    self-attention -> residual ; LayerNorm -> feed-forward (GELU) -> residual ] -> LayerNorm -> Dense softmax

* **Causal mask:** position *t* can only attend to positions <= *t*, so the model predicts the next token
  at every position (exactly like the LSTM) and never sees the future.
* **Pre-LayerNorm** blocks train stably without a long warm-up on a small laptop budget.
* Positions are absolute and start at 0 for the first token of the context window; training windows and
  inference contexts both start at position 0, so they are consistent (max context = ``max_len`` tokens).

TensorFlow/Keras is imported lazily by ``build_transformer`` and ``TransformerModel`` so the rest of the
application does not need it.
"""
from __future__ import annotations

from ml_models.keras_lm import KerasLanguageModel


def _define_layers():
    """Define the custom layer once TensorFlow is available (registers it for model loading)."""
    import keras
    from keras import layers, ops

    @keras.saving.register_keras_serializable(package="predictai")
    class PositionalEmbedding(layers.Layer):
        """Adds a learned embedding for each position 0..max_len-1 to the token embeddings."""

        def __init__(self, max_len: int, d_model: int, **kwargs):
            super().__init__(**kwargs)
            self.max_len = max_len
            self.d_model = d_model

        def build(self, input_shape):
            self.pos = self.add_weight(
                name="pos", shape=(self.max_len, self.d_model),
                initializer=keras.initializers.RandomNormal(stddev=0.02),
            )
            super().build(input_shape)

        def call(self, x):
            steps = ops.shape(x)[1]
            return x + ops.take(self.pos, ops.arange(steps), axis=0)

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"max_len": self.max_len, "d_model": self.d_model})
            return cfg

    return PositionalEmbedding


def register_custom_layers():
    """Idempotent: importing keras serialisation registry entries once is enough."""
    global _POSITIONAL
    if _POSITIONAL is None:
        _POSITIONAL = _define_layers()
    return _POSITIONAL


_POSITIONAL = None


def build_transformer(
    vocab_size: int,
    max_len: int = 30,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    ff_dim: int = 512,
    dropout: float = 0.1,
):
    from tensorflow import keras
    from tensorflow.keras import layers

    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    PositionalEmbedding = register_custom_layers()

    inputs = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    x = layers.Embedding(vocab_size, d_model, name="embedding")(inputs)
    x = PositionalEmbedding(max_len, d_model, name="positions")(x)
    x = layers.Dropout(dropout, name="embed_dropout")(x)

    for i in range(num_layers):
        h = layers.LayerNormalization(epsilon=1e-5, name=f"block{i}_ln1")(x)
        h = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout, name=f"block{i}_attention"
        )(h, h, use_causal_mask=True)
        h = layers.Dropout(dropout, name=f"block{i}_attn_dropout")(h)
        x = layers.Add(name=f"block{i}_add1")([x, h])

        h = layers.LayerNormalization(epsilon=1e-5, name=f"block{i}_ln2")(x)
        h = layers.Dense(ff_dim, activation="gelu", name=f"block{i}_ff1")(h)
        h = layers.Dense(d_model, name=f"block{i}_ff2")(h)
        h = layers.Dropout(dropout, name=f"block{i}_ff_dropout")(h)
        x = layers.Add(name=f"block{i}_add2")([x, h])

    x = layers.LayerNormalization(epsilon=1e-5, name="final_ln")(x)
    outputs = layers.Dense(vocab_size, activation="softmax", name="next_token")(x)
    return keras.Model(inputs, outputs, name="predictai_transformer")


class TransformerModel(KerasLanguageModel):
    model_id = "transformer"

    @classmethod
    def _register_custom_objects(cls) -> None:
        register_custom_layers()
