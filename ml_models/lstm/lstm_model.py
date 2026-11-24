"""LSTM language model (Embedding -> LSTM -> Dense softmax) built with TensorFlow/Keras.

TensorFlow is imported lazily so that the rest of the application (N-gram model, API) keeps working on
machines where TensorFlow is not installed.
"""
from __future__ import annotations

from ml_models.keras_lm import KerasLanguageModel


def build_lstm(vocab_size: int, embed_dim: int = 128, lstm_units: int = 192, dropout: float = 0.3):
    """Sequence-to-sequence next-token model: every time step predicts the next token."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    x = layers.Embedding(vocab_size, embed_dim, name="embedding")(inputs)
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(vocab_size, activation="softmax", name="next_token")(x)
    return keras.Model(inputs, outputs, name="predictai_lstm")


class LSTMModel(KerasLanguageModel):
    model_id = "lstm"
