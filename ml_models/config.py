"""Central configuration for the PredictAI ML pipeline.

Everything that must be identical between training, evaluation and inference
(vocabulary size, special tokens, evaluation protocol) lives here or in
``preprocessing/text_processing.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

DATA_DIR = ML_ROOT / "dataset" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Trained artifacts (vocabulary, N-gram counts, LSTM weights, evaluation report).
# Can be overridden so that the backend / tests can point at another folder.
SAVED_DIR = Path(os.environ.get("PREDICTAI_SAVED_MODELS", ML_ROOT / "saved_models"))

MODEL_VERSION = "v1"
DATASET_NAME = "WikiText-2 (v1, word-level)"
DATASET_LICENSE = "CC BY-SA 3.0 (Merity et al., 2016 - Wikipedia text)"

# ---- vocabulary -----------------------------------------------------------
MAX_VOCAB = 10_000  # including the 3 special tokens
MIN_FREQ = 2

# ---- N-gram ---------------------------------------------------------------
NGRAM_ORDER = 3
NGRAM_DISCOUNT_GRID = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
NGRAM_MIN_COUNT_BIGRAM = 1
NGRAM_MIN_COUNT_TRIGRAM = 1  # keep ALL trigrams: pruning singletons understated the back-off mass (validation perplexity 171 vs 136)

# ---- LSTM -----------------------------------------------------------------
LSTM_SEQ_LEN = 30
LSTM_EMBED_DIM = 128
LSTM_UNITS = 192
LSTM_DROPOUT = 0.2

# ---- Transformer ----------------------------------------------------------
TRANSFORMER_SEQ_LEN = 30       # also the number of learned positions
TRANSFORMER_D_MODEL = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF = 512
TRANSFORMER_DROPOUT = 0.1

# ---- inference ------------------------------------------------------------
MAX_CONTEXT_TOKENS = 30   # how many trailing tokens are fed to a model
MAX_INPUT_CHARS = 2000    # only the tail of the text is ever looked at

# ---- evaluation protocol (identical for every model) -----------------------
# Test targets are test-stream positions [EVAL_MIN_CONTEXT, ...) in blocks of
# EVAL_BLOCK.  Every target therefore has at least EVAL_MIN_CONTEXT tokens of
# left context available to the LSTM; the N-gram uses its own (shorter) window.
EVAL_MIN_CONTEXT = 20
EVAL_BLOCK = 10
LATENCY_SAMPLES = 300
LATENCY_WARMUP = 30
