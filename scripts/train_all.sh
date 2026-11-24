#!/usr/bin/env bash
# Download + prepare WikiText-2, train both models, evaluate. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ml_models.dataset.download_dataset
python -m ml_models.dataset.prepare_dataset
python -m ml_models.training.train_ngram
python -m ml_models.training.train_lstm "$@"      # e.g. --epochs 10 --max-minutes 60
python -m ml_models.training.train_transformer "$@"
python -m ml_models.training.train_ensemble      # fits the blend weights on validation data (about 2 min)
python -m ml_models.evaluation.evaluate
