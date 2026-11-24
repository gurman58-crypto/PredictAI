@echo off
REM Download + prepare WikiText-2, train both models, evaluate.
cd /d "%~dp0.."
python -m ml_models.dataset.download_dataset || exit /b 1
python -m ml_models.dataset.prepare_dataset || exit /b 1
python -m ml_models.training.train_ngram || exit /b 1
python -m ml_models.training.train_lstm %* || exit /b 1
python -m ml_models.training.train_transformer %* || exit /b 1
python -m ml_models.training.train_ensemble || exit /b 1
python -m ml_models.evaluation.evaluate || exit /b 1
