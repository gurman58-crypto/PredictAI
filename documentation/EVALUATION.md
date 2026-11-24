# Evaluation protocol

Script: `python -m ml_models.evaluation.evaluate` → `ml_models/saved_models/evaluation_results.json`, `evaluation_summary.csv`, `evaluation_examples.csv`.

## Data
The official WikiText-2 **test** split, never used for training, validation or hyper-parameter selection. The N-gram smoothing constant was chosen on the validation split; the LSTM's best epoch and early stopping use the validation split.

## Targets
Test-stream positions starting at index 20, in blocks of 10 (`EVAL_MIN_CONTEXT`, `EVAL_BLOCK` in `config.py`). **All models are scored on exactly these targets.** For each target the LSTM receives 20–29 preceding tokens (everything up to the target within its window); the N-gram uses only the preceding one or two tokens (its order), as it would at inference.

## Metrics
| Metric | Definition |
|---|---|
| Perplexity | `exp(mean −log p(target))` over **all** targets including `<unk>`; all models output a full distribution over the same 10 000-token vocabulary |
| Top-1 / Top-5 accuracy | share of targets whose true token is ranked 1st / within the top 5, ranking over all tokens except `<pad>`/`<unk>`; `<unk>` targets are skipped |
| Latency | median, mean and 95th percentile of `PredictionEngine.predict` (preprocessing + model + ranking) for 300 sampled contexts after 30 warm-up calls, single request at a time, **no network**. Depends on the machine (recorded in the report) |
| Model size | artifact file size on disk |

N-gram **order variants** (unigram, bigram, trigram) are evaluated on the same targets for the ablation table.

## Caveats
- Numbers are not comparable with published WikiText-2 results (different tokenisation, a lower-cased 10k vocabulary, `<unk>` handling).
- Accuracy is exact-match against one reference word; many other words may be perfectly good suggestions, so absolute accuracy is naturally low for next-word prediction.
- The report states the date, machine and test-set size it was produced with. If a model has no artifact it is reported as `pending` and no numbers are shown.

## Transformer notes

The Transformer is trained with the same windows, vocabulary and validation split as the LSTM and is scored on the identical test targets. It sees up to 29 left-context tokens (like the LSTM). Its validation loss was used only for early stopping and learning-rate reduction; no test data influenced any setting.

## Blend notes

PredictAI Blend is a linear mixture of the N-gram, LSTM and Transformer distributions. Weights were fitted by EM on half of the validation evaluation blocks (about 103,000 targets) and never on test data. It is scored on the identical test targets. Its reported size is the sum of the member artifacts, and its latency includes running every member.
