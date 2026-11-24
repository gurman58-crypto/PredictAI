# Architecture

```
 React (Vite)  ──Axios──▶  FastAPI  ──▶ PredictionEngine ──▶ N-gram (numpy arrays)
  Editor, Compare           routers       preprocessing        LSTM + Transformer (Keras)
        │                       │
        │                       └──▶ DocumentStore ──▶ MongoDB Atlas (PyMongo)
        └ debounce 250 ms, abort stale requests, keep text on failure
```

## Request flow for a suggestion
1. The editor sends the text **before the cursor** after a 250 ms pause (`usePredictor`), aborting any older in-flight request so a slow answer can never overwrite a newer one.
2. `POST /predict` → rate-limit check → inference in a worker thread with a 5 s timeout.
3. `TextPreprocessor.parse` → context ids (+ optional prefix) → `model.next_token_probs` → mask to word tokens (and prefix) → top-k.
4. The response goes back; only counts and latency are logged in the background.
5. Choosing a suggestion computes an exact edit range (`lib/text.js`) and applies it through the browser's editing command so `Ctrl+Z` works.

## Training flow
`download_dataset` → `prepare_dataset` (tokenise, vocabulary on train only, `.npy` splits) → `train_ngram` / `train_lstm` / `train_transformer` → `evaluate` (same test targets for both). Artifacts live in `ml_models/saved_models/` and are loaded by the API at start-up.

## Design decisions
- **In-process inference.** All models are small (tens of MB); no separate model server is needed.
- **Graceful degradation.** Missing/corrupt artifacts mark that model `unavailable` with a reason; the API and UI keep working.
- **One preprocessing module** for training, evaluation and serving.
- **Store interface** (`app/db/base.py`) with MongoDB and in-memory implementations tested by one shared contract.
- **Honest evaluation.** Metrics are produced only by a script on a held-out split; the UI shows "pending" otherwise.
