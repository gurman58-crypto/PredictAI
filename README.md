# PredictAI: Intelligent AI Writing Assistant

PredictAI suggests the next word while you type, shows the five most likely words, and can continue a sentence for you. It is a complete, working machine-learning application: three **independently trained language models** (an N-gram, an LSTM and a Transformer) plus **PredictAI Blend**, a weighted mixture of the three that is the default, and **writing tools** (grammar check, style suggestions, paraphrasing) served by a FastAPI backend, a React editor, MongoDB document storage, and an evaluation dashboard that shows **measured** results only.

> **Honesty rules built into the project**
> - There are no random, hard-coded or external-API suggestions. Every suggestion is computed by a model trained by the scripts in this repo.
> - The Compare page shows numbers only after `python -m ml_models.evaluation.evaluate` has really been run. Until then it says **"Evaluation pending"**.
> - A model that has not been trained is shown as **"not trained"** in the UI and API; it is never faked.

---

## 1. Features

| Area | What you get |
|---|---|
| Editor | Large text area, live suggestions after a 250 ms pause, 5 ranked suggestions with probability bars, insertion at the cursor (also mid-word completion), `Alt+1..5` shortcuts, **Continue sentence**, word and character counters, manual save (`Ctrl+S`), undo of insertions, text is never altered when a request fails or times out |
| Models | N-gram (unigram/bigram/trigram, interpolated absolute-discount smoothing with back-off) LSTM (Embedding → LSTM → Dense softmax) and a decoder-only Transformer (causal multi-head self-attention, TensorFlow/Keras); switch models at any time |
| Writing tools | **Live grammar and spell check**: mistakes get a red wavy underline while you type (after a short pause), click the word for ranked fixes (spelling suggestions are ranked by the language model), *Ignore*, *Add to dictionary*, *Fix all*; checks spelling against a 160,000-word dictionary plus repeated words, a/an, subject–verb agreement, capitals, punctuation; **style suggestions** (long sentences, passive voice, filler and wordy phrases, formal-tone check) with readability statistics; **paraphraser** that offers three alternative wordings of the selected sentence, each substitution checked for fluency by the trained language model |
| Comparison | Top-1 / Top-5 accuracy, perplexity, latency (median and p95) and model size, all measured on the **same held-out test split**; side-by-side live suggestions |
| Documents | Create, list, search, open, update, delete; stored in MongoDB with created/updated timestamps; scoped to the browser that created them |
| Backend | FastAPI, Pydantic validation, consistent JSON errors, CORS, rate limiting, OpenAPI docs at `/docs` |
| Privacy | Typed text is never written to logs or the database unless you press **Save**. Usage analytics store counts and timings only |
| UI | Landing page, dashboard, editor, my documents, model comparison, help; navy/white SaaS design, responsive (desktop/tablet/mobile), loading / empty / success / error states |

## 2. Folder structure

```
PredictAI/
├── frontend/                  React + Vite + Tailwind + Lucide + Axios
│   ├── src/
│   │   ├── pages/             Landing, Dashboard, Editor, Documents, Compare, About
│   │   ├── components/        AppShell, SuggestionBar, ModelSelector, HeroDemo, ui
│   │   ├── context/           Models (status/selection), Toast
│   │   ├── hooks/             usePredictor (debounce + abort stale requests)
│   │   └── lib/               api.js (Axios), text.js (+ tests), clientId.js
│   ├── .env.example  Dockerfile  nginx.conf  vercel.json  package.json
├── backend/                   FastAPI service
│   ├── app/
│   │   ├── main.py            application factory, CORS, routers
│   │   ├── routers/           system, predict, documents, analytics
│   │   ├── db/                mongo.py (PyMongo), memory.py (fallback), base.py, factory.py
│   │   ├── schemas.py         Pydantic models
│   │   ├── security.py        client id (ownership) + rate limiter
│   │   ├── errors.py  config.py
│   ├── tests/                 API + store + security tests
│   ├── requirements.txt  .env.example  Dockerfile
├── ml_models/
│   ├── config.py              the single source of truth for model/eval settings
│   ├── preprocessing/         text_processing.py: normaliser, tokenizer, vocabulary (shared by training AND API)
│   ├── dataset/               download_dataset.py, prepare_dataset.py   (data/ is created here, git-ignored)
│   ├── ngram/                 ngram_model.py
│   ├── keras_lm.py            shared wrapper for the Keras models
│   ├── lstm/                  lstm_model.py
│   ├── transformer/           transformer_model.py
│   ├── ensemble/              ensemble_model.py (the Blend)
│   ├── writing/               grammar.py, style.py, paraphrase.py, lexicon.py
│   ├── training/              train_ngram.py, train_lstm.py, train_transformer.py, train_ensemble.py, keras_common.py
│   ├── evaluation/            evaluate.py
│   ├── inference.py           PredictionEngine used by the API and the evaluation script
│   ├── saved_models/          vocab.json, ngram/, lstm/, transformer/, ensemble/, evaluation_results.json, evaluation_summary.csv
│   └── tests/                 preprocessing, N-gram, LSTM, Transformer, inference, evaluation tests
├── documentation/             API.md, ARCHITECTURE.md, PREPROCESSING.md, EVALUATION.md, DATABASE.md, SECURITY.md, DEPLOYMENT.md
├── scripts/                   train_all.sh / train_all.bat, package_zips.sh
├── docker-compose.yml  render.yaml  requirements.txt  pytest.ini  .env.example  .gitignore  README.md
```

## 3. Technologies

Frontend: React 18, Vite 5, Tailwind CSS 3, Lucide React, Axios, React Router, Vitest.
Backend: Python, FastAPI, Pydantic v2, PyMongo, Uvicorn.
Machine learning: NumPy, Pandas, NLTK (tokenizer), TensorFlow/Keras (LSTM, Transformer), WikiText-2.
Database: MongoDB (Atlas or local).

## 4. Prerequisites

- **Python 3.10 – 3.12** (3.13 also worked with TensorFlow 2.21), pip
- **Node.js 18+** and npm
- A **MongoDB Atlas** account (free tier) *or* local MongoDB. Optional: the app runs without it using an in-memory store (documents are lost on restart; `GET /` shows `"backend": "memory"`).
- About 2 GB free disk and a few GB RAM. No GPU is needed.

## 5. Quick start (copy-paste)

All Python commands are run from the **project root** (`PredictAI/`).

```bash
# 1. Python environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Dataset  (downloads WikiText-2, then tokenises it and builds the vocabulary)
python -m ml_models.dataset.download_dataset
python -m ml_models.dataset.prepare_dataset

# 3. Train  (N-gram: seconds; LSTM: see "LSTM training" for time)
python -m ml_models.training.train_ngram
python -m ml_models.training.train_lstm
python -m ml_models.training.train_transformer
python -m ml_models.training.train_ensemble     # fit the blend weights on validation data

# 4. Evaluate all models on the same test split
python -m ml_models.evaluation.evaluate

# 5. Backend (terminal 1)
cd backend
cp .env.example .env                 # Windows: copy .env.example .env
uvicorn app.main:app --reload --port 8000

# 6. Frontend (terminal 2)
cd frontend
cp .env.example .env
npm install
npm run dev                          # http://localhost:5173
```

`scripts/train_all.sh` (or `train_all.bat` on Windows) runs steps 2–4.

If this download already contains trained artifacts in `ml_models/saved_models/`, you can skip steps 2–4 and go straight to the backend and frontend. The evaluation (step 4) needs the dataset files, so run step 2 if you want to re-measure.

## 6. Dataset download instructions

`python -m ml_models.dataset.download_dataset` downloads **WikiText-2 (word-level)** into `ml_models/dataset/data/raw/`. WikiText-2 is Wikipedia text released under **CC BY-SA 3.0** (Merity et al., 2016); it is downloaded, not bundled.

| Option | Command |
|---|---|
| Default (GitHub mirror) | `python -m ml_models.dataset.download_dataset` |
| Official Hugging Face copy (needs `pip install pyarrow`) | `... --source huggingface` |
| Files you already have (`train.txt`, `valid.txt`, `test.txt`) | `... --source local --path /path/to/folder` |

`prepare_dataset` then (a) removes WikiText-specific artefacts, (b) tokenises with the **same tokenizer the API uses**, (c) builds a 10,000-word vocabulary from the **training split only**, (d) encodes all three splits and (e) writes `saved_models/vocab.json`. About 9–12% of tokens are outside the vocabulary and become `<unk>`; those are never suggested.

## 7. N-gram training

```bash
python -m ml_models.training.train_ngram
```
Takes a few seconds. It counts unigrams, bigrams and trigrams on the training split (all of them are kept; pruning rare trigrams was tested and made the model worse), then picks the smoothing constant D on the **validation** split (never on test) and saves `saved_models/ngram/ngram_model.npz` plus `ngram_meta.json`.

## 8. LSTM training

```bash
python -m ml_models.training.train_lstm                 # default: up to 8 epochs, 30-40 min budget
python -m ml_models.training.train_lstm --quick         # ~30 s pipeline check, NOT a usable model
python -m ml_models.training.train_lstm --epochs 15 --max-minutes 90   # better model, longer
```

Architecture: `Embedding(10 000, 128) → LSTM(192) → Dropout(0.2) → Dense(10 000, softmax)`, about 3.5 M parameters, 30-token windows, Adam with gradient clipping, learning-rate reduction on plateau and early stopping on validation loss. On a 2-core CPU one epoch takes about 4 minutes; a typical student laptop is in the same range. `--max-minutes` stops starting new epochs once the budget is used.

**Training is resumable.** If it is interrupted (laptop sleep, closed terminal), run the same command again and it continues from the last finished epoch.

Outputs: `saved_models/lstm/lstm_model.keras`, `lstm_meta.json` (epochs, parameters, training time) and `lstm_history.json`.

> More epochs generally give a lower perplexity. Do not quote any accuracy figure from this README or from a previous run: run the evaluation and use what it prints.

## 8b. Transformer training

```bash
python -m ml_models.training.train_transformer                 # default: up to 8 epochs, 45 min budget
python -m ml_models.training.train_transformer --quick         # pipeline check only, NOT a usable model
python -m ml_models.training.train_transformer --layers 3 --epochs 12 --max-minutes 90
```

Architecture: token embedding + learned positional embedding (30 positions) → 2 pre-LayerNorm decoder blocks (4-head causal self-attention, GELU feed-forward 512) → LayerNorm → `Dense(10 000, softmax)`, about 3.0 M parameters. Training uses a 400-step learning-rate warm-up, reduction on plateau, early stopping and a resumable checkpoint (re-run the same command if interrupted). Outputs: `saved_models/transformer/transformer_model.keras`, `transformer_meta.json`, `transformer_history.json`. Set `ENABLE_TRANSFORMER=false` in `backend/.env` to skip loading it.

## 8c. PredictAI Blend (default model)

`python -m ml_models.training.train_ensemble` (about 2 minutes) fits one weight per model so that `p(word) = w1·p_ngram + w2·p_lstm + w3·p_transformer`. The weights are found by maximum likelihood (EM) on the **validation** split only and stored in `saved_models/ensemble/ensemble_meta.json`; the test split is never used for tuning. If a member model is missing or disabled, the remaining weights are re-normalised. The blend is evaluated on the same test targets as the other models; its latency is roughly the sum of its members'.

## 8d. Writing tools (grammar, style, paraphrase)

The editor has a **Writing tools** panel under the text area; the same functions are available at `POST /writing/grammar`, `/writing/style` and `/writing/paraphrase` (see `documentation/API.md`).

* **Live checking:** the editor sends the text to `POST /writing/check` about 0.7 s after you stop typing (one request returns grammar and style). Results are kept aligned with your text while you keep typing; the word you are still typing is never flagged. Underlines are a transparent copy of the text drawn behind the real text box, so typing, selection and Ctrl+Z behave normally. Press `Alt+Enter` on an underlined word to open the fix menu from the keyboard. "Add to dictionary" is stored only in your browser. Browsers' own spell-check is switched off while live checking is on, and back on if you turn it off.
* **Spelling** uses a bundled English word list with frequencies (`ml_models/writing/data/`, derived from the MIT-licensed `pyspellchecker` word list, licence included) plus the language models' vocabulary. British spellings (colour, organise, centre) are accepted. Proper nouns, ALL-CAPS acronyms, camelCase, web addresses and file names are not flagged. Suggestions are the known words one or two edits away, ranked by how common they are and how well they fit the words before them (language model).
* **Grammar** and **Style** are transparent *rule-based* checkers (`ml_models/writing/grammar.py`, `style.py`). They find common, clearly identifiable problems and propose a fix; they are not full grammar parsers and will miss many errors. Style output is advice, not correction.
* **Paraphrase** (`paraphrase.py`, `lexicon.py`) is *not* a neural rewriter. It finds words and phrases that have an entry in a hand-written substitution lexicon, asks the trained language model how natural each alternative is in context (the words before it and the next two words after it), keeps only alternatives that score close to the original wording, and offers up to three variants. It changes vocabulary and phrasing, not sentence structure.
* No training is needed for these tools. Like the prediction endpoints they never log or store the text you send. Paraphrase works without a trained model but then cannot check fluency, and says so.

## 9. Evaluation

```bash
python -m ml_models.evaluation.evaluate
```
Writes `saved_models/evaluation_results.json` and `evaluation_summary.csv` (+ `evaluation_examples.csv` with qualitative examples). The backend serves the JSON at `GET /evaluation`; the **Compare models** page renders it. Protocol (details in `documentation/EVALUATION.md`):

- all models are scored on **exactly the same test targets** from the held-out WikiText-2 test split;
- perplexity uses all targets; top-1/top-5 skip targets that are `<unk>`;
- latency is end-to-end `predict()` time for the same sampled contexts after a warm-up (no HTTP);
- model size is the artifact size on disk.

Note one genuine difference the page also states: the LSTM and the Transformer see up to 29 words of left context, the N-gram only the last two.

## 10. MongoDB configuration

1. Create a free cluster at <https://www.mongodb.com/atlas>.
2. **Database Access → Add user** (username + password, role *Read and write to any database*).
3. **Network Access → Add IP address**: your IP for local development (for Render use the Render outbound IPs, or `0.0.0.0/0` only if you accept the risk, with a strong password).
4. **Connect → Drivers** and copy the connection string, e.g. `mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`. URL-encode special characters in the password.
5. Put it in `backend/.env`: `MONGODB_URI=mongodb+srv://...` and keep `MONGODB_DB=predictai`.
6. Restart the backend and open <http://localhost:8000/>: it must show `"database": {"backend": "mongodb", "connected": true}`.

Collections are created automatically: `documents` and `prediction_events` (counts/timings only), with indexes on `owner_id` + time. A local server works too: `MONGODB_URI=mongodb://localhost:27017`. See `documentation/DATABASE.md`.

## 11. Running the application

1. Backend: `cd backend && uvicorn app.main:app --reload --port 8000` → API docs at <http://localhost:8000/docs>.
2. Frontend: `cd frontend && npm run dev` → <http://localhost:5173>.
3. The sidebar footer shows each model's status. Open **Editor**, click into the text area and type.

Docker alternative (not run in the environment where this project was generated): `docker compose up --build` → <http://localhost:8080>.

## 12. API documentation

Interactive: `/docs` (Swagger) and `/redoc`. Full reference with examples: `documentation/API.md`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info, database backend, model status |
| GET | `/health` | Liveness probe |
| GET | `/models` | Models with `status` (`ready` / `unavailable`), version, reason |
| POST | `/predict` | `{text, model, top_k}` → ranked suggestions (next word or mid-word completion) |
| POST | `/complete` | `{text, model, max_tokens, temperature, ...}` → bounded continuation |
| GET | `/documents` | List the caller's documents (newest first, paginated) |
| POST | `/documents` | Create |
| GET | `/documents/{id}` | Read |
| PUT | `/documents/{id}` | Update title and/or content |
| DELETE | `/documents/{id}` | Delete |
| GET | `/analytics` | Aggregate counts and latency (no text) |
| POST | `/analytics/accepted` | Record that a suggestion was accepted |
| GET | `/evaluation` | Measured evaluation report or `{"status":"pending"}` |

Document routes need the header `X-Client-Id` (the frontend creates one per browser). Errors always look like `{"error": {"code": "...", "message": "..."}}`.

## 13. Testing

```bash
pytest                       # ML + backend tests (about 150 tests, ~10 s)
pytest ml_models/tests       # only ML
pytest backend/tests         # only API
cd frontend && npm test      # frontend text/insertion logic
cd frontend && npm run build # production build check
```
Tests build tiny real artifacts from a toy corpus, so they do not need the dataset or a long training run. The MongoDB store is tested against `mongomock` (a MongoDB-compatible in-process fake) with the same behaviour contract as the in-memory store; connect to a real Atlas cluster to exercise it against the real server.

## 14. Deployment

**Backend on Render** (details in `documentation/DEPLOYMENT.md`)
1. Push this repository to GitHub, **including** `ml_models/saved_models/` (the trained artifacts).
2. Render → New → Blueprint (uses `render.yaml`) or New Web Service with *Build* `pip install -r backend/requirements.txt` and *Start* `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Environment variables: `MONGODB_URI`, `CORS_ORIGINS=https://<your-app>.vercel.app`, `ENVIRONMENT=production`.
4. TensorFlow needs roughly 500 MB RAM. On Render's free 512 MB instance set `ENABLE_LSTM=false` and `ENABLE_TRANSFORMER=false` and `tensorflow-cpu` can be removed from `backend/requirements.txt`; the N-gram model keeps working and the UI shows the neural models as unavailable. The free plan also sleeps when idle, so the first request after a pause is slow.

**Frontend on Vercel**
1. Import the repository, set **Root Directory** to `frontend`, framework *Vite*.
2. Environment variable `VITE_API_URL=https://<your-service>.onrender.com`.
3. Deploy. `vercel.json` already rewrites all routes to `index.html`.
4. Add the Vercel URL to the backend's `CORS_ORIGINS` and redeploy the backend.

## 15. Common errors and troubleshooting

| Symptom | Fix |
|---|---|
| UI says "Server offline" / "Cannot reach the PredictAI server" | Backend not running, or `VITE_API_URL` wrong. Restart Vite after editing `.env` |
| Browser console: CORS error | Add the exact frontend origin (scheme + host + port, no trailing slash) to `CORS_ORIGINS` and restart the backend |
| A model shows "not trained" | Run steps 2–3; restart the backend; the reason is shown by `GET /models` |
| `vocab.json not found` | Run `python -m ml_models.dataset.prepare_dataset` |
| `ModuleNotFoundError: ml_models` | Run Python commands from the project root; for the API start uvicorn inside `backend/` (it adds the project root itself) or set `PREDICTAI_ML_ROOT` |
| Download fails / 403 | Use `--source local` with files you downloaded, or `--source huggingface` |
| `database: memory` on `GET /` | `MONGODB_URI` is empty or not loaded; check `backend/.env` |
| `database_unavailable` (503) | Wrong credentials, or your IP is not in Atlas Network Access |
| LSTM / Transformer training is slow | Use `--max-minutes`, fewer `--epochs`, or `--batch-size 256`; or use only the N-gram |
| TensorFlow install fails | Use Python 3.10–3.12 and `pip install tensorflow-cpu` |
| Documents vanished | They are tied to the browser's anonymous id; clearing site data creates a new library. Without MongoDB they are lost on restart |
| 429 Too many requests | Raise `RATE_LIMIT_PER_MINUTE` |

## 16. Limitations (read before your viva)

- Models learned from **Wikipedia only**: they are fluent for encyclopedic prose and weak for chat, code or names. Suggestions are the most probable words, not facts.
- The vocabulary is 10,000 words; rarer words become `<unk>` and are never suggested.
- "Ownership" is an anonymous per-browser id, **not** user login (see `documentation/SECURITY.md`). Do not store sensitive information.
- Perplexity compares models only on the same vocabulary and test targets, as done here; it is not comparable with published WikiText-2 numbers that use a different vocabulary/tokenisation.
- Continuation is sampled from the model's top words, so it is plausible rather than correct, and it can repeat or ramble.

## 17. Licence and attribution

Code: your project. Data: WikiText-2 © Merity et al., CC BY-SA 3.0 (Wikipedia). Cite: Merity, Xiong, Bradbury, Socher (2016), *Pointer Sentinel Mixture Models*.
