#!/usr/bin/env bash
# Build the four deliverable archives into <project>/../dist (or $1).
#   PredictAI_Frontend.zip, PredictAI_Backend.zip, PredictAI_ML_Models.zip, PredictAI_Complete_Project.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/../dist}"

# refuse to ship a half-trained LSTM/Transformer (a checkpoint without the metadata written at the end of training)
for M in lstm transformer; do
  D="$ROOT/ml_models/saved_models/$M"
  if [ -f "$D/${M}_model.keras" ] && [ ! -f "$D/${M}_meta.json" ]; then
    echo "ERROR: $D/${M}_model.keras has no ${M}_meta.json - training has not finished. Finish training or delete the file." >&2
    exit 1
  fi
done
STAGE="$(mktemp -d)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
trap 'rm -rf "$STAGE"' EXIT

# never ship: dependencies, build output, caches, the downloaded dataset, training scratch, real secrets
EXCLUDES=(--exclude=node_modules --exclude=dist --exclude=__pycache__ --exclude='*.pyc' --exclude=.pytest_cache
          --exclude=.vite --exclude='ml_models/dataset/data' --exclude=_backup --exclude='*.log'
          --exclude=.env --exclude='.env.local' --exclude=.DS_Store --exclude=.venv --exclude=venv)

copy() {  # copy <relative path> <destination dir>   (keeps the relative layout)
  (cd "$ROOT" && tar "${EXCLUDES[@]}" -cf - "$1") | (mkdir -p "$2" && cd "$2" && tar -xf -)
}

make_zip() {  # make_zip <stage folder name>
  rm -f "$OUT/$1.zip"
  (cd "$STAGE" && zip -qr "$OUT/$1.zip" "$1")
  echo "built $OUT/$1.zip ($(du -h "$OUT/$1.zip" | cut -f1))"
}

# ---------------------------------------------------------------- Frontend
F="$STAGE/PredictAI_Frontend"; mkdir -p "$F"
(cd "$ROOT/frontend" && tar "${EXCLUDES[@]}" -cf - .) | (cd "$F" && tar -xf -)
cat > "$F/README.md" <<'EOF'
# PredictAI - Frontend

React 18 + Vite + Tailwind CSS + Lucide React + Axios.

```bash
npm install
cp .env.example .env        # set VITE_API_URL (default http://localhost:8000)
npm run dev                 # http://localhost:5173
npm test                    # unit tests for the text-insertion logic
npm run build               # production build into dist/
```

It needs the PredictAI backend running (see the backend archive or the complete project README).
Deploy to Vercel: set Root Directory to this folder and `VITE_API_URL` to your API URL.
EOF
make_zip PredictAI_Frontend

# ---------------------------------------------------------------- Backend (+ the inference runtime it imports)
B="$STAGE/PredictAI_Backend"; mkdir -p "$B"
copy backend "$B"
for f in __init__.py config.py inference.py testing.py keras_lm.py preprocessing ngram lstm transformer ensemble writing saved_models; do copy "ml_models/$f" "$B"; done
copy documentation/API.md "$B"; copy documentation/SECURITY.md "$B"; copy documentation/DATABASE.md "$B"
copy render.yaml "$B"; copy .gitignore "$B"
cp "$ROOT/backend/requirements.txt" "$B/requirements.txt"
cat > "$B/pytest.ini" <<'EOF'
[pytest]
testpaths = backend/tests
pythonpath = . backend
filterwarnings =
    ignore::DeprecationWarning
EOF
cat > "$B/README.md" <<'EOF'
# PredictAI - Backend

FastAPI service that serves the trained N-gram and LSTM models (`ml_models/`, runtime part only) and stores
documents in MongoDB. Training code is in the ML Models / Complete archive; the trained artifacts are in
`ml_models/saved_models/`.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd backend
cp .env.example .env            # set MONGODB_URI (optional) and CORS_ORIGINS
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/ (status) and http://localhost:8000/docs (Swagger).
Tests (from this folder): `pytest`.
Variables: see `backend/.env.example`. API reference: `API.md`. Deployment on Render: `render.yaml`.
EOF
make_zip PredictAI_Backend

# ---------------------------------------------------------------- ML models
M="$STAGE/PredictAI_ML_Models"; mkdir -p "$M"
copy ml_models "$M"
copy documentation/PREPROCESSING.md "$M"; copy documentation/EVALUATION.md "$M"
mkdir -p "$M/scripts"; cp "$ROOT/scripts/train_all.sh" "$ROOT/scripts/train_all.bat" "$M/scripts/"
cp "$ROOT/ml_models/requirements.txt" "$M/requirements.txt"
cp "$ROOT/.gitignore" "$M/.gitignore"
cat > "$M/pytest.ini" <<'EOF'
[pytest]
testpaths = ml_models/tests
pythonpath = .
filterwarnings =
    ignore::DeprecationWarning
EOF
cat > "$M/README.md" <<'EOF'
# PredictAI - ML Models

Dataset scripts, shared preprocessing, N-gram and LSTM models, training and evaluation.
Run everything from this folder.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ml_models.dataset.download_dataset          # WikiText-2 (CC BY-SA 3.0), not bundled
python -m ml_models.dataset.prepare_dataset
python -m ml_models.training.train_ngram              # seconds
python -m ml_models.training.train_lstm               # ~4 min/epoch on 2 CPU cores; resumable; --max-minutes N
python -m ml_models.training.train_transformer        # ~4 min/epoch; resumable; --max-minutes N
python -m ml_models.training.train_ensemble           # ~2 min; fits the blend weights on validation data
python -m ml_models.evaluation.evaluate               # writes ml_models/saved_models/evaluation_results.json
pytest                                                # ML unit tests
```
Trained artifacts that were available when this archive was built are in `ml_models/saved_models/`.
Details: `PREPROCESSING.md`, `EVALUATION.md`.
EOF
make_zip PredictAI_ML_Models

# ---------------------------------------------------------------- Complete project
C="$STAGE/PredictAI"; mkdir -p "$C"
for f in frontend backend ml_models documentation scripts README.md .gitignore .env.example docker-compose.yml render.yaml requirements.txt pytest.ini; do copy "$f" "$C"; done
make_zip PredictAI
mv "$OUT/PredictAI.zip" "$OUT/PredictAI_Complete_Project.zip"
echo "built $OUT/PredictAI_Complete_Project.zip"
