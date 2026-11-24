# Deployment

## 1. Prepare
- Train the models locally (README steps 2–4) so `ml_models/saved_models/` contains `vocab.json`, `ngram/`, `lstm/` and `transformer/` (the LSTM file is ~15 MB, fine for GitHub).
- Create the Atlas cluster and user (README §10).
- Push the repository to GitHub. `.gitignore` already excludes `.env`, `node_modules`, `dist` and the raw dataset.

## 2. Backend → Render
1. New → **Blueprint** → select the repository (it reads `render.yaml`), or create a *Web Service* manually:
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
2. Environment variables
   | Key | Value |
   |---|---|
   | `MONGODB_URI` | your Atlas connection string |
   | `CORS_ORIGINS` | `https://<your-app>.vercel.app` (add more origins comma-separated) |
   | `ENVIRONMENT` | `production` |
   | `ENABLE_LSTM` | `true`, or `false` on a 512 MB instance |
   | `ENABLE_TRANSFORMER` | `true`, or `false` on a 512 MB instance |
3. After deploy open `https://<service>.onrender.com/` and check `database.connected: true` and the model statuses.
4. Atlas Network Access must allow Render's outbound addresses (or `0.0.0.0/0` with a strong password).

Memory: TensorFlow plus the LSTM needs about 500 MB; use the *Starter* plan or `ENABLE_LSTM=false` and `ENABLE_TRANSFORMER=false` (then remove `tensorflow-cpu` from `backend/requirements.txt` to cut build time and memory). Free instances sleep after inactivity, so the first request can take up to a minute.

## 3. Frontend → Vercel
1. New Project → import the repository → **Root Directory: `frontend`** → Framework: Vite (build `npm run build`, output `dist`).
2. Environment variable: `VITE_API_URL=https://<service>.onrender.com` (no trailing slash). Vite bakes it in at build time, so redeploy after changing it.
3. `frontend/vercel.json` rewrites every path to `index.html` so deep links such as `/editor/abc` work.

## 4. Verify
Open the Vercel URL → the sidebar footer lists the models as ready → type in the editor → save a document → refresh and confirm it persists.

## Docker (optional, untested in the build environment)
`docker compose up --build` starts MongoDB, the API (port 8000) and the frontend on nginx (port 8080).
