"""FastAPI application factory.

Run locally:  uvicorn app.main:app --reload --port 8000     (from the backend/ folder)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.db.factory import create_store
from app.errors import register_exception_handlers
from app.routers import analytics, documents, predict, system, writing
from app.security import RateLimiter

from ml_models.inference import PredictionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("predictai.api")

DESCRIPTION = """
**PredictAI** - next-word prediction and sentence continuation served by three independently trained
language models (an N-gram baseline, an LSTM and a Transformer) plus their blend, and writing tools:
grammar and spelling checks, style suggestions and a paraphraser (`/writing/*`).

* `/predict` and `/complete` run the real trained models - no hard-coded or random suggestions.
* `/documents` stores each browser's documents in MongoDB (ownership via the `X-Client-Id` header).
* `/evaluation` returns *measured* metrics from the shared held-out test split, or `pending`.

Suggestion scores are model probabilities, not calibrated confidence.
"""


def create_app(settings: Settings | None = None, engine=None, store=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.engine = engine or PredictionEngine(
            settings.saved_models_dir, enable_lstm=settings.enable_lstm, enable_transformer=settings.enable_transformer
        )
        app.state.store = store or create_store(settings)
        app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
        for s in app.state.engine.statuses():
            log.info("model %-5s %s%s", s.id, s.status, f" ({s.reason})" if s.reason else "")
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    if "*" in settings.cors_origins and settings.is_production:
        log.warning("CORS_ORIGINS contains '*' in production - restrict it to your frontend origin")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Client-Id"],
        expose_headers=["Retry-After"],
        max_age=600,
    )
    register_exception_handlers(app)
    app.include_router(system.router)
    app.include_router(predict.router)
    app.include_router(documents.router)
    app.include_router(analytics.router)
    app.include_router(writing.router)
    return app


app = create_app()
