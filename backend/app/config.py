"""Runtime configuration, read from environment variables (see backend/.env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    app_name: str = "PredictAI API"
    version: str = "1.0.0"
    environment: str = "development"
    cors_origins: tuple[str, ...] = DEFAULT_DEV_ORIGINS
    mongodb_uri: str | None = None
    mongodb_db: str = "predictai"
    saved_models_dir: Path | None = None
    enable_lstm: bool = True
    enable_transformer: bool = True
    rate_limit_per_minute: int = 300          # per client IP, /predict and /complete; 0 disables
    predict_timeout_seconds: float = 5.0
    max_documents_per_client: int = 200
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        origins_raw = os.environ.get("CORS_ORIGINS", "").strip()
        origins = tuple(o.strip().rstrip("/") for o in origins_raw.split(",") if o.strip()) or DEFAULT_DEV_ORIGINS
        saved = os.environ.get("SAVED_MODELS_DIR", "").strip()
        return cls(
            environment=os.environ.get("ENVIRONMENT", "development").strip().lower(),
            cors_origins=origins,
            mongodb_uri=os.environ.get("MONGODB_URI", "").strip() or None,
            mongodb_db=os.environ.get("MONGODB_DB", "predictai").strip() or "predictai",
            saved_models_dir=Path(saved) if saved else None,
            enable_lstm=_env_bool("ENABLE_LSTM", True),
            enable_transformer=_env_bool("ENABLE_TRANSFORMER", True),
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 300),
            predict_timeout_seconds=_env_float("PREDICT_TIMEOUT_SECONDS", 5.0),
            max_documents_per_client=_env_int("MAX_DOCUMENTS_PER_CLIENT", 200),
        )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"
