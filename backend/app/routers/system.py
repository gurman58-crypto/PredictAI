from __future__ import annotations

import json

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from app.schemas import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/", tags=["system"], summary="Service information and health")
async def root(request: Request):
    state = request.app.state
    connected = await run_in_threadpool(state.store.ping)
    models = {s.id: s.status for s in state.engine.statuses()}
    return {
        "name": state.settings.app_name,
        "version": state.settings.version,
        "status": "ok" if state.engine.any_ready() else "degraded",
        "environment": state.settings.environment,
        "database": {"backend": state.store.backend, "connected": connected},
        "models": models,
        "docs": "/docs",
    }


@router.get("/health", tags=["system"], summary="Lightweight liveness probe (used by Render)")
def health(request: Request):
    return {"status": "ok" if request.app.state.engine.any_ready() else "degraded"}


@router.get("/models", response_model=ModelsResponse, tags=["models"], summary="List models and their status")
def list_models(request: Request):
    engine = request.app.state.engine
    infos = [ModelInfo(**s.__dict__) for s in engine.statuses()]
    default = next((m.id for m in infos if m.status == "ready"), None)
    return ModelsResponse(models=infos, default_model=default)


@router.get("/evaluation", tags=["models"], summary="Measured evaluation results (same held-out test split)")
def evaluation(request: Request):
    """Returns the report written by ``python -m ml_models.evaluation.evaluate``.

    If the models have not been evaluated yet the response is ``{"status": "pending"}`` - results are
    never invented.
    """
    path = request.app.state.engine.saved_dir / "evaluation_results.json"
    if not path.exists():
        return {
            "status": "pending",
            "message": "No evaluation has been run yet. Train the models, then run: "
            "python -m ml_models.evaluation.evaluate",
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "pending", "message": "The evaluation report could not be read; re-run the evaluation."}
    return {"status": "available", **report}
