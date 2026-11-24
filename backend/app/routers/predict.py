from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.errors import ApiError
from app.schemas import (
    CompleteRequest,
    CompleteResponse,
    PredictRequest,
    PredictResponse,
    Suggestion,
)
from app.security import enforce_rate_limit, optional_client_id

router = APIRouter(tags=["prediction"], dependencies=[Depends(enforce_rate_limit)])
log = logging.getLogger("predictai.api")


async def _run(request: Request, fn, *args, **kwargs):
    """Run blocking inference in a worker thread with a server-side timeout."""
    timeout = request.app.state.settings.predict_timeout_seconds
    try:
        return await asyncio.wait_for(run_in_threadpool(fn, *args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        raise ApiError(504, "prediction_timeout", "The model took too long to answer. Your text was not changed.")


def _record(store, event_type, model, owner_id, latency_ms, mode):
    """Aggregate metadata only - the typed text is never passed here."""
    try:
        store.record_event(event_type, model, owner_id, latency_ms, mode)
    except Exception as exc:  # noqa: BLE001 - analytics must never break prediction
        log.warning("could not record %s event: %s", event_type, type(exc).__name__)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Top-k next-word suggestions",
    responses={503: {"description": "Selected model is not available"}, 504: {"description": "Inference timed out"}},
)
async def predict(
    body: PredictRequest,
    request: Request,
    background: BackgroundTasks,
    client_id: str | None = Depends(optional_client_id),
):
    result = await _run(request, request.app.state.engine.predict, body.text, body.model, body.top_k)
    background.add_task(_record, request.app.state.store, "predict", result.model, client_id, result.latency_ms, result.mode)
    return PredictResponse(
        model=result.model,
        model_version=result.model_version,
        mode=result.mode,
        prefix=result.prefix,
        context=result.context_tokens,
        predictions=[Suggestion(word=w, score=round(s, 6)) for w, s in result.predictions],
        latency_ms=round(result.latency_ms, 3),
    )


@router.post(
    "/complete",
    response_model=CompleteResponse,
    summary="Bounded sentence continuation",
    responses={503: {"description": "Selected model is not available"}, 504: {"description": "Inference timed out"}},
)
async def complete(
    body: CompleteRequest,
    request: Request,
    background: BackgroundTasks,
    client_id: str | None = Depends(optional_client_id),
):
    result = await _run(
        request,
        request.app.state.engine.complete,
        body.text,
        body.model,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stop_at_sentence=body.stop_at_sentence,
        seed=body.seed,
    )
    background.add_task(_record, request.app.state.store, "complete", result.model, client_id, result.latency_ms, None)
    return CompleteResponse(
        model=result.model,
        model_version=result.model_version,
        continuation=result.continuation,
        tokens=result.tokens,
        stopped_because=result.stopped_because,
        latency_ms=round(result.latency_ms, 3),
    )
