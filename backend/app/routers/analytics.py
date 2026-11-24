from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.schemas import AcceptedEvent, AnalyticsResponse, ModelUsage
from app.security import optional_client_id

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", response_model=AnalyticsResponse, summary="Aggregate usage statistics")
def analytics(request: Request, client_id: str | None = Depends(optional_client_id)):
    """Counts and average latency per model.  Scoped to the caller when ``X-Client-Id`` is sent,
    otherwise global.  Only metadata is stored - never the text that was typed."""
    data = request.app.state.store.analytics(client_id)
    return AnalyticsResponse(
        scope="client" if client_id else "global",
        predictions=data["predictions"],
        completions=data["completions"],
        accepted=data["accepted"],
        acceptance_rate=data["acceptance_rate"],
        by_model={k: ModelUsage(**v) for k, v in data["by_model"].items()},
    )


@router.post("/accepted", status_code=204, summary="Record that a suggestion was accepted")
def suggestion_accepted(
    body: AcceptedEvent,
    request: Request,
    background: BackgroundTasks,
    client_id: str | None = Depends(optional_client_id),
):
    background.add_task(request.app.state.store.record_event, "accept", body.model, client_id, None, None)
