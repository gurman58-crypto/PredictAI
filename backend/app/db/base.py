"""Storage interface shared by the MongoDB implementation and the in-memory fallback."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Protocol

DOC_ID_RE = re.compile(r"^[0-9a-f]{24}$")
PREVIEW_CHARS = 160


def utcnow() -> datetime:
    """Current UTC time truncated to milliseconds - MongoDB stores datetimes at that precision,
    so truncating here keeps what we return identical to what is stored."""
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=(now.microsecond // 1000) * 1000)


def count_words(text: str) -> int:
    return len(text.split())


def make_preview(text: str) -> str:
    return " ".join(text.split())[:PREVIEW_CHARS]


def to_document(raw: dict) -> dict:
    """Normalise a stored record into the public document shape."""
    return {
        "id": raw["id"],
        "title": raw["title"],
        "content": raw["content"],
        "word_count": raw["word_count"],
        "char_count": raw["char_count"],
        "created_at": raw["created_at"],
        "updated_at": raw["updated_at"],
    }


def to_summary(raw: dict) -> dict:
    doc = to_document(raw)
    content = doc.pop("content")
    doc["preview"] = make_preview(content)
    return doc


def summarise_events(rows: list[tuple[str, str, int, float | None]]) -> dict:
    """rows = (event type, model, count, avg latency) -> API analytics payload (without scope)."""
    by_model: dict[str, dict] = {}
    for etype, model, count, avg_latency in rows:
        usage = by_model.setdefault(
            model, {"predictions": 0, "completions": 0, "accepted": 0, "avg_latency_ms": None}
        )
        if etype == "predict":
            usage["predictions"] += count
            usage["avg_latency_ms"] = round(avg_latency, 2) if avg_latency is not None else None
        elif etype == "complete":
            usage["completions"] += count
        elif etype == "accept":
            usage["accepted"] += count
    predictions = sum(u["predictions"] for u in by_model.values())
    accepted = sum(u["accepted"] for u in by_model.values())
    return {
        "predictions": predictions,
        "completions": sum(u["completions"] for u in by_model.values()),
        "accepted": accepted,
        "acceptance_rate": round(accepted / predictions, 4) if predictions else None,
        "by_model": by_model,
    }


class DocumentStore(Protocol):
    backend: str

    def ping(self) -> bool: ...
    def list_documents(self, owner_id: str, limit: int, skip: int) -> tuple[list[dict], int]: ...
    def count_documents(self, owner_id: str) -> int: ...
    def create_document(self, owner_id: str, title: str, content: str) -> dict: ...
    def get_document(self, owner_id: str, doc_id: str) -> dict | None: ...
    def update_document(self, owner_id: str, doc_id: str, title: str | None, content: str | None) -> dict | None: ...
    def delete_document(self, owner_id: str, doc_id: str) -> bool: ...
    def record_event(self, event_type: str, model: str, owner_id: str | None, latency_ms: float | None, mode: str | None) -> None: ...
    def analytics(self, owner_id: str | None) -> dict: ...
