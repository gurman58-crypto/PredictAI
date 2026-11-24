"""In-memory store used ONLY when MONGODB_URI is not configured (local demos and unit tests).

Data is lost when the server stops.  /  reports ``database.backend == "memory"`` so this can
never be mistaken for real persistence.
"""
from __future__ import annotations

import threading
import uuid
from collections import defaultdict, deque

from app.db.base import (
    DOC_ID_RE,
    count_words,
    summarise_events,
    to_document,
    to_summary,
    utcnow,
)

MAX_EVENTS = 20_000


class MemoryDocumentStore:
    backend = "memory"

    def __init__(self):
        self._docs: dict[str, dict] = {}
        self._events: deque = deque(maxlen=MAX_EVENTS)
        self._lock = threading.RLock()

    @property
    def events(self) -> list[dict]:
        """Read-only copy, used by tests to prove that no raw text is logged."""
        with self._lock:
            return list(self._events)

    def ping(self) -> bool:
        return True

    # ---------------------------------------------------------------- documents
    def list_documents(self, owner_id, limit, skip):
        with self._lock:
            mine = [d for d in self._docs.values() if d["owner_id"] == owner_id]
            mine.sort(key=lambda d: d["updated_at"], reverse=True)
            return [to_summary(d) for d in mine[skip : skip + limit]], len(mine)

    def count_documents(self, owner_id):
        with self._lock:
            return sum(1 for d in self._docs.values() if d["owner_id"] == owner_id)

    def create_document(self, owner_id, title, content):
        now = utcnow()
        raw = {
            "id": uuid.uuid4().hex[:24],
            "owner_id": owner_id,
            "title": title,
            "content": content,
            "word_count": count_words(content),
            "char_count": len(content),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._docs[raw["id"]] = raw
        return to_document(raw)

    def _owned(self, owner_id, doc_id):
        if not DOC_ID_RE.match(doc_id):
            return None
        doc = self._docs.get(doc_id)
        return doc if doc is not None and doc["owner_id"] == owner_id else None

    def get_document(self, owner_id, doc_id):
        with self._lock:
            doc = self._owned(owner_id, doc_id)
            return to_document(doc) if doc else None

    def update_document(self, owner_id, doc_id, title, content):
        with self._lock:
            doc = self._owned(owner_id, doc_id)
            if doc is None:
                return None
            if title is not None:
                doc["title"] = title
            if content is not None:
                doc["content"] = content
                doc["word_count"] = count_words(content)
                doc["char_count"] = len(content)
            doc["updated_at"] = utcnow()
            return to_document(doc)

    def delete_document(self, owner_id, doc_id):
        with self._lock:
            if self._owned(owner_id, doc_id) is None:
                return False
            del self._docs[doc_id]
            return True

    # ------------------------------------------------------------------ analytics
    def record_event(self, event_type, model, owner_id, latency_ms, mode):
        with self._lock:
            self._events.append(
                {
                    "type": event_type,
                    "model": model,
                    "owner_id": owner_id,
                    "latency_ms": latency_ms,
                    "mode": mode,
                    "timestamp": utcnow(),
                }
            )

    def analytics(self, owner_id):
        groups: dict[tuple, list] = defaultdict(list)
        with self._lock:
            for e in self._events:
                if owner_id is None or e["owner_id"] == owner_id:
                    groups[(e["type"], e["model"])].append(e["latency_ms"])
        rows = []
        for (etype, model), lat in groups.items():
            vals = [v for v in lat if v is not None]
            rows.append((etype, model, len(lat), sum(vals) / len(vals) if vals else None))
        return summarise_events(rows)
