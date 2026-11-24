"""MongoDB (Atlas) implementation using PyMongo.

Collections
  documents          {_id, owner_id, title, content, word_count, char_count, created_at, updated_at}
  prediction_events  {_id, type, model, owner_id, latency_ms, mode, timestamp}   (no text, ever)

Every document query includes ``owner_id`` - that is the ownership check.
"""
from __future__ import annotations

import logging

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING, ReturnDocument
from pymongo.errors import PyMongoError

from app.db.base import count_words, summarise_events, to_document, to_summary, utcnow

log = logging.getLogger("predictai.db")


def _oid(doc_id: str) -> ObjectId | None:
    try:
        return ObjectId(doc_id)
    except (InvalidId, TypeError):
        return None


def _public(raw: dict) -> dict:
    raw = dict(raw)
    raw["id"] = str(raw.pop("_id"))
    return raw


class MongoDocumentStore:
    backend = "mongodb"

    def __init__(self, client, db_name: str = "predictai"):
        self.client = client
        self.db = client[db_name]
        self.documents = self.db["documents"]
        self.events = self.db["prediction_events"]
        self._indexes_ready = False

    @classmethod
    def from_uri(cls, uri: str, db_name: str = "predictai") -> "MongoDocumentStore":
        from pymongo import MongoClient

        # lazy connection: the API still starts if Atlas is unreachable and reports 503 on DB routes
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True, appname="predictai")
        return cls(client, db_name)

    def ensure_indexes(self) -> bool:
        if self._indexes_ready:
            return True
        try:
            self.documents.create_index([("owner_id", 1), ("updated_at", DESCENDING)])
            self.events.create_index([("owner_id", 1), ("timestamp", DESCENDING)])
            self._indexes_ready = True
        except PyMongoError as exc:
            log.warning("could not create indexes yet: %s", type(exc).__name__)
        return self._indexes_ready

    def ping(self) -> bool:
        try:
            self.client.admin.command("ping")
            self.ensure_indexes()
            return True
        except PyMongoError:
            return False

    # ---------------------------------------------------------------- documents
    def list_documents(self, owner_id, limit, skip):
        total = self.documents.count_documents({"owner_id": owner_id})
        cursor = (
            self.documents.find({"owner_id": owner_id})
            .sort("updated_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        return [to_summary(_public(d)) for d in cursor], total

    def count_documents(self, owner_id):
        return self.documents.count_documents({"owner_id": owner_id})

    def create_document(self, owner_id, title, content):
        now = utcnow()
        raw = {
            "owner_id": owner_id,
            "title": title,
            "content": content,
            "word_count": count_words(content),
            "char_count": len(content),
            "created_at": now,
            "updated_at": now,
        }
        raw["_id"] = self.documents.insert_one(raw).inserted_id
        return to_document(_public(raw))

    def get_document(self, owner_id, doc_id):
        oid = _oid(doc_id)
        if oid is None:
            return None
        raw = self.documents.find_one({"_id": oid, "owner_id": owner_id})
        return to_document(_public(raw)) if raw else None

    def update_document(self, owner_id, doc_id, title, content):
        oid = _oid(doc_id)
        if oid is None:
            return None
        changes: dict = {"updated_at": utcnow()}
        if title is not None:
            changes["title"] = title
        if content is not None:
            changes.update(content=content, word_count=count_words(content), char_count=len(content))
        raw = self.documents.find_one_and_update(
            {"_id": oid, "owner_id": owner_id}, {"$set": changes}, return_document=ReturnDocument.AFTER
        )
        return to_document(_public(raw)) if raw else None

    def delete_document(self, owner_id, doc_id):
        oid = _oid(doc_id)
        if oid is None:
            return False
        return self.documents.delete_one({"_id": oid, "owner_id": owner_id}).deleted_count == 1

    # ------------------------------------------------------------------ analytics
    def record_event(self, event_type, model, owner_id, latency_ms, mode):
        self.events.insert_one(
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
        match = {"owner_id": owner_id} if owner_id else {}
        pipeline = [
            {"$match": match},
            {
                "$group": {
                    "_id": {"type": "$type", "model": "$model"},
                    "count": {"$sum": 1},
                    "avg_latency": {"$avg": "$latency_ms"},
                }
            },
        ]
        rows = [(r["_id"]["type"], r["_id"]["model"], r["count"], r["avg_latency"]) for r in self.events.aggregate(pipeline)]
        return summarise_events(rows)
