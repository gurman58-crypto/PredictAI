from __future__ import annotations

import logging

from app.config import Settings
from app.db.memory import MemoryDocumentStore
from app.db.mongo import MongoDocumentStore

log = logging.getLogger("predictai.db")


def create_store(settings: Settings):
    """MongoDB when MONGODB_URI is set, otherwise the (non-persistent) in-memory store."""
    if settings.mongodb_uri:
        store = MongoDocumentStore.from_uri(settings.mongodb_uri, settings.mongodb_db)
        if store.ping():
            log.info("connected to MongoDB database %r", settings.mongodb_db)
        else:
            log.error("MongoDB is configured but not reachable - document routes will return 503")
        return store
    log.warning("MONGODB_URI is not set: using the in-memory store (documents are lost on restart)")
    return MemoryDocumentStore()
