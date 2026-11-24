"""The same behaviour contract is verified for the in-memory store and the MongoDB store.

The MongoDB store runs against ``mongomock`` (an in-process MongoDB-compatible fake) so the tests need
no database server.  The code path is the real PyMongo-based ``MongoDocumentStore``.
"""
import mongomock
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.memory import MemoryDocumentStore
from app.db.mongo import MongoDocumentStore
from app.main import create_app

A, B = "owner-a-0000000001", "owner-b-0000000002"


@pytest.fixture(params=["memory", "mongodb"])
def store(request):
    if request.param == "memory":
        return MemoryDocumentStore()
    return MongoDocumentStore(mongomock.MongoClient(tz_aware=True), "predictai_test")


def test_crud_contract(store):
    assert store.ping()
    doc = store.create_document(A, "T", "one two three")
    assert store.get_document(A, doc["id"])["word_count"] == 3
    upd = store.update_document(A, doc["id"], None, "one two")
    assert upd["word_count"] == 2 and upd["title"] == "T" and upd["updated_at"] >= doc["updated_at"]
    assert store.update_document(A, doc["id"], "New", None)["content"] == "one two"
    assert store.count_documents(A) == 1
    assert store.delete_document(A, doc["id"]) is True
    assert store.get_document(A, doc["id"]) is None
    assert store.delete_document(A, doc["id"]) is False


def test_ownership_is_enforced_in_every_operation(store):
    doc = store.create_document(A, "mine", "text")
    assert store.get_document(B, doc["id"]) is None
    assert store.update_document(B, doc["id"], "x", "y") is None
    assert store.delete_document(B, doc["id"]) is False
    assert store.list_documents(B, 10, 0) == ([], 0)
    assert store.count_documents(B) == 0
    assert store.get_document(A, doc["id"])["title"] == "mine"


def test_invalid_ids_do_not_raise(store):
    for bad in ("", "xyz", "0" * 24, "z" * 24):
        assert store.get_document(A, bad) is None
        assert store.update_document(A, bad, "t", None) is None
        assert store.delete_document(A, bad) is False


def test_list_sorted_by_updated_desc_and_paginated(store):
    ids = [store.create_document(A, f"d{i}", "w") ["id"] for i in range(4)]
    store.update_document(A, ids[0], None, "changed")
    docs, total = store.list_documents(A, 10, 0)
    assert total == 4 and docs[0]["id"] == ids[0] and "content" not in docs[0] and "preview" in docs[0]
    assert len(store.list_documents(A, 2, 2)[0]) == 2


def test_analytics_aggregation_and_privacy(store):
    store.record_event("predict", "ngram", A, 2.0, "next_word")
    store.record_event("predict", "ngram", A, 4.0, "completion")
    store.record_event("predict", "lstm", B, 10.0, "next_word")
    store.record_event("accept", "ngram", A, None, None)
    store.record_event("complete", "lstm", B, 20.0, None)
    mine = store.analytics(A)
    assert mine["predictions"] == 2 and mine["accepted"] == 1 and mine["acceptance_rate"] == 0.5
    assert mine["by_model"]["ngram"]["avg_latency_ms"] == 3.0
    everyone = store.analytics(None)
    assert everyone["predictions"] == 3 and everyone["completions"] == 1
    assert set(everyone["by_model"]) == {"ngram", "lstm"}


def test_api_works_end_to_end_on_top_of_either_store(engine, store, alice, bob):
    app = create_app(Settings(rate_limit_per_minute=0), engine=engine, store=store)
    with TestClient(app) as client:
        assert client.get("/").json()["database"] == {"backend": store.backend, "connected": True}
        doc = client.post("/documents", json={"title": "Hi", "content": "a b"}, headers=alice).json()
        assert client.get(f"/documents/{doc['id']}", headers=bob).status_code == 404
        assert client.put(f"/documents/{doc['id']}", json={"content": "a b c"}, headers=alice).json()["word_count"] == 3
        assert client.get("/documents", headers=alice).json()["total"] == 1
        client.post("/predict", json={"text": "the "}, headers=alice)
        assert client.get("/analytics", headers=alice).json()["predictions"] == 1
        assert client.delete(f"/documents/{doc['id']}", headers=alice).status_code == 200


def test_unreachable_database_returns_503_not_a_crash(engine, alice):
    class Down(MemoryDocumentStore):
        backend = "mongodb"

        def ping(self):
            return False

        def list_documents(self, *a, **k):
            from pymongo.errors import ServerSelectionTimeoutError

            raise ServerSelectionTimeoutError("cannot reach cluster at mongodb+srv://user:PASSWORD@host")

    app = create_app(Settings(rate_limit_per_minute=0), engine=engine, store=Down())
    with TestClient(app) as client:
        assert client.get("/").json()["database"]["connected"] is False
        res = client.get("/documents", headers=alice)
        assert res.status_code == 503 and res.json()["error"]["code"] == "database_unavailable"
        assert "PASSWORD" not in res.text and "mongodb+srv" not in res.text   # connection string never leaks
        assert client.post("/predict", json={"text": "the "}).status_code == 200   # predictions keep working
