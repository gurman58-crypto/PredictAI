"""The Transformer served through the real API (tiny model trained on a toy corpus)."""
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("tensorflow")

from app.config import Settings  # noqa: E402
from app.db.memory import MemoryDocumentStore  # noqa: E402
from app.main import create_app  # noqa: E402
from ml_models.inference import PredictionEngine  # noqa: E402
from ml_models.testing import build_toy_saved_dir  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    saved = build_toy_saved_dir(tmp_path_factory.mktemp("api_transformer"), with_transformer=True)
    app = create_app(Settings(rate_limit_per_minute=0), engine=PredictionEngine(saved), store=MemoryDocumentStore())
    with TestClient(app) as c:
        yield c


def test_models_lists_the_transformer_as_ready(client):
    body = client.get("/models").json()
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["transformer"]["status"] == "ready" and by_id["transformer"]["version"] == "toy"
    assert by_id["lstm"]["status"] == "unavailable"
    assert client.get("/").json()["models"]["transformer"] == "ready"


def test_predict_with_the_transformer(client):
    res = client.post("/predict", json={"text": "The cat sat on the ", "model": "transformer", "top_k": 5})
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "transformer" and body["model_version"] == "toy" and body["mode"] == "next_word"
    scores = [p["score"] for p in body["predictions"]]
    assert 1 <= len(scores) <= 5 and scores == sorted(scores, reverse=True)


def test_predict_handles_long_and_empty_text_with_the_transformer(client):
    long_text = "the cat sat on the mat. " * 200      # far more than the 30-token position table
    for text in (long_text, "", "zzzz qqqq "):
        res = client.post("/predict", json={"text": text, "model": "transformer"})
        assert res.status_code == 200 and res.json()["predictions"], text[:20]


def test_complete_with_the_transformer_is_bounded_and_reproducible(client):
    payload = {"text": "The cat", "model": "transformer", "max_tokens": 6, "seed": 11}
    a, b = client.post("/complete", json=payload).json(), client.post("/complete", json=payload).json()
    assert a["continuation"] == b["continuation"] and 0 < len(a["tokens"]) <= 6 and a["model"] == "transformer"


def test_analytics_tracks_the_transformer_separately(client):
    client.post("/predict", json={"text": "the ", "model": "transformer"})
    by_model = client.get("/analytics").json()["by_model"]
    assert by_model["transformer"]["predictions"] >= 1


def test_blend_is_served_and_is_the_default_when_tuned(tmp_path_factory):
    saved = build_toy_saved_dir(tmp_path_factory.mktemp("api_blend"), with_transformer=True, with_ensemble=True)
    app = create_app(Settings(rate_limit_per_minute=0), engine=PredictionEngine(saved), store=MemoryDocumentStore())
    with TestClient(app) as c:
        body = c.get("/models").json()
        assert body["default_model"] == "ensemble"
        r = c.post("/predict", json={"text": "the history of the", "model": "ensemble"})
        assert r.status_code == 200 and r.json()["model"] == "ensemble" and r.json()["predictions"]
        r = c.post("/complete", json={"text": "the history of", "model": "ensemble"})
        assert r.status_code == 200
