import json
import time

import pytest


def test_predict_returns_ranked_suggestions(client):
    res = client.post("/predict", json={"text": "The cat sat on the ", "model": "ngram", "top_k": 5})
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "ngram" and body["mode"] == "next_word" and body["model_version"]
    words = [p["word"] for p in body["predictions"]]
    scores = [p["score"] for p in body["predictions"]]
    assert 1 <= len(words) <= 5 and scores == sorted(scores, reverse=True)
    assert body["context"][-2:] == ["on", "the"] and body["latency_ms"] >= 0


def test_predict_defaults_work_with_empty_body(client):
    res = client.post("/predict", json={})
    assert res.status_code == 200 and res.json()["predictions"]


def test_predict_inside_a_word_returns_completions(client):
    body = client.post("/predict", json={"text": "the cat sa"}).json()
    assert body["mode"] == "completion" and body["prefix"] == "sa"
    assert all(p["word"].startswith("sa") for p in body["predictions"])


@pytest.mark.parametrize("text", ["", "   ", "???", "zzzz qqqq ", "日本語 "])
def test_predict_handles_empty_punctuation_and_unknown_input(client, text):
    res = client.post("/predict", json={"text": text})
    assert res.status_code == 200 and len(res.json()["predictions"]) > 0


@pytest.mark.parametrize(
    "payload",
    [
        {"top_k": 0}, {"top_k": 11}, {"top_k": "five"},
        {"model": "gpt"}, {"text": "x" * 5001}, {"text": 123},
    ],
)
def test_predict_validation_errors_use_the_error_format(client, payload):
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    err = res.json()["error"]
    assert err["code"] == "validation_error" and err["details"]


def test_validation_errors_do_not_echo_the_input_text(client):
    secret = "my-private-diary-entry " * 400  # > 5000 chars
    res = client.post("/predict", json={"text": secret})
    assert res.status_code == 422 and "private-diary" not in res.text


def test_malformed_json_is_a_clean_422(client):
    res = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422 and res.json()["error"]["code"] == "validation_error"


def test_unavailable_model_returns_503_with_reason(client):
    res = client.post("/predict", json={"text": "hello", "model": "lstm"})
    assert res.status_code == 503
    err = res.json()["error"]
    assert err["code"] == "model_unavailable" and "not trained" in err["message"]
    assert "/home" not in res.text and "Traceback" not in res.text   # no paths / stack traces


def test_complete_returns_bounded_continuation(client):
    res = client.post("/complete", json={"text": "The cat sat", "max_tokens": 6, "seed": 5})
    assert res.status_code == 200
    body = res.json()
    assert 0 < len(body["tokens"]) <= 6 and body["continuation"].startswith(" ")
    assert client.post("/complete", json={"text": "The cat sat", "max_tokens": 6, "seed": 5}).json()["continuation"] == body["continuation"]


@pytest.mark.parametrize("payload", [{"max_tokens": 0}, {"max_tokens": 61}, {"temperature": 0}, {"temperature": 5}, {"model": "x"}, {"seed": -1}])
def test_complete_validation(client, payload):
    assert client.post("/complete", json={"text": "hi", **payload}).status_code == 422


def test_complete_with_unavailable_model_is_503(client):
    assert client.post("/complete", json={"text": "hi", "model": "lstm"}).status_code == 503
    assert client.post("/complete", json={"text": "hi", "model": "transformer"}).status_code == 503


def test_untrained_transformer_returns_503_naming_the_training_script(client):
    res = client.post("/predict", json={"text": "hello", "model": "transformer"})
    assert res.status_code == 503 and "train_transformer.py" in res.json()["error"]["message"]


def test_rate_limit_returns_429_with_retry_after(make_client):
    client = make_client(rate_limit_per_minute=3)
    codes = [client.post("/predict", json={"text": "the "}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    res = client.post("/predict", json={"text": "the "})
    assert int(res.headers["retry-after"]) >= 1 and res.json()["error"]["code"] == "rate_limited"
    assert client.get("/models").status_code == 200   # other routes are not limited


def test_slow_inference_times_out_without_a_stack_trace(make_client, engine, monkeypatch):
    def slow(*a, **k):
        time.sleep(0.6)

    monkeypatch.setattr(engine, "predict", slow)
    client = make_client(predict_timeout_seconds=0.1)
    res = client.post("/predict", json={"text": "hello "})
    assert res.status_code == 504 and res.json()["error"]["code"] == "prediction_timeout"


def test_unexpected_errors_become_a_generic_500(make_client, engine, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail /srv/models/x")

    monkeypatch.setattr(engine, "predict", boom)
    client = make_client(raise_server_exceptions=False)
    res = client.post("/predict", json={"text": "hello "})
    assert res.status_code == 500
    assert res.json() == {"error": {"code": "internal_error", "message": "An unexpected error occurred."}}


# ------------------------------------------------------------------ analytics / privacy
def test_events_store_metadata_only_never_the_typed_text(client, store, alice):
    secret = "my-secret-sentence-about-nothing"
    client.post("/predict", json={"text": secret + " "}, headers=alice)
    client.post("/complete", json={"text": secret + " ", "max_tokens": 3}, headers=alice)
    events = store.events
    assert {e["type"] for e in events} == {"predict", "complete"}
    dump = json.dumps(events, default=str)
    assert "secret" not in dump and secret not in dump
    assert set(events[0]) == {"type", "model", "owner_id", "latency_ms", "mode", "timestamp"}


def test_analytics_counts_predictions_and_accepted_suggestions(client, alice, bob):
    for _ in range(3):
        client.post("/predict", json={"text": "the "}, headers=alice)
    client.post("/predict", json={"text": "the "}, headers=bob)
    assert client.post("/analytics/accepted", json={"model": "ngram"}, headers=alice).status_code == 204

    mine = client.get("/analytics", headers=alice).json()
    assert mine["scope"] == "client" and mine["predictions"] == 3 and mine["accepted"] == 1
    assert mine["acceptance_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert mine["by_model"]["ngram"]["avg_latency_ms"] is not None
    everyone = client.get("/analytics").json()
    assert everyone["scope"] == "global" and everyone["predictions"] == 4


def test_analytics_empty_state(client):
    body = client.get("/analytics").json()
    assert body["predictions"] == 0 and body["acceptance_rate"] is None and body["by_model"] == {}


def test_analytics_accepted_rejects_unknown_model(client):
    assert client.post("/analytics/accepted", json={"model": "gpt"}).status_code == 422


def test_invalid_client_id_is_rejected(client):
    assert client.post("/predict", json={"text": "a"}, headers={"X-Client-Id": "short"}).status_code == 400
