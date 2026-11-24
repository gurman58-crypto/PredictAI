def test_root_reports_service_database_and_models(client):
    body = client.get("/").json()
    assert body["name"] == "PredictAI API" and body["status"] == "ok"
    assert body["database"] == {"backend": "memory", "connected": True}
    assert body["models"] == {"ensemble": "unavailable", "ngram": "ready", "lstm": "unavailable", "transformer": "unavailable"}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models_lists_status_and_reason_for_missing_model(client):
    body = client.get("/models").json()
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["ngram"]["status"] == "ready" and by_id["ngram"]["version"]
    assert by_id["lstm"]["status"] == "unavailable" and "not trained" in by_id["lstm"]["reason"]
    assert by_id["transformer"]["status"] == "unavailable" and "train_transformer.py" in by_id["transformer"]["reason"]
    assert [m["id"] for m in body["models"]] == ["ensemble", "ngram", "lstm", "transformer"]
    assert by_id["ensemble"]["status"] == "unavailable" and "train_ensemble.py" in by_id["ensemble"]["reason"]
    assert body["default_model"] == "ngram"


def test_evaluation_is_pending_until_a_real_report_exists(client):
    body = client.get("/evaluation").json()
    assert body["status"] == "pending" and "evaluate" in body["message"]


def test_evaluation_returns_the_report_when_present(client, saved_report):
    body = client.get("/evaluation").json()
    assert body["status"] == "available" and body["experiment_version"] == "test"


def test_unknown_route_uses_the_error_format(client):
    res = client.get("/nope")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_wrong_method_uses_the_error_format(client):
    res = client.delete("/predict")
    assert res.status_code == 405 and res.json()["error"]["code"] == "method_not_allowed"


def test_openapi_documents_every_required_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    expected = {
        ("/", "get"), ("/models", "get"), ("/predict", "post"), ("/complete", "post"),
        ("/documents", "get"), ("/documents", "post"), ("/documents/{doc_id}", "get"),
        ("/documents/{doc_id}", "put"), ("/documents/{doc_id}", "delete"), ("/analytics", "get"),
        ("/evaluation", "get"),
    }
    for path, method in expected:
        assert method in paths[path], f"{method.upper()} {path} missing from OpenAPI"


def test_cors_allows_the_configured_origin_only(make_client):
    client = make_client(cors_origins=("https://app.example.com",))
    ok = client.options(
        "/predict",
        headers={"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type,x-client-id"},
    )
    assert ok.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "x-client-id" in ok.headers["access-control-allow-headers"].lower()
    bad = client.options(
        "/predict",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers
