import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.memory import MemoryDocumentStore
from app.main import create_app
from ml_models.inference import PredictionEngine
from ml_models.testing import build_toy_saved_dir

ALICE = {"X-Client-Id": "alice-client-id-0001"}
BOB = {"X-Client-Id": "bob-client-id-000002"}


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    return build_toy_saved_dir(tmp_path_factory.mktemp("saved_models"))


@pytest.fixture(scope="session")
def engine(toy_dir):
    return PredictionEngine(toy_dir)


@pytest.fixture()
def store():
    return MemoryDocumentStore()


@pytest.fixture()
def make_client(engine, store):
    """Factory so individual tests can tweak the settings (rate limit, timeout, ...)."""
    clients = []

    def _make(raise_server_exceptions=True, **settings):
        settings.setdefault("rate_limit_per_minute", 0)
        app = create_app(Settings(**settings), engine=engine, store=store)
        client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def client(make_client):
    return make_client()


@pytest.fixture()
def alice():
    return ALICE


@pytest.fixture()
def bob():
    return BOB


@pytest.fixture()
def saved_report(toy_dir):
    """Writes a fake evaluation_results.json into the toy dir for the /evaluation test, then removes it."""
    path = toy_dir / "evaluation_results.json"
    path.write_text(json.dumps({"experiment_version": "test", "models": {}}))
    yield path
    path.unlink(missing_ok=True)
