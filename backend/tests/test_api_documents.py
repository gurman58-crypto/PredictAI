import time

import pytest


def create(client, headers, title="Notes", content="hello world again"):
    res = client.post("/documents", json={"title": title, "content": content}, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()


def test_full_crud_flow(client, alice):
    doc = create(client, alice, "My essay", "The quick brown fox")
    assert doc["word_count"] == 4 and doc["char_count"] == 19 and len(doc["id"]) == 24
    assert doc["created_at"] == doc["updated_at"]

    got = client.get(f"/documents/{doc['id']}", headers=alice).json()
    assert got["content"] == "The quick brown fox" and got["title"] == "My essay"

    time.sleep(0.01)
    upd = client.put(f"/documents/{doc['id']}", json={"content": "A much longer piece of writing here"}, headers=alice)
    assert upd.status_code == 200
    body = upd.json()
    assert body["title"] == "My essay" and body["word_count"] == 7      # title untouched
    assert body["updated_at"] > body["created_at"]

    renamed = client.put(f"/documents/{doc['id']}", json={"title": "  Renamed  "}, headers=alice).json()
    assert renamed["title"] == "Renamed" and renamed["content"].startswith("A much")   # title stripped, content kept

    deleted = client.delete(f"/documents/{doc['id']}", headers=alice)
    assert deleted.status_code == 200 and deleted.json() == {"deleted": True, "id": doc["id"]}
    assert client.get(f"/documents/{doc['id']}", headers=alice).status_code == 404
    assert client.delete(f"/documents/{doc['id']}", headers=alice).status_code == 404


def test_create_defaults(client, alice):
    doc = client.post("/documents", json={}, headers=alice).json()
    assert doc["title"] == "Untitled document" and doc["content"] == "" and doc["word_count"] == 0


def test_list_is_newest_first_with_preview_and_counts(client, alice):
    first = create(client, alice, "First", "a b c")
    time.sleep(0.01)
    second = create(client, alice, "Second", "x   y\n\nz " * 100)
    time.sleep(0.01)
    client.put(f"/documents/{first['id']}", json={"content": "a b c d"}, headers=alice)   # now most recently updated

    body = client.get("/documents", headers=alice).json()
    assert body["total"] == 2 and body["count"] == 2
    assert [d["title"] for d in body["documents"]] == ["First", "Second"]
    assert "content" not in body["documents"][0]
    assert len(body["documents"][1]["preview"]) <= 160 and "\n" not in body["documents"][1]["preview"]
    assert body["documents"][0]["word_count"] == 4
    assert second["id"] == body["documents"][1]["id"]


def test_list_pagination(client, alice):
    for i in range(5):
        create(client, alice, f"Doc {i}")
    page = client.get("/documents?limit=2&skip=1", headers=alice).json()
    assert page["count"] == 2 and page["total"] == 5
    assert client.get("/documents?limit=0", headers=alice).status_code == 422
    assert client.get("/documents?limit=101", headers=alice).status_code == 422
    assert client.get("/documents?skip=-1", headers=alice).status_code == 422


def test_empty_list(client, alice):
    assert client.get("/documents", headers=alice).json() == {"documents": [], "count": 0, "total": 0}


# ----------------------------------------------------------------------- ownership
def test_documents_are_private_to_their_owner(client, alice, bob):
    doc = create(client, alice, "Alice only", "secret text")
    url = f"/documents/{doc['id']}"

    assert client.get(url, headers=bob).status_code == 404
    assert client.put(url, json={"title": "hacked"}, headers=bob).status_code == 404
    assert client.delete(url, headers=bob).status_code == 404
    assert client.get("/documents", headers=bob).json()["total"] == 0

    still = client.get(url, headers=alice).json()                  # untouched by Bob's attempts
    assert still["title"] == "Alice only" and still["content"] == "secret text"


def test_other_clients_get_the_same_404_as_missing_documents(client, alice, bob):
    doc = create(client, alice)
    foreign = client.get(f"/documents/{doc['id']}", headers=bob)
    missing = client.get(f"/documents/{'0' * 24}", headers=bob)
    assert foreign.status_code == missing.status_code == 404 and foreign.json() == missing.json()


@pytest.mark.parametrize("method", ["get", "post", "put", "delete"])
def test_client_id_is_required_for_document_routes(client, method):
    url = "/documents" if method in ("get", "post") else "/documents/" + "0" * 24
    kwargs = {"json": {"title": "x", "content": "y"}} if method in ("post", "put") else {}
    res = getattr(client, method)(url, **kwargs)
    assert res.status_code == 401 and res.json()["error"]["code"] == "missing_client_id"


def test_malformed_client_id_is_rejected(client):
    res = client.get("/documents", headers={"X-Client-Id": "bad id!"})
    assert res.status_code == 400 and res.json()["error"]["code"] == "invalid_client_id"


# ---------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "payload",
    [{"title": ""}, {"title": "   "}, {"title": "x" * 201}, {"content": "y" * 200_001}, {"title": 5}, {"content": ["a"]}],
)
def test_create_validation(client, alice, payload):
    res = client.post("/documents", json=payload, headers=alice)
    assert res.status_code == 422 and res.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("payload", [{}, {"title": ""}, {"content": "z" * 200_001}])
def test_update_validation(client, alice, payload):
    doc = create(client, alice)
    assert client.put(f"/documents/{doc['id']}", json=payload, headers=alice).status_code == 422


@pytest.mark.parametrize("doc_id", ["abc", "not-an-object-id", "0" * 23, "G" * 24, "../../etc/passwd"])
def test_malformed_document_ids_are_404(client, alice, doc_id):
    assert client.get(f"/documents/{doc_id}", headers=alice).status_code == 404


def test_document_limit_per_client(make_client, alice, bob):
    client = make_client(max_documents_per_client=2)
    create(client, alice)
    create(client, alice)
    res = client.post("/documents", json={"title": "third"}, headers=alice)
    assert res.status_code == 409 and res.json()["error"]["code"] == "document_limit_reached"
    assert client.post("/documents", json={"title": "bobs"}, headers=bob).status_code == 201   # limit is per client


def test_unicode_and_large_content_roundtrip(client, alice):
    text = "Zażółć gęślą jaźń — 日本語のテキスト 🙂\n" * 2000
    doc = create(client, alice, "Ünïcode ✓", text)
    got = client.get(f"/documents/{doc['id']}", headers=alice).json()
    assert got["content"] == text and got["title"] == "Ünïcode ✓" and got["char_count"] == len(text)
