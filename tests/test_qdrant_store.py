"""M1.2 Qdrant store: HTTP mocked, so this asserts the real request shapes offline."""
import uuid

import adapters.qdrant_store as qs_mod
from adapters.qdrant_store import QdrantStore, point_id


def test_point_id_is_uuid_stable_and_unique():
    pid = point_id("R1#0")
    uuid.UUID(pid)  # raises if not a valid uuid
    assert pid == point_id("R1#0")
    assert pid != point_id("R1#1")


def test_ensure_collection_creates_when_missing(monkeypatch):
    calls = []

    def fake(method, url, payload=None, headers=None, timeout=60):
        calls.append((method, url, payload))
        return {"result": {"exists": False}} if url.endswith("/exists") else {}

    monkeypatch.setattr(qs_mod, "request_json", fake)
    QdrantStore(collection="c", url="http://q").ensure_collection(1024)
    puts = [c for c in calls if c[0] == "PUT"]
    assert puts, "should create the collection when missing"
    body = puts[0][2]
    assert body["vectors"]["dense"]["size"] == 1024
    assert body["sparse_vectors"]["sparse"]["modifier"] == "idf"


def test_ensure_collection_is_idempotent_when_present(monkeypatch):
    methods = []

    def fake(method, url, payload=None, headers=None, timeout=60):
        methods.append(method)
        return {"result": {"exists": True}} if url.endswith("/exists") else {}

    monkeypatch.setattr(qs_mod, "request_json", fake)
    QdrantStore(collection="c", url="http://q").ensure_collection(1024)
    assert "PUT" not in methods  # no create when it already exists


def test_upsert_builds_named_vectors_with_uuid_id(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload)
        return {}

    monkeypatch.setattr(qs_mod, "request_json", fake)
    QdrantStore(collection="c", url="http://q").upsert(
        [{"id": "R1", "text": "hi", "payload": {"lang": "en"},
          "dense": [0.1], "sparse": {"indices": [3], "values": [0.5]}}])
    assert "/points?wait=true" in seen["url"]
    point = seen["payload"]["points"][0]
    uuid.UUID(point["id"])  # stored under a uuid, not the raw chunk id
    assert point["vector"]["dense"] == [0.1]
    assert point["vector"]["sparse"] == {"indices": [3], "values": [0.5]}
    assert point["payload"]["chunk_id"] == "R1"
    assert point["payload"]["text"] == "hi"


def test_api_key_is_sent_as_a_header_when_set(monkeypatch):
    # Qdrant Cloud rejects unauthenticated requests, so the key must ride on every call.
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen["headers"] = headers
        return {"result": {"exists": True}}

    monkeypatch.setattr(qs_mod, "request_json", fake)
    QdrantStore(collection="c", url="http://q", api_key="secret-key")._exists()
    assert seen["headers"] == {"api-key": "secret-key"}


def test_no_api_key_sends_no_header(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen["headers"] = headers
        return {"result": {"exists": True}}

    monkeypatch.setattr(qs_mod, "request_json", fake)
    QdrantStore(collection="c", url="http://q", api_key="")._exists()
    assert seen["headers"] is None  # local Qdrant: no header


def test_dense_only_is_a_plain_query_no_fusion(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload)
        return {"result": {"points": [{"id": "x", "payload": {"text": "t"}}]}}

    monkeypatch.setattr(qs_mod, "request_json", fake)
    QdrantStore(collection="c", url="http://q").hybrid_search(
        [0.1], {"indices": [3], "values": [0.5]}, top_k=5, where={"lang": "fr"}, dense_only=True)
    body = seen["payload"]
    assert body["query"] == [0.1]          # a bare vector = plain kNN
    assert body["using"] == "dense"
    assert "prefetch" not in body          # no fusion for dense-only
    assert body["limit"] == 5
    assert body["filter"] == {"must": [{"key": "lang", "match": {"value": "fr"}}]}


def test_hybrid_search_uses_rrf_and_must_match_filter(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload)
        return {"result": {"points": [{"id": "x", "payload": {"text": "t"}}]}}

    monkeypatch.setattr(qs_mod, "request_json", fake)
    points = QdrantStore(collection="c", url="http://q").hybrid_search(
        [0.1], {"indices": [3], "values": [0.5]}, top_k=5, where={"lang": "fr"})
    assert "/points/query" in seen["url"]
    body = seen["payload"]
    assert body["query"] == {"fusion": "rrf"}
    assert len(body["prefetch"]) == 2
    expected = {"must": [{"key": "lang", "match": {"value": "fr"}}]}
    for branch in body["prefetch"]:
        assert branch["filter"] == expected
    assert points[0]["id"] == "x"
