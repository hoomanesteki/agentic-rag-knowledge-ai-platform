"""M1.2 Qdrant store: the HTTP is mocked, so this checks the request shape offline."""
import adapters.qdrant_store as qs_mod
from adapters.qdrant_store import QdrantStore, point_id


def test_point_id_is_stable_and_unique():
    assert point_id("R1#0") == point_id("R1#0")
    assert point_id("R1#0") != point_id("R1#1")


def test_upsert_builds_named_vectors(monkeypatch):
    calls = {}

    def fake_request(method, url, payload=None, headers=None, timeout=60):
        calls["payload"] = payload
        return {}

    monkeypatch.setattr(qs_mod, "request_json", fake_request)
    store = QdrantStore(collection="c", url="http://localhost:6333")
    store.upsert([{"id": "R1", "text": "hi", "payload": {"lang": "en"},
                   "dense": [0.1], "sparse": {"indices": [3], "values": [0.5]}}])
    point = calls["payload"]["points"][0]
    assert point["vector"]["dense"] == [0.1]
    assert point["vector"]["sparse"] == {"indices": [3], "values": [0.5]}
    assert point["payload"]["chunk_id"] == "R1"
    assert point["payload"]["text"] == "hi"


def test_hybrid_search_uses_rrf_fusion(monkeypatch):
    def fake_request(method, url, payload=None, headers=None, timeout=60):
        assert payload["query"] == {"fusion": "rrf"}
        assert len(payload["prefetch"]) == 2
        return {"result": {"points": [{"id": "x", "payload": {"text": "t"}}]}}

    monkeypatch.setattr(qs_mod, "request_json", fake_request)
    store = QdrantStore(collection="c", url="http://localhost:6333")
    points = store.hybrid_search([0.1], {"indices": [3], "values": [0.5]}, top_k=5)
    assert points and points[0]["id"] == "x"
