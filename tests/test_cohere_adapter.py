"""Cohere embedder and reranker: HTTP mocked, so this asserts the real request/response shape
(v2 API) offline, without a key."""
import pytest

import adapters.cohere as cohere_mod
from adapters.cohere import CohereEmbedder, CohereReranker


def test_embed_maps_input_type_and_parses_v2_shape(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload, headers=headers)
        return {"embeddings": {"float": [[float(i)] for i, _ in enumerate(payload["texts"])]}}

    monkeypatch.setattr(cohere_mod, "request_json", fake)
    out = CohereEmbedder(model="embed-v4.0", api_key="k").embed(["a", "b", "c"], input_type="query")
    assert out == [[0.0], [1.0], [2.0]]
    assert seen["url"].endswith("/v2/embed")
    assert seen["payload"]["model"] == "embed-v4.0"
    assert seen["payload"]["input_type"] == "search_query"  # our "query" maps to Cohere's name
    assert seen["payload"]["embedding_types"] == ["float"]
    assert seen["payload"]["output_dimension"] == 1536  # v4 gets an explicit size
    assert seen["headers"]["Authorization"] == "Bearer k"


def test_embed_document_type_and_v3_omits_output_dimension(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(payload=payload)
        return {"embeddings": {"float": [[1.0] for _ in payload["texts"]]}}

    monkeypatch.setattr(cohere_mod, "request_json", fake)
    CohereEmbedder(model="embed-multilingual-v3.0", api_key="k").embed(["a"], input_type="document")
    assert seen["payload"]["input_type"] == "search_document"
    assert "output_dimension" not in seen["payload"]  # only v4 accepts a chosen size


def test_embed_batches_at_96(monkeypatch):
    calls = {"n": 0}

    def fake(method, url, payload=None, headers=None, timeout=60):
        calls["n"] += 1
        return {"embeddings": {"float": [[0.0] for _ in payload["texts"]]}}

    monkeypatch.setattr(cohere_mod, "request_json", fake)
    out = CohereEmbedder(model="embed-v4.0", api_key="k").embed([str(i) for i in range(200)])
    assert calls["n"] == 3  # 96 + 96 + 8
    assert len(out) == 200


def test_embed_dim_property():
    assert CohereEmbedder(model="embed-v4.0", api_key="k").dim == 1536
    assert CohereEmbedder(model="embed-multilingual-v3.0", api_key="k").dim == 1024


def test_embed_missing_key_raises():
    emb = CohereEmbedder(model="embed-v4.0", api_key="x")
    emb.api_key = ""
    with pytest.raises(RuntimeError):
        emb.embed(["a"])


def test_rerank_sends_shape_and_sorts_by_score(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload)
        # return out of order, to prove we sort by score
        return {"results": [{"index": 0, "relevance_score": 0.1},
                            {"index": 2, "relevance_score": 0.9},
                            {"index": 1, "relevance_score": 0.5}]}

    monkeypatch.setattr(cohere_mod, "request_json", fake)
    ranked = CohereReranker(model="rerank-v3.5", api_key="k").rerank("q", ["a", "b", "c"], top_n=3)
    assert ranked == [(2, 0.9), (1, 0.5), (0, 0.1)]
    assert seen["url"].endswith("/v2/rerank")
    assert seen["payload"]["model"] == "rerank-v3.5"
    assert seen["payload"]["query"] == "q"
    assert seen["payload"]["top_n"] == 3


def test_rerank_empty_documents_returns_empty():
    assert CohereReranker(model="rerank-v3.5", api_key="k").rerank("q", [], top_n=3) == []


def test_rerank_missing_key_raises():
    rr = CohereReranker(model="rerank-v3.5", api_key="x")
    rr.api_key = ""
    with pytest.raises(RuntimeError):
        rr.rerank("q", ["a"])


def test_unknown_non_v4_model_raises_at_construction():
    # guessing a dim would risk a silent Qdrant mismatch, so an unknown fixed-size model fails fast
    with pytest.raises(ValueError):
        CohereEmbedder(model="embed-mystery-v9", api_key="k")


def test_future_v4_model_is_allowed_and_pins_1536():
    emb = CohereEmbedder(model="embed-v4.1", api_key="k")  # unknown but v4, size is ours to pick
    assert emb.dim == 1536


def test_unknown_input_type_raises():
    # validated before any HTTP call, so no mock is needed
    with pytest.raises(ValueError):
        CohereEmbedder(model="embed-v4.0", api_key="k").embed(["a"], input_type="passage")


# --- The two-key fallback (_post): trial key -> paid key -> (caller) local sparse ----------------
# This is the resilience feature layer 2 of the embeddings and reranker chains; without a test a
# regression here would pass make check while the documented fallback silently dies.

@pytest.fixture(autouse=True)
def _clear_dead_keys():
    cohere_mod._dead_keys.clear()
    yield
    cohere_mod._dead_keys.clear()


def _router(behavior):
    """Fake request_json that dispatches on the bearer key and records the keys it saw."""
    calls = []

    def fake(method, url, payload=None, headers=None, timeout=60):
        key = headers["Authorization"].removeprefix("Bearer ")
        calls.append(key)
        outcome = behavior[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake.calls = calls
    return fake


def test_post_rolls_over_to_the_fallback_on_429(monkeypatch):
    fake = _router({"trial": RuntimeError("HTTP 429 rate limited"), "paid": {"ok": True}})
    monkeypatch.setattr(cohere_mod, "request_json", fake)
    assert cohere_mod._post("u", {}, "trial", "paid") == {"ok": True}
    assert fake.calls == ["trial", "paid"]  # capped trial key, then the paid key
    assert "trial" not in cohere_mod._dead_keys  # a quota 429 can reset, so it is not memoized


def test_post_rolls_over_on_401_and_remembers_the_dead_key(monkeypatch):
    fake = _router({"trial": RuntimeError("HTTP 401 unauthorized"), "paid": {"ok": True}})
    monkeypatch.setattr(cohere_mod, "request_json", fake)
    assert cohere_mod._post("u", {}, "trial", "paid") == {"ok": True}
    assert "trial" in cohere_mod._dead_keys  # a rejected key stays rejected
    # a second call must SKIP the known-bad trial key and go straight to the paid key
    assert cohere_mod._post("u", {}, "trial", "paid") == {"ok": True}
    assert fake.calls == ["trial", "paid", "paid"]  # the trial key is tried once, never retried


def test_post_does_not_roll_over_on_400(monkeypatch):
    fake = _router({"trial": RuntimeError("HTTP 400 bad request")})
    monkeypatch.setattr(cohere_mod, "request_json", fake)
    with pytest.raises(RuntimeError):
        cohere_mod._post("u", {}, "trial", "paid")  # a bad request would fail on the paid key too
    assert fake.calls == ["trial"]
    assert "trial" not in cohere_mod._dead_keys


def test_post_without_a_fallback_key_raises(monkeypatch):
    fake = _router({"trial": RuntimeError("HTTP 429 rate limited")})
    monkeypatch.setattr(cohere_mod, "request_json", fake)
    with pytest.raises(RuntimeError):
        cohere_mod._post("u", {}, "trial", "")  # no fallback configured, nowhere to roll over
    assert fake.calls == ["trial"]
