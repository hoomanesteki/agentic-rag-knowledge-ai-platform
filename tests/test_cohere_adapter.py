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
