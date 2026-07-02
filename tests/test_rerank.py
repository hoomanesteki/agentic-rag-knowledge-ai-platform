"""M2.2 reranker: the offline fake reorders, the Voyage client shape is right, and retrieve()
applies the reranker over a wider pool with bounds-safe index mapping."""
import pytest

import adapters.voyage_rerank as vr_mod
from adapters.base import Chunk
from adapters.factory import make_embedder, make_reranker, make_store
from adapters.fakes import LexicalReranker
from adapters.voyage_rerank import VoyageReranker
from pipeline.answer import retrieve
from retrieval.sparse import SparseEncoder


class ReverseReranker:
    """Returns the fetched docs in reversed order, so a test can prove the reranker is applied
    regardless of the store's own ranking."""

    def rerank(self, query, documents, top_n=8):
        n = len(documents)
        return [(i, float(n - i)) for i in reversed(range(n))][:top_n]


class BadReranker:
    def rerank(self, query, documents, top_n=8):
        return [(99, 1.0)]  # out-of-range index


def test_lexical_reranker_orders_by_overlap():
    docs = ["nothing relevant here", "the flow legging runs small", "a belt bag for travel"]
    ranked = LexicalReranker().rerank("legging runs small", docs, top_n=2)
    assert len(ranked) == 2
    assert ranked[0][0] == 1
    assert ranked[0][1] >= ranked[1][1]


def test_voyage_reranker_shape(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload, headers=headers)
        return {"data": [{"index": 0, "relevance_score": 0.4},
                         {"index": 2, "relevance_score": 0.9}]}  # unsorted on purpose

    monkeypatch.setattr(vr_mod, "request_json", fake)
    ranked = VoyageReranker(model="rerank-2.5", api_key="k").rerank("q", ["a", "b", "c"], top_n=2)
    assert ranked == [(2, 0.9), (0, 0.4)]  # sorted by score, not array order
    assert seen["url"].endswith("/rerank")
    assert seen["payload"]["model"] == "rerank-2.5"
    assert seen["payload"]["query"] == "q"
    assert seen["payload"]["documents"] == ["a", "b", "c"]
    assert seen["payload"]["top_k"] == 2
    assert seen["headers"]["Authorization"] == "Bearer k"


def test_voyage_reranker_empty_docs_makes_no_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not call the API for empty documents")

    monkeypatch.setattr(vr_mod, "request_json", boom)
    assert VoyageReranker(api_key="k").rerank("q", []) == []


def test_voyage_reranker_missing_key_raises():
    client = VoyageReranker(api_key="placeholder")
    client.api_key = ""  # force empty regardless of any .env on the machine
    with pytest.raises(RuntimeError):
        client.rerank("q", ["a"])


def test_make_reranker_from_provider():
    assert make_reranker("none") is None
    assert isinstance(make_reranker("fake"), LexicalReranker)


def _seed(docs):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    dense = embedder.embed([d.text for d in docs])
    points = [
        {"id": d.id, "text": d.text, "payload": d.metadata,
         "dense": dv, "sparse": {"indices": sv.indices, "values": sv.values}}
        for d, dv, sv in zip(docs, dense, [encoder.encode(d.text) for d in docs])
    ]
    store.upsert(points)
    return embedder, store


def _docs():
    return [
        Chunk(id="c1", text="alpha one", metadata={"product_id": "P1"}),
        Chunk(id="c2", text="beta two", metadata={"product_id": "P2"}),
        Chunk(id="c3", text="gamma three", metadata={"product_id": "P3"}),
    ]


def test_reranker_changes_order_and_replaces_score():
    embedder, store = _seed(_docs())
    base = [h["payload"]["chunk_id"] for h in retrieve("alpha", embedder, store, top_k=3)]
    reranked = retrieve("alpha", embedder, store, top_k=3, reranker=ReverseReranker(), top_k_in=50)
    ids = [h["payload"]["chunk_id"] for h in reranked]
    assert ids == list(reversed(base))          # the reranker's permutation is applied
    assert reranked[0]["score"] == reranked[0]["rerank_score"]  # score replaced by rerank score


def test_reranker_widens_pool_beyond_top_k():
    embedder, store = _seed(_docs())
    base_top = retrieve("alpha", embedder, store, top_k=1)[0]["payload"]["chunk_id"]
    reranked_top = retrieve("alpha", embedder, store, top_k=1,
                            reranker=ReverseReranker(), top_k_in=50)[0]["payload"]["chunk_id"]
    assert base_top != reranked_top  # reranker saw the wider pool, not just the hybrid top-1


def test_out_of_range_index_raises():
    embedder, store = _seed(_docs())
    with pytest.raises(RuntimeError):
        retrieve("alpha", embedder, store, top_k=1, reranker=BadReranker())


def test_none_reranker_matches_plain_hybrid():
    embedder, store = _seed(_docs())
    without = [h["payload"]["chunk_id"] for h in retrieve("beta", embedder, store, top_k=2)]
    assert len(without) == 2  # unchanged M2.1 behavior, no error
