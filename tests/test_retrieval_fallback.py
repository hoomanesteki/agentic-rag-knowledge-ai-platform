"""Retrieval degrades instead of failing when a hosted provider is down.

The dense embedder (Cohere) and the reranker (Cohere) are hosted and metered. If either is over
quota or unreachable, the assistant must still answer on the local sparse (BM25) leg rather than
lose the turn. These tests pin that fallback so it cannot regress.
"""
from adapters.factory import make_embedder, make_store
from pipeline.answer import retrieve
from retrieval.sparse import SparseEncoder

_DOCS = [
    {"id": "d1", "text": "The Aster Flow Legging is a compressive high-rise legging for yoga."},
    {"id": "d2", "text": "The Aster Storm Shell Jacket is a waterproof rain jacket for commutes."},
]


def _seed():
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    dense = embedder.embed([d["text"] for d in _DOCS])
    store.upsert([
        {"id": d["id"], "text": d["text"], "payload": {"doc_type": "product", "text": d["text"]},
         "dense": dv, "sparse": {"indices": encoder.encode(d["text"]).indices,
                                 "values": encoder.encode(d["text"]).values}}
        for d, dv in zip(_DOCS, dense)])
    return store


class _DeadEmbedder:
    dim = 384

    def embed(self, texts, input_type="document"):
        raise RuntimeError("Cohere 429: quota exhausted")


class _DeadReranker:
    def rerank(self, query, documents, top_n=8):
        raise RuntimeError("Cohere rerank unavailable")


def test_dense_embedder_failure_falls_back_to_sparse_retrieval():
    hits = retrieve("legging for yoga", _DeadEmbedder(), make_embedder("fake") and _seed(), top_k=2)
    assert hits, "retrieval must still return results on the sparse leg"
    assert hits[0]["id"] == "d1", "sparse retrieval should still rank the legging first"


def test_reranker_failure_falls_back_to_hybrid_order():
    store = _seed()
    hits = retrieve("waterproof rain jacket", make_embedder("fake"), store, top_k=2,
                    reranker=_DeadReranker())
    assert hits, "retrieval must survive a reranker outage by keeping the hybrid order"
    assert any(h["id"] == "d2" for h in hits)
