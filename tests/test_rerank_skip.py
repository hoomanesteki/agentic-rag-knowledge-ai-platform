"""The rerank-skip gate drops the single largest cost line of a text turn only where reranking
cannot change the answer, and stays OFF by default so it is opt-in. These pin the decision logic
and prove a skip actually avoids the paid rerank call, with the reason surfaced for the trace."""
from adapters.factory import make_embedder, make_store
from pipeline.answer import _rerank_skip_reason, retrieve
from retrieval.sparse import SparseEncoder


def test_off_by_default_even_for_a_skippable_query(monkeypatch):
    monkeypatch.delenv("RERANK_SKIP", raising=False)
    assert _rerank_skip_reason("where is my order", [{"score": 0.5}] * 3, 8) is None


def test_own_order_lookup_skips(monkeypatch):
    monkeypatch.setenv("RERANK_SKIP", "on")
    # an identity-gated own-order set is tiny and self-relevant, so reranking it is wasted spend
    assert _rerank_skip_reason("where is my order", [{"score": 0.5}] * 20, 8) == "own-order lookup"


def test_small_pool_skips(monkeypatch):
    monkeypatch.setenv("RERANK_SKIP", "on")
    # a pool no larger than the cut has nothing to prune; reranking can only reorder marginally
    assert _rerank_skip_reason("do you sell leggings", [{"score": 0.5}] * 3, 8) == "pool<=top_k"


def test_a_dominant_top_hit_still_reranks(monkeypatch):
    # the clean-margin skip was REJECTED by the golden calibration (it halved abstain_recall), so
    # even a query whose rank 1 dominates the cut must still rerank on a wide pool
    monkeypatch.setenv("RERANK_SKIP", "on")
    hits = [{"score": 1.0}] + [{"score": 0.2}] * 11  # would have tripped clean-margin
    assert _rerank_skip_reason("do you sell leggings", hits, 8) is None


class _CountingReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, texts, top_n):
        self.calls += 1
        return [(i, 1.0 - 0.01 * i) for i in range(min(top_n, len(texts)))]


def _seed_store(n=3):
    embedder, encoder, store = make_embedder("fake"), SparseEncoder(), make_store("memory")
    docs = [{"id": "D{}".format(i), "text": "leggings option {}".format(i),
             "payload": {"doc_type": "review", "text": "leggings option {}".format(i)}}
            for i in range(n)]
    dense = embedder.embed([d["text"] for d in docs])
    store.upsert([{**docs[i], "dense": dense[i],
                   "sparse": {"indices": encoder.encode(docs[i]["text"]).indices,
                              "values": encoder.encode(docs[i]["text"]).values}}
                  for i in range(n)])
    return embedder, store


def test_skip_avoids_the_paid_rerank_call_and_records_the_reason(monkeypatch):
    monkeypatch.setenv("RERANK_SKIP", "on")
    embedder, store = _seed_store(3)  # 3 hits <= top_k 8 -> pool<=top_k skip
    rr = _CountingReranker()
    out: dict = {}
    retrieve("do you sell leggings", embedder, store, top_k=8, reranker=rr,
             rerank_skip_out=out)
    assert rr.calls == 0 and out.get("skipped") is True and out["reason"] == "pool<=top_k"


def test_disabled_gate_reranks_normally(monkeypatch):
    monkeypatch.delenv("RERANK_SKIP", raising=False)
    embedder, store = _seed_store(3)
    rr = _CountingReranker()
    retrieve("do you sell leggings", embedder, store, top_k=8, reranker=rr)
    assert rr.calls == 1  # gate off: the reranker runs as before
