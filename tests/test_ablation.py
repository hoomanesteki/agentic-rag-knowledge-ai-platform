"""M2.5 ablation: dense-only retrieval mode and the report builder (offline with fakes)."""
from adapters.factory import make_embedder, make_store
from adapters.fakes import InMemoryHybridStore
from evaluation.harness import evaluate
from evaluation.report import build_ablation_report
from retrieval.sparse import SparseEncoder


def _seed(docs):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    dense = embedder.embed([d["text"] for d in docs])
    store.upsert([
        {**d, "dense": dv, "sparse": {"indices": sv.indices, "values": sv.values}}
        for d, dv, sv in zip(docs, dense, [encoder.encode(d["text"]) for d in docs])
    ])
    return embedder, store


def _docs():
    return [
        {"id": "R1", "text": "the flow legging runs small so size up",
         "payload": {"doc_type": "review", "product_id": "P002"}},
        {"id": "R2", "text": "the belt bag is great for travel",
         "payload": {"doc_type": "review", "product_id": "P006"}},
    ]


def test_dense_only_search_returns_scored_hits():
    from pipeline.answer import retrieve
    embedder, store = _seed(_docs())
    hits = retrieve("does the legging run small", embedder, store, top_k=2, dense_only=True)
    assert hits
    assert all(isinstance(h["score"], float) for h in hits)
    assert hits[0]["payload"]["chunk_id"] == "R1"  # dense still finds the relevant doc


def test_dense_only_differs_from_hybrid():
    # Hand-built vectors: P1 wins on dense, P2 wins on sparse, F spreads the sparse ranks so
    # RRF is not a symmetric tie. dense_only must top P1; hybrid must top P2.
    store = InMemoryHybridStore()
    store.upsert([
        {"id": "P1", "text": "a", "payload": {},
         "dense": [1.0, 0.0, 0.0], "sparse": {"indices": [0], "values": [1.0]}},
        {"id": "P2", "text": "b", "payload": {},
         "dense": [0.6, 0.8, 0.0], "sparse": {"indices": [7], "values": [9.0]}},
        {"id": "F", "text": "c", "payload": {},
         "dense": [0.0, 0.0, 1.0], "sparse": {"indices": [7], "values": [2.0]}},
    ])
    q_dense = [1.0, 0.0, 0.0]
    q_sparse = {"indices": [7], "values": [9.0]}
    dense = store.hybrid_search(q_dense, q_sparse, top_k=3, dense_only=True)
    hybrid = store.hybrid_search(q_dense, q_sparse, top_k=3, dense_only=False)
    assert dense[0]["id"] == "P1"    # cosine winner, sparse ignored
    assert hybrid[0]["id"] == "P2"   # sparse pulls P2 to the top under fusion


def test_evaluate_records_dense_only_flag():
    embedder, store = _seed(_docs())
    golden = [{"id": "G1", "lang": "en", "type": "answerable", "route": "qualitative",
               "question": "does the flow legging run small", "expected_entities": ["P002"]}]
    sc = evaluate(golden, embedder=embedder, store=store, entity_fields=["product_id"],
                  dense_only=True)
    assert sc["dense_only"] is True
    assert sc["overall"]["retrieval"]["hit_rate_at_k"] == 1.0


def test_build_ablation_report_has_variants_and_table():
    embedder, store = _seed(_docs())
    golden = [{"id": "G1", "lang": "en", "type": "answerable", "route": "qualitative",
               "question": "does the flow legging run small", "expected_entities": ["P002"]}]
    results = [
        ("dense", evaluate(golden, embedder=embedder, store=store,
                           entity_fields=["product_id"], dense_only=True)),
        ("hybrid", evaluate(golden, embedder=embedder, store=store,
                            entity_fields=["product_id"])),
    ]
    report = build_ablation_report(results, domain="apparel_ecommerce", note="test")
    assert "# Retrieval ablation (apparel_ecommerce)" in report
    assert "| variant | scope |" in report
    assert "| dense | overall |" in report
    assert "| hybrid | overall |" in report
