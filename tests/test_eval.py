"""M2.1 eval harness: metric units and offline scorecard behavior with the fakes."""
from adapters.factory import make_embedder, make_store
from evaluation.harness import evaluate, format_scorecard
from evaluation.metrics import hit_at_k, mean, reciprocal_rank
from retrieval.sparse import SparseEncoder

ENTITY_FIELDS = ["product_id"]


def test_hit_at_k():
    assert hit_at_k([False, True, False]) == 1.0
    assert hit_at_k([False, False]) == 0.0
    assert hit_at_k([]) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, True, False]) == 0.5
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_mean():
    assert mean([1.0, 0.0]) == 0.5
    assert mean([]) == 0.0


def _seed(docs):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    dense = embedder.embed([d["text"] for d in docs])
    points = [
        {**d, "dense": dv, "sparse": {"indices": sv.indices, "values": sv.values}}
        for d, dv, sv in zip(docs, dense, [encoder.encode(d["text"]) for d in docs])
    ]
    store.upsert(points)
    return embedder, store


def _reviews():
    return [
        {"id": "R1", "text": "the flow legging runs small so size up",
         "payload": {"doc_type": "review", "product_id": "P002"}},
        {"id": "R2", "text": "the belt bag is great for travel and daily use",
         "payload": {"doc_type": "review", "product_id": "P006"}},
        {"id": "R3", "text": "an unrelated distractor about weather forecasts",
         "payload": {"doc_type": "review", "product_id": "P099"}},
    ]


def test_scoping_ranking_and_gate():
    embedder, store = _seed(_reviews())
    golden = [
        {"id": "G1", "lang": "en", "type": "answerable", "route": "qualitative",
         "question": "does the flow legging run small", "expected_entities": ["P002"]},
        {"id": "G2", "lang": "en", "type": "answerable", "route": "factual",
         "question": "how much is the legging", "expected_entities": ["P002"]},  # deferred
        {"id": "G3", "lang": "en", "type": "out_of_domain",
         "question": "what is the capital of France"},
    ]
    sc = evaluate(golden, embedder=embedder, store=store, entity_fields=ENTITY_FIELDS, top_k=8)
    assert sc["coverage"]["measured"] == 1     # only the qualitative question is measurable now
    assert sc["coverage"]["deferred"] == 1     # the factual one is deferred to M4
    r = sc["overall"]["retrieval"]
    assert r["n"] == 1
    assert r["hit_rate_at_k"] == 1.0
    assert r["entity_recall_at_k"] == 1.0
    assert r["mrr"] == 1.0                      # the relevant review ranks first
    assert r["false_abstain_rate"] == 0.0      # the answerable question is not wrongly abstained
    assert sc["overall"]["gate"]["abstain_recall"] == 1.0


def test_relevance_uses_only_entity_fields():
    # rating happens to equal the expected entity, but it is not an entity field
    docs = [{"id": "R1", "text": "nice comfortable product",
             "payload": {"doc_type": "review", "product_id": "P001", "rating": "P002"}}]
    embedder, store = _seed(docs)
    golden = [{"id": "G1", "lang": "en", "type": "answerable", "route": "qualitative",
               "question": "nice comfortable product", "expected_entities": ["P002"]}]
    sc = evaluate(golden, embedder=embedder, store=store, entity_fields=["product_id"], top_k=8)
    assert sc["overall"]["retrieval"]["hit_rate_at_k"] == 0.0  # no false positive via rating


def test_degenerate_flag_on_empty_store():
    embedder = make_embedder("fake")
    store = make_store("memory")  # nothing ingested
    golden = [{"id": "G1", "lang": "en", "type": "answerable", "route": "qualitative",
               "question": "anything at all", "expected_entities": ["P002"]}]
    sc = evaluate(golden, embedder=embedder, store=store, entity_fields=["product_id"], top_k=8)
    assert sc["degenerate"] is True


def test_false_abstain_rate_catches_wrongly_abstained_answerable():
    docs = [{"id": "R1", "text": "the flow legging runs small",
             "payload": {"doc_type": "review", "product_id": "P002"}}]
    embedder, store = _seed(docs)
    golden = [{"id": "G1", "lang": "en", "type": "answerable", "route": "qualitative",
               "question": "zzz qqq", "expected_entities": ["P002"]}]  # no overlap -> abstains
    sc = evaluate(golden, embedder=embedder, store=store, entity_fields=["product_id"], top_k=8)
    assert sc["overall"]["retrieval"]["false_abstain_rate"] == 1.0


def test_format_scorecard_smoke():
    embedder, store = _seed(_reviews())
    golden = [{"id": "G1", "lang": "en", "type": "out_of_domain",
               "question": "what is the capital of France"}]
    out = format_scorecard(
        evaluate(golden, embedder=embedder, store=store, entity_fields=["product_id"], top_k=8))
    assert out.startswith("Eval scorecard")
    assert "coverage:" in out
