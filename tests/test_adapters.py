"""M1.1/M1.2 adapter seams: the fakes are swappable and answer offline with no keys.
This is what makes the local-vs-hosted portability claim real."""
from adapters import Chunk
from adapters.factory import make_embedder, make_llm, make_store, make_vector_store
from retrieval.sparse import SparseEncoder


def test_offline_dense_retrieval_roundtrip():
    embedder = make_embedder("fake")
    store = make_vector_store("memory")
    docs = [
        Chunk(id="c1", text="the flow legging runs small so size up", metadata={"lang": "en"}),
        Chunk(id="c2", text="the belt bag is great for travel", metadata={"lang": "en"}),
        Chunk(id="c3", text="the tee is breathable for hot yoga", metadata={"lang": "en"}),
    ]
    store.upsert(docs, embedder.embed([d.text for d in docs]))
    query = embedder.embed(["does the legging run small"])[0]
    hits = store.search(query, top_k=1)
    assert hits and hits[0].id == "c1"


def test_metadata_filter_restricts_results():
    embedder = make_embedder("fake")
    store = make_vector_store("memory")
    docs = [
        Chunk(id="en1", text="soft hoodie", metadata={"lang": "en"}),
        Chunk(id="fr1", text="hoodie doux", metadata={"lang": "fr"}),
    ]
    store.upsert(docs, embedder.embed([d.text for d in docs]))
    query = embedder.embed(["hoodie"])[0]
    hits = store.search(query, top_k=5, where={"lang": "fr"})
    assert [h.id for h in hits] == ["fr1"]


def test_embedder_accepts_input_type():
    embedder = make_embedder("fake")
    assert embedder.embed(["x"], input_type="query")  # asymmetric-model kwarg is accepted


def test_llm_fake_runs_offline():
    result = make_llm("fake").generate("hello", system="be brief")
    assert result.text and result.model == "fake"


def test_offline_hybrid_store_roundtrip():
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    docs = [
        {"id": "c1", "text": "the flow legging runs small", "payload": {"lang": "en"}},
        {"id": "c2", "text": "the belt bag is great for travel", "payload": {"lang": "en"}},
    ]
    dense = embedder.embed([d["text"] for d in docs])
    points = [
        {**d, "dense": dv, "sparse": {"indices": sv.indices, "values": sv.values}}
        for d, dv, sv in zip(docs, dense, [encoder.encode(d["text"]) for d in docs])
    ]
    store.upsert(points)
    dense_q = embedder.embed(["does the legging run small"], input_type="query")[0]
    sparse_q = encoder.encode("does the legging run small")
    hits = store.hybrid_search(dense_q, {"indices": sparse_q.indices, "values": sparse_q.values},
                               top_k=1)
    assert hits and hits[0]["id"] == "c1"
    assert hits[0]["payload"]["text"] == "the flow legging runs small"
