"""M1.1 adapter seam: the fakes are swappable and answer a query fully offline, no keys.
This is what makes the local-vs-hosted portability claim real."""
from adapters import Chunk
from adapters.factory import make_embedder, make_llm, make_vector_store


def test_offline_retrieval_roundtrip():
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


def test_upsert_is_idempotent_by_id():
    embedder = make_embedder("fake")
    store = make_vector_store("memory")
    doc = [Chunk(id="c1", text="hello", metadata={})]
    store.upsert(doc, embedder.embed(["hello"]))
    store.upsert(doc, embedder.embed(["hello"]))  # same id again
    hits = store.search(embedder.embed(["hello"])[0], top_k=10)
    assert len([h for h in hits if h.id == "c1"]) == 1


def test_llm_fake_runs_offline():
    out = make_llm("fake").complete("hello", system="be brief")
    assert isinstance(out, str) and out
