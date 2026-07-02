"""M7.3 flywheel: a resolved human answer becomes retrievable and grows the verified eval set,
and thumbs suggest a gate threshold."""
import json

from adapters.factory import make_embedder, make_store
from pipeline.answer import retrieve
from rag.flywheel import grow_verified_eval, reindex_verified, suggest_threshold


def _item(item_id, question, answer):
    return {"id": item_id, "question": question, "answer": answer, "answered_by": "operator"}


def test_reindexed_answer_is_retrievable():
    embedder, store = make_embedder("fake"), make_store("memory")
    items = [_item("a1", "how long is the refund window", "Refunds are accepted within 30 days.")]
    assert reindex_verified(items, embedder, store) == 1
    hits = retrieve("how long is the refund window", embedder, store)
    assert any(h["id"] == "verified:a1" for h in hits)  # the human answer is now retrievable


def test_reindex_is_idempotent():
    embedder, store = make_embedder("fake"), make_store("memory")
    items = [_item("a1", "q", "a")]
    reindex_verified(items, embedder, store)
    reindex_verified(items, embedder, store)  # same id overwrites, not duplicates
    hits = retrieve("q", embedder, store)
    assert sum(1 for h in hits if h["id"] == "verified:a1") == 1


def test_grow_verified_eval_appends_once(tmp_path):
    path = str(tmp_path / "verified.jsonl")
    items = [_item("a1", "what is the SLA", "99.9 percent")]
    assert grow_verified_eval(items, path) == 1
    assert grow_verified_eval(items, path) == 0  # already written, not duplicated
    rows = [json.loads(x) for x in open(path).read().splitlines() if x.strip()]
    assert len(rows) == 1 and rows[0]["type"] == "answerable" and rows[0]["source"] == "hitl"


def test_suggest_threshold_reacts_to_thumbs():
    base = 0.34
    assert suggest_threshold({"overall": {"thumbs_up": 1, "thumbs_down": 1}}, base)[
        "suggested"] == base  # too few thumbs -> hold
    up = suggest_threshold({"overall": {"thumbs_up": 2, "thumbs_down": 8}}, base)
    assert up["suggested"] > base and up["down_rate"] == 0.8  # many downs -> raise
    down = suggest_threshold({"overall": {"thumbs_up": 19, "thumbs_down": 1}}, base)
    assert down["suggested"] < base  # rating well -> slightly more permissive
