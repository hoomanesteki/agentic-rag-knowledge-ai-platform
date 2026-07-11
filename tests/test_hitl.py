"""M6.5 human-in-the-loop: the durable review queue and escalation wiring. Proves an escalated
question lands in the queue, a human answer closes it (once), and the answer is stored as verified
knowledge. Escalation is driven through the omni orchestrator's LLM-free handoff.
"""
import json

from adapters.factory import make_embedder, make_llm, make_store
from rag.hitl import ReviewQueue
from rag.omni import stream_omni
from retrieval.sparse import SparseEncoder


def _store_with(text):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    dense = embedder.embed([text])[0]
    sparse = encoder.encode(text)
    store.upsert([{"id": "D1", "text": text, "payload": {"doc_type": "review"},
                   "dense": dense, "sparse": {"indices": sparse.indices, "values": sparse.values}}])
    return embedder, store


def test_queue_enqueue_list_and_resolve(tmp_path):
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    item_id = queue.enqueue("what is the SLA?", domain="apparel_ecommerce", route="factual",
                            now=1.0)
    assert [i["id"] for i in queue.list_open()] == [item_id]

    assert queue.resolve(item_id, "The SLA is 99.9 percent.", "operator", now=2.0) is True
    assert queue.list_open() == []                      # closed, no longer open
    assert queue.resolve(item_id, "again", "operator2") is False  # claim-safe: only once
    item = queue.get(item_id)
    assert item["status"] == "closed" and item["answered_by"] == "operator"


def test_retry_with_same_message_id_does_not_duplicate(tmp_path):
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    first = queue.enqueue("what is the SLA?", message_id="m1")
    again = queue.enqueue("what is the SLA?", message_id="m1")  # a retry of the same turn
    assert first == again and len(queue.list_open()) == 1


def test_closed_items_are_re_exportable_from_the_db(tmp_path):
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    item_id = queue.enqueue("how do I export data?", domain="apparel_ecommerce", now=1.0)
    queue.resolve(item_id, "Settings, Data, Export.", "operator", now=2.0)
    closed = queue.closed_since(0.0)
    assert len(closed) == 1 and closed[0]["answer"] == "Settings, Data, Export."


def test_lang_flows_through_the_queue_to_closed_items(tmp_path):
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    item_id = queue.enqueue("quelle est la limite", domain="apparel_ecommerce", lang="fr", now=1.0)
    queue.resolve(item_id, "100 par minute", "operator", now=2.0)
    assert queue.closed_since(0.0)[0]["lang"] == "fr"  # lang is available for the flywheel


def test_closed_since_filters_by_domain_and_tracks_a_watermark(tmp_path):
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    a = queue.enqueue("qa", domain="A")
    queue.resolve(a, "answer a", "op", now=10.0)
    b = queue.enqueue("qb", domain="B")
    queue.resolve(b, "answer b", "op", now=20.0)
    assert [i["domain"] for i in queue.closed_since(0.0, domain="A")] == ["A"]  # only domain A
    assert queue.flywheel_watermark("A") == 0.0
    queue.advance_flywheel_watermark("A", 10.0)
    assert queue.flywheel_watermark("A") == 10.0  # persisted per domain


def test_resolve_writes_verified_knowledge(tmp_path):
    verified = tmp_path / "verified.jsonl"
    queue = ReviewQueue(str(tmp_path / "rq.db"), verified_path=str(verified))
    item_id = queue.enqueue("how do I export data?", domain="apparel_ecommerce")
    queue.resolve(item_id, "Go to Settings, Data, Export.", "operator")
    rows = [json.loads(x) for x in verified.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["source"] == "hitl" and rows[0]["answer"].startswith("Go to Settings")


def _final(events):
    return [e for e in events if e.get("type") == "final"][-1]


def test_escalation_enqueues_and_a_human_closes_it(tmp_path):
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    events = list(stream_omni("I want to talk to a human please", embedder=None, store=None,
                              llm=None, review_queue=queue, domain="apparel_ecommerce",
                              trace_path=str(tmp_path / "t.jsonl")))
    final = _final(events)
    assert final["tier"] == "escalate"
    open_items = queue.list_open()
    assert len(open_items) == 1 and open_items[0]["id"] == final["escalation_id"]
    assert queue.resolve(open_items[0]["id"], "A specialist will follow up.", "operator") is True
    assert queue.list_open() == []


def test_auto_answer_does_not_enqueue(tmp_path):
    embedder, store = _store_with("the flow legging runs small so size up one")
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    events = list(stream_omni("does the flow legging run small", embedder=embedder, store=store,
                              llm=make_llm("fake"), review_queue=queue,
                              domain="apparel_ecommerce", trace_path=str(tmp_path / "t.jsonl")))
    assert _final(events)["tier"] != "escalate"
    assert queue.list_open() == []
