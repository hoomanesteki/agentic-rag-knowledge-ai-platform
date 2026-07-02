"""M6.5 human-in-the-loop: the durable review queue, escalation wiring, and the LangGraph
checkpointer. Proves an escalated question lands in the queue, a human answer closes it (once),
the answer is stored as verified knowledge, and a checkpointed graph run's state survives.
"""
import json

from langgraph.checkpoint.memory import MemorySaver

from adapters.factory import make_embedder, make_llm, make_store
from rag.agent import answer_with_agent
from rag.hitl import ReviewQueue
from rag.supervisor import build_supervisor_graph
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
    item_id = queue.enqueue("what is the SLA?", domain="saas_support", route="factual", now=1.0)
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
    item_id = queue.enqueue("how do I export data?", domain="saas_support", now=1.0)
    queue.resolve(item_id, "Settings, Data, Export.", "operator", now=2.0)
    closed = queue.closed_since(0.0)
    assert len(closed) == 1 and closed[0]["answer"] == "Settings, Data, Export."


def test_resolve_writes_verified_knowledge(tmp_path):
    verified = tmp_path / "verified.jsonl"
    queue = ReviewQueue(str(tmp_path / "rq.db"), verified_path=str(verified))
    item_id = queue.enqueue("how do I export data?", domain="saas_support")
    queue.resolve(item_id, "Go to Settings, Data, Export.", "operator")
    rows = [json.loads(x) for x in verified.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["source"] == "hitl" and rows[0]["answer"].startswith("Go to Settings")


def test_escalation_enqueues_and_a_human_closes_it(tmp_path):
    embedder, store = _store_with("the flow legging runs small")
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    result = answer_with_agent("what is the boiling point of water",
                               components={"embedder": embedder, "store": store,
                                           "llm": make_llm("fake")},
                               review_queue=queue, domain="apparel_ecommerce",
                               trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "escalate"
    assert result.trace["escalation_id"] is not None
    open_items = queue.list_open()
    assert len(open_items) == 1 and open_items[0]["id"] == result.trace["escalation_id"]
    assert queue.resolve(open_items[0]["id"], "Around 100 C at sea level.", "operator") is True
    assert queue.list_open() == []


def test_auto_answer_does_not_enqueue(tmp_path):
    embedder, store = _store_with("the flow legging runs small so size up one")
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    result = answer_with_agent("does the flow legging run small",
                               components={"embedder": embedder, "store": store,
                                           "llm": make_llm("fake")},
                               review_queue=queue, domain="apparel_ecommerce",
                               trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "auto"
    assert result.trace["escalation_id"] is None and queue.list_open() == []


def test_checkpointer_persists_turn_state(tmp_path):
    embedder, store = _store_with("the flow legging runs small so size up one")
    components = {"embedder": embedder, "store": store, "llm": make_llm("fake")}
    graph = build_supervisor_graph(components, trace_path=str(tmp_path / "t.jsonl"),
                                   checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    state = graph.invoke({"query": "does the flow legging run small", "history": [],
                          "message_id": "m1"}, config=config)
    saved = graph.get_state(config)
    assert saved.values["answer"] == state["answer"]  # the run's state survives, so it can resume
