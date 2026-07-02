"""M6.4 gate and bounded agent loop, offline on the fakes. Proves a confident turn is auto, a
question with no relevant evidence escalates immediately, a borderline one runs the bounded loop
and then escalates, and an unresolved conflict escalates while keeping the governed value.
"""
from adapters.base import LLMResult
from adapters.factory import make_embedder, make_llm, make_store
from data.lakehouse import build_lakehouse
from data.metrics import MetricResolver
from rag.agent import answer_with_agent, decide_tier
from retrieval.sparse import SparseEncoder


class SlotWrongLLM:
    """Slot-fills the metric, then answers with the review's wrong number under a conflict."""

    model = "fake"

    def generate(self, prompt, *, system=None, max_tokens=512):
        text = ('{"metric": "return_rate_by_size", "params": {"size": "M"}}'
                if "Available metrics:" in prompt else "the return rate is 0.9 [2]")
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")


def _store_with(text):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    dense = embedder.embed([text])[0]
    sparse = encoder.encode(text)
    store.upsert([{"id": "D1", "text": text, "payload": {"doc_type": "review"},
                   "dense": dense, "sparse": {"indices": sparse.indices, "values": sparse.values}}])
    return embedder, store


def _resolver(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    return MetricResolver("apparel_ecommerce", db)


def test_gate_auto_for_confident_answer(tmp_path):
    embedder, store = _store_with("the flow legging runs small so size up one")
    result = answer_with_agent("does the flow legging run small",
                               components={"embedder": embedder, "store": store,
                                           "llm": make_llm("fake")},
                               trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "auto" and result.trace["agent_steps"] == 0


def test_gate_escalates_when_nothing_relevant(tmp_path):
    embedder, store = _store_with("the flow legging runs small")
    result = answer_with_agent("what is the boiling point of water",
                               components={"embedder": embedder, "store": store,
                                           "llm": make_llm("fake")},
                               trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "escalate" and result.trace["agent_steps"] == 0
    assert result.citations == [] and "flagged" in result.answer


def test_borderline_question_runs_loop_then_escalates(tmp_path):
    # one shared content word out of five: below the abstain threshold but not zero, so the agent
    # loop runs to its cap and then escalates
    embedder, store = _store_with("the flow legging is comfortable")
    result = answer_with_agent("flow boiling melting freezing point",
                               components={"embedder": embedder, "store": store,
                                           "llm": make_llm("fake")},
                               max_steps=2, trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "escalate"
    # the loop ran, saw the reformulation surface no new evidence, and stopped early (bounded)
    assert result.trace["agent_steps"] == 1


def test_unresolved_conflict_escalates_with_governed_value(tmp_path):
    embedder, store = _store_with("the return rate for size M is 0.9 according to a shopper")
    result = answer_with_agent("what is the return rate for size M",
                               components={"embedder": embedder, "store": store,
                                           "llm": SlotWrongLLM(),
                                           "metric_resolver": _resolver(tmp_path)},
                               trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "escalate"  # a conflict the model got wrong needs a human
    assert result.trace["conflict_resolved"] is False
    assert "0.5" in result.answer  # but the governed value is what shipped, not the guess


def test_decide_tier_cases():
    base = {"confidence": 0.8, "grounding": 0.5, "step": 0, "max_steps": 2}
    assert decide_tier("auto", conflict_resolved=True, **base) == "auto"
    assert decide_tier("auto", conflict_resolved=False, **base) == "escalate"
    assert decide_tier("abstain", conflict_resolved=True, confidence=0.0, grounding=0.0,
                       step=0, max_steps=2) == "escalate"
    assert decide_tier("abstain", conflict_resolved=True, confidence=0.2, grounding=0.0,
                       step=0, max_steps=2) == "agent"
    assert decide_tier("abstain", conflict_resolved=True, confidence=0.2, grounding=0.0,
                       step=2, max_steps=2) == "escalate"
    # a weakly-grounded auto answer takes another pass when a grounding floor is set
    assert decide_tier("auto", conflict_resolved=True, confidence=0.8, grounding=0.1,
                       step=0, max_steps=2, min_grounding=0.5) == "agent"
