"""The LangGraph corrective-RAG lane (CHAT_BRAIN=graph, rag/graph_brain.py).

These tests stub retrieve + stream_answer to drive the graph's ORCHESTRATION deterministically and
offline: the happy path, the document-grade loop, the generation-grade loop, the must-terminate
guarantee (no runaway), budget safety, the SSE contract, and that the PII-gated auth_text is
threaded down to retrieval. PII / injection / grounding themselves live in stream_answer
and are covered by the safety tests; here we prove the graph runs every answer through that one
gated surface and is bounded. Skipped cleanly on a base install without LangGraph.
"""
import pytest

pytest.importorskip("langgraph")

import rag.graph_brain as gb  # noqa: E402
from adapters.budget import BudgetExceeded  # noqa: E402


def _hit(text, doc_type=None, chunk_id="c1"):
    return {"id": chunk_id, "score": 1.0,
            "payload": {"text": text, "chunk_id": chunk_id, "doc_type": doc_type}}


class _Answer:
    """Stub stream_answer: record each call, yield a token + a final with a fixed verdict."""

    def __init__(self, tier="auto", grounding=1.0, answer="Recycled nylon and polyester [1]."):
        self.tier, self.grounding, self.answer = tier, grounding, answer
        self.calls = []

    def __call__(self, question, **kwargs):
        self.calls.append({"question": question, **kwargs})
        yield {"type": "token", "text": self.answer}
        yield {"type": "final", "message_id": kwargs.get("message_id", "m"), "answer": self.answer,
               "tier": self.tier, "confidence": 0.9, "grounding": self.grounding, "citations": []}


def _wire(monkeypatch, *, hits, answer, rewrite=lambda q, h, llm: q):
    captured = {"retrieve": []}

    def fake_retrieve(question, embedder, store, *, top_k=8, reranker=None, top_k_in=50,
                      auth_text=None, lang=None, **kw):
        captured["retrieve"].append({"question": question, "auth_text": auth_text})
        return list(hits)

    monkeypatch.setattr(gb, "retrieve", fake_retrieve)
    monkeypatch.setattr(gb, "stream_answer", answer)
    monkeypatch.setattr(gb, "rewrite_followup", rewrite)
    return captured


def _run(query="what are the leggings made of", **kw):
    return list(gb.stream_graph(query, embedder=None, store=None, llm=object(), **kw))


def test_happy_path_grounded_answer_ends(monkeypatch):
    ans = _Answer(tier="auto", grounding=1.0)
    _wire(monkeypatch, hits=[_hit("leggings made of recycled nylon polyester")], answer=ans)
    events = _run()
    assert [e["type"] for e in events] == ["token", "final"]
    assert events[-1]["tier"] == "auto" and events[-1]["grounding"] == 1.0
    assert len(ans.calls) == 1  # relevant docs -> generate once -> useful -> END


def test_generation_grade_loop_terminates_at_abstain(monkeypatch):
    # relevant docs each pass, but every generation is weakly grounded, so the generation-grade loop
    # rewrites and regenerates up to MAX_RETRIES, then settles on the pipeline's abstain final.
    ans = _Answer(tier="abstain", grounding=0.1)
    rewrites = {"n": 0}

    def rewrite(q, h, llm):
        rewrites["n"] += 1
        return q + " more"

    _wire(monkeypatch, hits=[_hit("relevant context words leggings")], answer=ans, rewrite=rewrite)
    events = _run()
    assert events[-1]["type"] == "final" and events[-1]["tier"] == "abstain"
    assert len(ans.calls) == gb.MAX_RETRIES + 1  # first attempt + one per retry
    assert rewrites["n"] == gb.MAX_RETRIES        # exactly MAX_RETRIES corrective rewrites


def test_weak_documents_loop_then_answers(monkeypatch):
    # empty retrieval -> weak grade -> rewrite loop; after MAX_RETRIES the doc-grade gives up and
    # lets the gated pipeline answer/abstain. Must terminate, never raise GraphRecursionError.
    ans = _Answer(tier="auto", grounding=1.0)
    cap = _wire(monkeypatch, hits=[], answer=ans)
    events = _run()
    assert events[-1]["type"] == "final"
    # retrieve ran the initial pass plus one per rewrite; bounded, not runaway
    assert len(cap["retrieve"]) <= gb.MAX_RETRIES + 2


def test_budget_breach_yields_a_safe_final(monkeypatch):
    def boom(question, **kwargs):
        raise BudgetExceeded("max_calls")
        yield  # pragma: no cover - make it a generator

    _wire(monkeypatch, hits=[_hit("x")], answer=boom)
    events = _run()
    assert [e["type"] for e in events] == ["token", "final"]
    assert events[-1]["tier"] == "abstain" and events[-1]["lane"] == "budget"
    assert events[-1]["budget"]["stopped"] == "max_calls"


def test_sse_contract_shape(monkeypatch):
    _wire(monkeypatch, hits=[_hit("leggings recycled nylon")], answer=_Answer())
    events = _run()
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1
    for key in ("message_id", "answer", "tier", "confidence", "grounding", "citations"):
        assert key in finals[0]


def test_auth_text_threaded_for_pii_gate(monkeypatch):
    # anonymous + block_order_pii -> empty auth_text (no order can unlock); a signed-in shopper's
    # own name+email is threaded so retrieval's order gate can authorize their own records.
    cap = _wire(monkeypatch, hits=[_hit("x")], answer=_Answer())
    _run("where is my order", block_order_pii=True)
    assert cap["retrieve"][0]["auth_text"] == ""

    cap2 = _wire(monkeypatch, hits=[_hit("x")], answer=_Answer())
    _run("where is my order", auth_identity=("Aaron Esteki", "info@esteki.ca"))
    joined = cap2["retrieve"][0]["auth_text"].lower()
    assert "aaron" in joined and "info@esteki.ca" in joined
