"""The omni fast path routes a turn, then answers through the same gated pipeline. These tests stub
stream_answer so they check the ORCHESTRATION (which lane, which persona, clarify short-circuit)
without needing a live index or model. PII parity is inherited from stream_answer itself, which is
covered by the existing safety tests."""
import rag.omni as omni
from rag.roles import lane_persona, role_fragment


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, query, **kwargs):
        self.calls.append(kwargs)
        yield {"type": "final", "answer": "ok", "lane": kwargs.get("lane")}


def test_own_order_turn_routes_through_the_care_lane(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("where is my order", embedder=None, store=None, llm=None))
    assert rec.calls[-1]["lane"] == "care"
    assert rec.calls[-1]["persona"] is None  # the assistant, not the specialist
    assert rec.calls[-1]["role_fragment"]  # a non-empty focus was applied


def test_explicit_human_request_speaks_as_the_specialist(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("please connect me to a human", embedder=None, store=None, llm=None))
    assert rec.calls[-1]["lane"] == "escalation"
    assert rec.calls[-1]["persona"] == "agent"


def test_ambiguous_turn_asks_instead_of_answering(monkeypatch, tmp_path):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)

    class Unclear:  # the 8B tie-break says it cannot tell the two intents apart
        def generate(self, *a, **k):
            class R:
                text = '{"lane": "unclear"}'
            return R()

    events = list(omni.stream_omni("it is not right", embedder=None, store=None, llm=None,
                                   small_llm=Unclear(), trace_path=str(tmp_path / "t.jsonl")))
    assert not rec.calls  # never retrieved or answered
    assert events[-1]["answer"].endswith("?")  # asked a question


def test_lane_roles_and_personas():
    assert lane_persona("escalation") == "agent"
    for lane in ("stylist", "care", "complaint", "answers"):
        assert lane_persona(lane) is None
        assert role_fragment(lane)  # service lanes carry a focus
    assert role_fragment("escalation") == ""  # escalation focus lives in the specialist prompt


def test_multitask_turn_fans_out_to_both_lanes_and_stitches(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    events = list(omni.stream_omni("suggest a gift for my mum and check where my order is",
                                   embedder=None, store=None, llm=None))
    lanes = [c["lane"] for c in rec.calls]
    assert "stylist" in lanes and "care" in lanes  # both parts handled
    assert len(rec.calls) == 2
    assert events[-1]["lane"] == "multi"


def test_single_shopping_turn_with_and_is_not_split(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("a red and blue jacket for running", embedder=None, store=None, llm=None))
    assert len(rec.calls) == 1  # one shopping turn, not two clauses


def test_multitask_puts_a_complaint_clause_first(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("suggest a warmer coat and my last order never arrived",
                          embedder=None, store=None, llm=None))
    assert rec.calls[0]["lane"] == "complaint"  # empathy leads regardless of typed order
