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


class _Queue:
    def __init__(self):
        self.filed = []

    def enqueue(self, question, **kw):
        self.filed.append((question, kw))
        return "abcd1234efgh"


def test_escalation_files_a_case_brief_and_confirms(monkeypatch, tmp_path):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    rq = _Queue()
    events = list(omni.stream_omni(
        "please connect me to a human", embedder=None, store=None, llm=None, review_queue=rq,
        domain="apparel_ecommerce", auth_identity=("Aaron Esteki", "info@esteki.ca"),
        message_id="m1", trace_path=str(tmp_path / "t.jsonl")))
    assert not rec.calls  # escalation does not go through the normal answer path
    assert rq.filed and "Escalation:" in rq.filed[0][0]  # a case brief was filed
    final = events[-1]
    assert final["tier"] == "escalate" and final["escalation_id"] == "abcd1234efgh"
    assert "1." in final["answer"] and "2." in final["answer"]  # a numbered confirm list
    assert "info@esteki.ca" in final["answer"]  # the shopper's own email, echoed to confirm


def test_escalation_without_a_queue_still_confirms(tmp_path):
    events = list(omni.stream_omni("I want to talk to a representative", embedder=None, store=None,
                                   llm=None, trace_path=str(tmp_path / "t.jsonl")))
    assert events[-1]["tier"] == "escalate"  # degrades gracefully with no queue wired


def test_agent_mode_repeated_human_request_does_not_refile(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    rq = _Queue()
    list(omni.stream_omni("seriously get me a human now", persona="agent", embedder=None,
                          store=None, llm=None, review_queue=rq))
    assert not rq.filed  # already with the specialist: no duplicate case
    assert rec.calls and rec.calls[-1]["persona"] == "agent"


def test_comma_joined_multitask_fans_out(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("my jacket ripped, show me a replacement", embedder=None, store=None,
                          llm=None))
    lanes = [c["lane"] for c in rec.calls]
    assert "complaint" in lanes and "stylist" in lanes  # comma-joined intents both handled


def test_verification_turn_with_an_email_is_kept_whole(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("check my order and my email is jo@example.com", embedder=None,
                          store=None, llm=None))
    assert len(rec.calls) == 1  # not split, so the name-plus-email gate sees the whole turn


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


# --- regression: post-verification hardening (multi-task drops, escalation false promise) ---

def test_multitask_answers_clause_is_answered_not_dropped(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("where is my order, suggest a warm coat, and what is your return policy",
                          embedder=None, store=None, llm=None))
    lanes = [c["lane"] for c in rec.calls]
    assert "answers" in lanes  # the policy sub-question is stitched in, not filtered out


def test_multitask_floats_a_late_complaint_to_the_front(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(omni, "stream_answer", rec)
    list(omni.stream_omni("suggest a coat, check my order, and my last order never arrived",
                          embedder=None, store=None, llm=None))
    assert rec.calls[0]["lane"] == "complaint"  # empathy leads even from a late clause


def test_escalation_without_a_queue_makes_no_false_promise(tmp_path):
    events = list(omni.stream_omni("please connect me with a representative", embedder=None,
                                   store=None, llm=None, trace_path=str(tmp_path / "t.jsonl")))
    ans = events[-1]["answer"]
    assert "logged this" not in ans and "follow up by email" not in ans
