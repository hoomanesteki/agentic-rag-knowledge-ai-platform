"""The loop contract. Every retry or fan-out in the deterministic orchestrator must carry (a) a hard
step cap, (b) a strictly shrinking budget or a growing evidence set, and (c) a clean finish or an
escalation on exhaustion, so a turn can never loop or run away. The retired LangGraph agent loop
enforced this with a novelty stop; omni + TurnBudget enforce it structurally, and these pin it."""
import rag.omni as omni
from adapters.budget import BudgetExceeded, TurnBudget
from adapters.fakes import EchoLLM


def test_fan_out_has_a_hard_step_cap():
    assert 1 <= omni._MAX_CLAUSES <= 4  # a run-on multi-intent turn cannot spawn unbounded clauses


def test_multitask_work_is_bounded_by_the_step_cap(monkeypatch):
    calls = []

    def rec(query, **kwargs):
        calls.append(kwargs.get("lane"))
        yield {"type": "final", "answer": kwargs.get("lane"), "lane": kwargs.get("lane"),
               "tier": "auto", "grounding": 0.9, "confidence": 0.9, "citations": []}

    monkeypatch.setattr(omni, "stream_answer", rec)
    # a deliberately run-on multi-intent turn: the pipeline is called at most cap + one reroute
    list(omni.stream_omni(
        "suggest a gift, check my order, what is your return policy, recommend a coat, "
        "and track my package",
        embedder=None, store=None, llm=EchoLLM(), auth_identity=("A", "a@b.com")))
    assert 1 <= len(calls) <= omni._MAX_CLAUSES + 1


def test_shrinking_budget_halts_the_next_call():
    b = TurnBudget(max_calls=1)
    b.charge(tokens=10)  # one call spent
    try:
        b.check()
        halted = False
    except BudgetExceeded:
        halted = True
    assert halted  # (b): the shrinking budget stops the next call rather than looping


def test_budget_breach_finishes_the_turn_instead_of_hanging():
    # (c): a breach before any final still yields a clean, non-empty final (a handoff), not nothing
    b = TurnBudget(max_calls=0)  # already exhausted

    def raising(query, **kwargs):
        b.check()  # raises immediately
        yield {"type": "final"}

    events = list(omni._emit_with_budget(raising("x"), b))
    final = [e for e in events if e.get("type") == "final"][-1]
    assert final["answer"] and final["tier"] == "abstain"
