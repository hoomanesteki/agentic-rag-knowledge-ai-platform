"""The per-turn budget must ENFORCE its ceilings, not just report them: a call is checked before it
runs, and a breach stops the turn with the answers already in hand rather than looping. These pin
the four ceilings, the LLM wrapper, and the omni integration (single-task snapshot, multi-task
early stop)."""
import pytest

import rag.omni as omni
from adapters.budget import BudgetedLLM, BudgetExceeded, TurnBudget
from adapters.fakes import EchoLLM


def test_check_trips_on_each_ceiling():
    b = TurnBudget(max_calls=2)
    b.check()  # fresh: fine
    b.charge(tokens=10)
    b.charge(tokens=10)
    with pytest.raises(BudgetExceeded) as e:
        b.check()  # 2 calls already spent
    assert e.value.reason == "max_calls"

    b = TurnBudget(max_tokens=100)
    b.charge(tokens=100)
    with pytest.raises(BudgetExceeded) as e:
        b.check()
    assert e.value.reason == "max_tokens"

    b = TurnBudget(max_usd=0.01)
    b.charge(usd=0.01)
    with pytest.raises(BudgetExceeded) as e:
        b.check()
    assert e.value.reason == "max_usd"


def test_deadline_trips_on_wall_clock():
    now = [1000.0]
    b = TurnBudget(max_seconds=5.0, clock=lambda: now[0])
    b.check()  # elapsed 0
    now[0] = 1006.0  # 6s later, past the 5s deadline
    with pytest.raises(BudgetExceeded) as e:
        b.check()
    assert e.value.reason == "deadline"


def test_snapshot_reports_spend_and_limits():
    b = TurnBudget(max_calls=8)
    b.charge(tokens=42, usd=0.001)
    snap = b.snapshot()
    assert snap["calls"] == 1 and snap["tokens"] == 42
    assert snap["limits"]["calls"] == 8


def test_budgeted_llm_charges_and_then_blocks():
    b = TurnBudget(max_calls=1)
    wrapped = BudgetedLLM(EchoLLM(), b)
    assert wrapped.model is not None or wrapped.model is None  # model surface is exposed
    out = "".join(wrapped.stream("two words here"))
    assert out  # first call runs
    assert b.calls == 1 and b.tokens > 0  # and is charged
    with pytest.raises(BudgetExceeded):
        # a second call is checked BEFORE it runs and refused
        list(wrapped.stream("again please"))


def test_budgeted_llm_fills_usage_out_like_the_raw_client():
    b = TurnBudget()
    usage: dict = {}
    "".join(BudgetedLLM(EchoLLM(), b).stream("two words here", usage_out=usage))
    assert usage["prompt_tokens"] == 3 and usage["completion_tokens"] == 1


def test_single_task_final_carries_a_budget_snapshot(monkeypatch):
    def fake_stream(query, **kwargs):
        yield {"type": "final", "answer": "ok", "lane": kwargs.get("lane"),
               "grounding": 0.9, "confidence": 0.9, "citations": []}

    monkeypatch.setattr(omni, "stream_answer", fake_stream)
    events = list(omni.stream_omni("does the flow legging run small", embedder=None, store=None,
                                   llm=EchoLLM()))
    final = events[-1]
    assert "budget" in final and "limits" in final["budget"]


def test_multitask_stops_early_when_the_budget_is_spent(monkeypatch):
    # each clause "spends" one model call; a max_calls=1 budget must answer the first clause and
    # then stop instead of fanning out to the second, and still return a final
    def fake_stream(query, **kwargs):
        llm = kwargs.get("llm")
        if llm is not None:
            llm.generate("probe")  # consume one budget call, as a real generation would
        yield {"type": "final", "answer": kwargs.get("lane"), "lane": kwargs.get("lane"),
               "grounding": 0.9, "confidence": 0.9, "citations": []}

    monkeypatch.setattr(omni, "stream_answer", fake_stream)
    events = list(omni.stream_omni(
        "suggest a gift for my mum and check where my order is", embedder=None, store=None,
        llm=EchoLLM(), auth_identity=("Aaron", "a@b.com"), budget=TurnBudget(max_calls=1)))
    final = events[-1]
    assert final["lane"] == "multi"
    assert final["budget"]["calls"] == 1  # only the first clause ran
    # exactly one lane made it into the stitched reply, not both
    assert ("stylist" in final["answer"]) ^ ("care" in final["answer"])
