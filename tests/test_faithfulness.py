"""The online faithfulness ladder samples answered turns and scores them with a different-family
judge, never blocking the served turn. These pin the sampling policy (deterministic, low-grounding
always, small random fraction), the never-block enqueue (off by default), and the offline drain."""
import json

from adapters.base import LLMResult
from mlops.faithfulness import (
    _sampled,
    check_reason,
    drain_queue,
    enqueue_candidate,
    score_candidate,
)


def test_sampling_is_deterministic_and_rate_bounded():
    mid = "abc123"
    assert _sampled(mid, 1.0) is True       # rate 1 -> always
    assert _sampled(mid, 0.0) is False       # rate 0 -> never
    # stable: the same id samples the same way every time (no RNG)
    assert _sampled(mid, 0.5) == _sampled(mid, 0.5)
    # over many ids the fraction sampled is near the rate
    hit = sum(_sampled("m{}".format(i), 0.1) for i in range(2000))
    assert 120 < hit < 280  # ~10% of 2000, with slack


def test_low_grounding_is_always_checked_then_sampled():
    assert check_reason(0.4, "x", low_grounding=0.7, sample_rate=0.0) == "low_grounding"
    # healthy grounding but selected by the random sample
    assert check_reason(1.0, "x", low_grounding=0.7, sample_rate=1.0) == "sampled"
    # healthy and not sampled -> skip
    assert check_reason(1.0, "x", low_grounding=0.7, sample_rate=0.0) is None


def test_enqueue_is_a_no_op_unless_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("FAITHFULNESS_SAMPLING", raising=False)
    q = tmp_path / "q.jsonl"
    assert enqueue_candidate("m1", "q", "a", [{"text": "c"}], 0.1, queue_path=str(q)) is None
    assert not q.exists()  # default off: nothing written, nothing blocked


def test_enqueue_writes_a_candidate_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FAITHFULNESS_SAMPLING", "on")
    q = tmp_path / "q.jsonl"
    reason = enqueue_candidate("m1", "how much is it", "it is 98 [1]",
                               [{"text": "the legging costs 98"}], 0.2, queue_path=str(q))
    assert reason == "low_grounding"  # 0.2 < 0.7
    rec = json.loads(q.read_text().strip())
    assert rec["contexts"] == ["the legging costs 98"] and rec["message_id"] == "m1"


class _FakeJudge:
    """Says every atomic statement is supported, so faithfulness resolves to 1.0."""

    model = "fake-judge"

    def generate(self, prompt, *, system=None, max_tokens=512):
        text = "[true]" if "supported" in (system or "").lower() else '["the item costs 98"]'
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake-judge")


def test_score_and_drain_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("FAITHFULNESS_SAMPLING", "on")
    q = tmp_path / "q.jsonl"
    s = tmp_path / "scores.jsonl"
    enqueue_candidate("m1", "how much", "the item costs 98 [1]",
                      [{"text": "the item costs 98 dollars"}], 0.3, queue_path=str(q))
    assert score_candidate(_FakeJudge(), {"answer": "the item costs 98",
                                          "contexts": ["the item costs 98 dollars"]}) == 1.0
    summary = drain_queue(_FakeJudge(), queue_path=str(q), scores_path=str(s))
    assert summary["scored"] == 1 and summary["mean"] == 1.0
    assert q.read_text() == ""  # queue drained
    assert json.loads(s.read_text().strip())["faithfulness"] == 1.0
