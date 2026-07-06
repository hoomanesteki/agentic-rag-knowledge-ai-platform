"""M7.2 quality and M7.5 health aggregation over the request traces and thumbs feedback."""
from evaluation.monitoring import (
    aggregate_business,
    aggregate_health,
    aggregate_quality,
    read_jsonl,
)


def test_read_jsonl_missing_and_bad_lines(tmp_path):
    assert read_jsonl(str(tmp_path / "nope.jsonl")) == []
    path = tmp_path / "t.jsonl"
    path.write_text('{"a": 1}\n\nnot json\n{"a": 2}\n')
    rows = read_jsonl(str(path))
    assert rows == [{"a": 1}, {"a": 2}]  # blank and unparseable lines skipped


def test_aggregate_quality_by_language():
    traces = [
        {"message_id": "m1", "lang": "en", "tier": "auto", "grounding": 0.8},
        {"message_id": "m2", "lang": "en", "tier": "escalate"},
        {"message_id": "m3", "lang": "fr", "tier": "auto", "grounding": 0.6},
        {"message_id": "m4", "lang": "fr", "tier": "abstain"},
    ]
    feedback = [{"message_id": "m1", "verdict": "up"}, {"message_id": "m3", "verdict": "down"}]
    q = aggregate_quality(traces, feedback)

    assert q["overall"]["total"] == 4
    assert q["overall"]["escalation_rate"] == 0.25
    assert q["overall"]["thumbs_up"] == 1 and q["overall"]["thumbs_down"] == 1

    en, fr = q["by_language"]["en"], q["by_language"]["fr"]
    assert en["thumbs_up"] == 1 and en["avg_grounding"] == 0.8
    assert fr["thumbs_down"] == 1 and fr["abstain_rate"] == 0.5
    assert fr["avg_grounding"] == 0.6


def test_aggregate_business_kpis_and_unit_economics():
    # 2 answered, 1 escalated, 1 abstained -> served 4; containment = resolved-without-a-human = 3/4
    traces = [
        {"tier": "auto", "cost": 0.002}, {"tier": "auto", "cost": 0.004},
        {"tier": "escalate", "cost": 0.0}, {"tier": "abstain", "cost": 0.001},
        {"tier": "error"},  # infra failure: excluded from the business rates
    ]
    feedback = [{"verdict": "up"}, {"verdict": "up"}, {"verdict": "down"}]
    b = aggregate_business(traces, feedback, turns_per_session=8)

    assert b["served_turns"] == 4  # the error turn is excluded
    assert b["answer_rate"] == 0.5  # 2 of 4 served turns got a real answer
    assert b["containment_rate"] == 0.75  # 3 of 4 resolved without a human
    assert b["escalation_rate"] == 0.25
    assert b["satisfaction"] == round(2 / 3, 3)  # 2 up of 3 thumbs
    # all four served turns carry a numeric cost (the escalate turn's is 0.0), so the mean is /4
    assert b["avg_cost_per_turn"] == round((0.002 + 0.004 + 0.0 + 0.001) / 4, 6)
    assert b["cost_per_session_est"] == round(b["avg_cost_per_turn"] * 8, 4)


def test_aggregate_business_empty_is_safe():
    b = aggregate_business([], [])
    assert b["served_turns"] == 0 and b["answer_rate"] == 0.0
    assert b["avg_cost_per_turn"] is None and b["cost_per_session_est"] is None
    assert b["satisfaction"] is None


def test_missing_lang_buckets_as_unknown():
    q = aggregate_quality([{"message_id": "x", "tier": "auto", "grounding": 0.5}], [])
    assert "unknown" in q["by_language"]
    assert q["by_language"]["unknown"]["total"] == 1


def test_rates_exclude_infra_failures():
    # a degraded turn must not flatter the escalation rate: 1 escalate over 2 served (not 3 total)
    traces = [
        {"message_id": "m1", "lang": "en", "tier": "auto", "grounding": 0.9},
        {"message_id": "m2", "lang": "en", "tier": "escalate"},
        {"message_id": "m3", "lang": "en", "tier": "degraded"},
    ]
    q = aggregate_quality(traces, [])
    assert q["overall"]["total"] == 3 and q["overall"]["served"] == 2
    assert q["overall"]["escalation_rate"] == 0.5


def test_feedback_without_a_trace_is_unmatched_not_a_phantom_language():
    q = aggregate_quality([], [{"message_id": "gone", "verdict": "up"}])
    assert q["overall"]["thumbs_up"] == 1
    assert q["overall"]["unmatched_feedback"] == 1
    assert q["by_language"] == {}  # no phantom bucket for a thumb with no trace


def test_aggregate_health_computes_ops_metrics():
    traces = [
        {"lang": "en", "tier": "auto", "latency_ms": 100, "cost": 0.001, "grounding": 0.8,
         "ts": 0.0},
        {"lang": "en", "tier": "auto", "latency_ms": 200, "cost": 0.003, "grounding": 0.6,
         "ts": 30.0},
        {"lang": "en", "tier": "error", "latency_ms": 50, "ts": 60.0},
    ]
    h = aggregate_health(traces)
    o = h["overall"]
    assert o["total"] == 3
    assert o["error_rate"] == round(1 / 3, 3)
    assert o["p95_latency_ms"] == 200
    assert o["avg_cost"] == 0.002               # only the two costed turns
    assert o["throughput_per_min"] == 3.0       # 3 requests over a 60s span
    assert "grounding_trend" in o


def test_health_empty_is_safe():
    h = aggregate_health([])
    assert h["overall"]["total"] == 0 and h["overall"]["p95_latency_ms"] is None
