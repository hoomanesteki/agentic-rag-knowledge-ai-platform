"""M7.2 quality aggregation over the request traces and thumbs feedback."""
from evaluation.monitoring import aggregate_quality, read_jsonl


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
