"""Continuous Training policy: CT fires on drift, new labeled data, or the schedule; it opens a
prompt experiment ONLY when a quality signal warrants one; and it PROPOSES a candidate that beats
the baseline with the gate and safety green. There is no auto-promote path. These lock in the
trigger, signal-to-action, and promotion rules, offline with no infrastructure (the core is pure).
"""
from mlops.ct import classify_signals, decide_promotion, evaluate_trigger, run_ct_cycle


def test_trigger_fires_on_each_signal_and_reports_why():
    fired, reasons = evaluate_trigger(drift_score=2.0, drift_threshold=1.0, new_labeled=0,
                                      min_new_labeled=10)
    assert fired and any("drift" in r for r in reasons)

    fired, reasons = evaluate_trigger(drift_score=0.0, drift_threshold=1.0, new_labeled=25,
                                      min_new_labeled=10)
    assert fired and any("verified" in r for r in reasons)

    fired, reasons = evaluate_trigger(drift_score=None, drift_threshold=1.0, new_labeled=0,
                                      min_new_labeled=10, scheduled=True)
    assert fired and any("scheduled" in r for r in reasons)


def test_trigger_stays_quiet_when_nothing_changed():
    fired, reasons = evaluate_trigger(drift_score=0.0, drift_threshold=1.0, new_labeled=3,
                                      min_new_labeled=10)
    assert not fired and reasons == []


def test_promotion_requires_beating_baseline_gate_and_safety():
    ok = dict(min_gain=0.01, gate_passed=True, safety_passed=True)
    assert decide_promotion(baseline_score=0.80, candidate_score=0.85, **ok)
    # gain below the margin -> no
    assert not decide_promotion(baseline_score=0.80, candidate_score=0.805, **ok)
    # a failed gate or safety check -> no, even with a big gain
    assert not decide_promotion(baseline_score=0.80, candidate_score=0.90,
                                min_gain=0.01, gate_passed=False, safety_passed=True)
    assert not decide_promotion(baseline_score=0.80, candidate_score=0.90,
                                min_gain=0.01, gate_passed=True, safety_passed=False)
    # an un-evaluated candidate (missing score) can never be promoted
    assert not decide_promotion(baseline_score=None, candidate_score=0.90, **ok)


def test_cycle_without_a_trigger_does_nothing():
    calls = []
    report = run_ct_cycle(trigger_fired=False, reasons=[],
                          train=lambda: calls.append("train") or {},
                          gate=lambda: calls.append("gate") or {"passed": True})
    assert not report.triggered and calls == []  # never retrains when no trigger fired


def test_cycle_proposes_by_default_and_does_not_self_deploy():
    report = run_ct_cycle(
        trigger_fired=True, reasons=["scheduled cadence"],
        train=lambda: {"baseline_score": 0.80, "candidate_score": 0.86, "safety_passed": True,
                       "candidate_path": "mlops/prompt_registry/tiebreak_system.candidate.json"},
        gate=lambda: {"passed": True, "score": 1.0}, min_gain=0.01)
    assert report.promote_recommended is True
    assert report.promoted is False  # human-gated: proposed, not shipped
    assert any("PROPOSED" in n for n in report.notes)


def test_cycle_never_self_promotes_even_on_a_clear_win():
    # there is no auto-promote path: a candidate that clears every bar is still only PROPOSED
    report = run_ct_cycle(
        trigger_fired=True, reasons=["forced"],
        train=lambda: {"baseline_score": 0.80, "candidate_score": 0.90, "safety_passed": True,
                       "candidate_path": "x"},
        gate=lambda: {"passed": True, "score": 1.0}, min_gain=0.01)
    assert report.promote_recommended is True and report.promoted is False


def test_cycle_stops_at_notify_when_the_signal_does_not_warrant_an_experiment():
    calls = []
    report = run_ct_cycle(
        trigger_fired=True, reasons=["drift: query_embedding"],
        train=lambda: calls.append("train") or {"candidate_score": 0.9},
        gate=lambda: calls.append("gate") or {"passed": True}, experiment_warranted=False)
    assert report.triggered and calls == []  # data/index drift: no experiment opened
    assert any("NOTIFY" in n for n in report.notes)


def test_classify_signals_opens_only_on_quality_or_new_data():
    quality = {"monitors": {"grounding": {"drift": True}, "query_embedding": {"drift": False}}}
    assert classify_signals(quality)["experiment_warranted"] is True

    data_only = {"monitors": {"query_embedding": {"drift": True},
                              "retrieval_score": {"drift": True}, "grounding": {"drift": False}}}
    out = classify_signals(data_only)
    assert out["experiment_warranted"] is False and "NOTIFY" in out["action"]
    assert out["data_signals"] == ["query_embedding", "retrieval_score"]

    # enough new verified answers warrants an experiment even with no drift
    assert classify_signals({}, new_labeled=20, min_new_labeled=10)["experiment_warranted"] is True


def test_cycle_keeps_baseline_when_candidate_regresses_the_gate():
    report = run_ct_cycle(
        trigger_fired=True, reasons=["drift"],
        train=lambda: {"baseline_score": 0.80, "candidate_score": 0.90, "safety_passed": True},
        gate=lambda: {"passed": False, "score": 0.6}, min_gain=0.01)
    assert not report.promote_recommended and not report.promoted
    assert any("no promotion" in n for n in report.notes)
