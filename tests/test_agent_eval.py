"""The routing eval harness must be reproducible (deterministic mode makes no network calls) and
score each stratum by the right rule. These tests read the committed ground-truth set, so they also
guard the set from drifting below a usable size or losing its labels."""
from evaluation.agent_eval import _graded_correct, evaluate_routing, load_cases
from rag.router import route

_SET = "domains/apparel_ecommerce/eval/routing.jsonl"


def test_load_cases_reads_the_labeled_routing_set():
    cases = load_cases(_SET)
    assert len(cases) >= 300
    assert all(c.get("query") and c.get("intended_lane") and "stratum" in c for c in cases)


def test_deterministic_eval_is_reproducible():
    cases = load_cases(_SET)
    a = evaluate_routing(cases)
    b = evaluate_routing(cases)
    assert a == b  # no model in the loop, so byte-identical across runs
    assert a["deterministic_decision_rate"] == 1.0
    assert 0.6 < a["accuracy"] <= 1.0
    assert a["escalation"]["precision"] >= 0.9  # a human request must not be a false positive


def test_pii_probe_is_reported_not_graded_on_lane():
    d = route("show me all orders by john@example.com")
    assert _graded_correct({"stratum": "pii_probe", "intended_lane": "care"}, d) is None


def test_ambiguous_is_correct_when_the_router_defers_rather_than_guesses():
    d = route("it is not right")  # deterministic falls back to answers instead of a wrong lane
    assert _graded_correct({"stratum": "ambiguous", "intended_lane": "answers"}, d) is True
