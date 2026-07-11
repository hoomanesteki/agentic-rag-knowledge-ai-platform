"""The NOTIFY step turns a CT outcome into one deduped, actionable GitHub issue. These pin the issue
body for both outcomes (a proposed candidate vs a data/index notify-only) and that posting is a safe
dry run without gh, so notify can never break a CT run."""
from mlops.notify import build_issue, post_issue


def test_issue_for_a_proposed_candidate_carries_the_next_human_commands():
    report = {
        "domain": "apparel_ecommerce", "candidate_path": "prompts/tiebreak.candidate.json",
        "candidate_score": 0.86, "baseline_score": 0.80, "gain": 0.06,
        "gate_passed": True, "safety_passed": True, "promote_recommended": True,
        "signals": {"drift_note": "grounding drift", "new_labeled": 20, "min_new_labeled": 10,
                    "classification": {"experiment_warranted": True,
                                       "quality_signals": ["grounding"], "data_signals": [],
                                       "action": "register a candidate prompt experiment"}},
    }
    issue = build_issue(report)
    assert issue["label"] == "ct-signal:apparel_ecommerce"
    assert "candidate proposed" in issue["title"]
    assert "make shadow" in issue["body"] and "make registry-promote" in issue["body"]
    assert "PROPOSED only, never auto-promoted" in issue["body"]


def test_issue_for_data_drift_stops_at_notify_and_says_do_not_experiment():
    report = {
        "domain": "default",
        "signals": {"drift_note": "query_embedding drift", "new_labeled": 0, "min_new_labeled": 10,
                    "classification": {"experiment_warranted": False, "quality_signals": [],
                                       "data_signals": ["query_embedding"],
                                       "action": "NOTIFY only: investigate index"}},
    }
    issue = build_issue(report)
    assert "notify only" in issue["title"]
    assert "no candidate opened" in issue["body"]
    assert "do NOT open a prompt experiment" in issue["body"]
    assert "make shadow" not in issue["body"]  # nothing to shadow: no candidate


def test_post_issue_is_a_safe_dry_run_without_gh():
    issue = {"title": "t", "label": "l", "body": "b"}
    result = post_issue(issue, dry_run=True)
    assert result["posted"] is False and result["dry_run"] is True
