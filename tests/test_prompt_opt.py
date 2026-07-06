"""The prompt-optimization loop is control logic, tested here offline with fake evaluate/propose
functions so it needs no model. It must keep the best scoring candidate, keep the baseline when
nothing beats it, respect the safety gate even when the unsafe candidate scores higher, and never
promote on its own (it only returns a best_prompt for a human to review)."""
import json

from mlops.prompt_opt import optimize_prompt, save_candidate


def _evaluate_from(scores):
    def ev(prompt):
        return scores.get(prompt, 0.0), []  # (score, misrouted)
    return ev


def test_keeps_the_best_candidate_and_reports_the_delta():
    scores = {"base": 0.50, "better": 0.70, "worse": 0.40}
    rounds = iter([["worse", "better"], ["worse"]])
    r = optimize_prompt("base", evaluate=_evaluate_from(scores),
                        propose=lambda p, m, n: next(rounds), rounds=2, n_candidates=2)
    assert r["improved"] and r["best_prompt"] == "better"
    assert r["baseline_score"] == 0.5 and r["best_train_score"] == 0.7


def test_keeps_the_baseline_when_nothing_beats_it():
    scores = {"base": 0.80, "c1": 0.70, "c2": 0.79}
    r = optimize_prompt("base", evaluate=_evaluate_from(scores),
                        propose=lambda p, m, n: ["c1", "c2"], rounds=1, n_candidates=2)
    assert not r["improved"] and r["best_prompt"] == "base"


def test_safety_gate_rejects_an_unsafe_candidate_even_if_it_scores_higher():
    scores = {"base": 0.50, "unsafe": 0.90, "safe": 0.60}
    r = optimize_prompt("base", evaluate=_evaluate_from(scores),
                        propose=lambda p, m, n: ["unsafe", "safe"],
                        safety=lambda p: p != "unsafe", rounds=1, n_candidates=2)
    assert r["best_prompt"] == "safe"  # the higher-scoring unsafe candidate was gated out
    assert any(h.get("rejected") == "safety" for h in r["history"])


def test_save_candidate_writes_a_proposed_artifact(tmp_path):
    path = save_candidate("t", "the prompt", {"baseline_test": 0.5}, root=str(tmp_path))
    d = json.load(open(path))
    assert d["status"] == "proposed" and d["prompt"] == "the prompt" and d["baseline_test"] == 0.5
