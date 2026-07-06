"""Continuous Training (CT): the scheduled, gated loop that keeps the assistant current as new
labeled data and drift arrive, without ever auto-shipping an unsafe change.

CI, CD, and CT are three different loops, kept distinct on purpose:

  CI  runs on every push: lint, tests, and the eval gate -> is the CODE correct?  (ci.yml)
  CD  ships once CI is green -> is the code DEPLOYED?
  CT  runs on a schedule, on measured DRIFT, or on enough new human-verified data: it re-indexes the
      new data, recomputes governed features, and RE-OPTIMIZES the prompts against the ground-truth
      eval, then GATES the candidate on held-out accuracy AND safety. CT only ever PROPOSES a
      promotion; a human approves the deploy. Nothing retrains-and-ships itself, which is the drift
      and reward-hacking guardrail (a candidate that games the metric never reaches production).

"Training" here is not gradient descent on model weights (the LLM is a hosted Groq model): it is the
data-and-prompt layer we DO own, retrained on a cadence. The retrainable assets are the retrieval
index (re-embed new reviews and descriptions), the governed enrichment features (recompute consensus
on new reviews), and the router/answer PROMPTS (the OPRO loop in mlops.prompt_opt). Each is
versioned and gated the same way a model would be.

This module is the pure decision core (trigger policy + promotion policy + the report), so it is
unit-testable offline with no infrastructure. scripts/run_ct.py wires the real drift report, the
review-queue flywheel, the prompt-optimization loop, the eval gate, and the MLflow sink.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CTReport:
    """The outcome of one CT cycle: what fired it, what training produced, and whether a promotion
    is recommended (and, only in auto mode, applied). Serialized to
    evaluation/reports/ct_report.json and logged to MLflow so every cycle is auditable."""
    triggered: bool
    reasons: list = field(default_factory=list)
    baseline_score: float | None = None
    candidate_score: float | None = None
    gate_passed: bool | None = None
    safety_passed: bool | None = None
    promote_recommended: bool = False
    promoted: bool = False
    candidate_path: str = ""
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "reasons": self.reasons,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "gain": (round(self.candidate_score - self.baseline_score, 4)
                     if self.baseline_score is not None and self.candidate_score is not None
                     else None),
            "gate_passed": self.gate_passed,
            "safety_passed": self.safety_passed,
            "promote_recommended": self.promote_recommended,
            "promoted": self.promoted,
            "candidate_path": self.candidate_path,
            "steps": self.steps,
            "notes": self.notes,
        }


def evaluate_trigger(*, drift_score: float | None, drift_threshold: float, new_labeled: int,
                     min_new_labeled: int, scheduled: bool = False,
                     forced: bool = False) -> tuple[bool, list[str]]:
    """Decide whether a CT cycle should run, and say WHY. CT fires on any of: an operator forcing
    it, the scheduled cadence, measured drift over threshold, or enough new human-verified answers
    to be worth retraining on. Returning the reasons makes the run self-documenting."""
    reasons: list[str] = []
    if forced:
        reasons.append("forced by operator")
    if scheduled:
        reasons.append("scheduled cadence")
    if drift_score is not None and drift_score >= drift_threshold:
        reasons.append("drift {:.3f} >= threshold {:.3f}".format(drift_score, drift_threshold))
    if min_new_labeled > 0 and new_labeled >= min_new_labeled:
        reasons.append("{} new verified answers (>= {})".format(new_labeled, min_new_labeled))
    return bool(reasons), reasons


def decide_promotion(*, baseline_score: float | None, candidate_score: float | None,
                     min_gain: float, gate_passed: bool | None,
                     safety_passed: bool | None) -> bool:
    """Promote ONLY when the candidate clears every bar: it beats the baseline by at least min_gain
    on the held-out eval, the regression gate is green, and the safety check passed. A missing score
    or a failed gate/safety check is an automatic no, so an un-evaluated or unsafe candidate can
    never be recommended."""
    if baseline_score is None or candidate_score is None:
        return False
    if not gate_passed or not safety_passed:
        return False
    return (candidate_score - baseline_score) >= min_gain


def run_ct_cycle(*, trigger_fired: bool, reasons, train, gate, min_gain: float = 0.01,
                 auto_promote: bool = False) -> CTReport:
    """Run one CT cycle over injected steps. `train()` retrains the prompt against the ground truth
    and returns {baseline_score, candidate_score, safety_passed, candidate_path, note}; `gate()`
    runs the regression gate and returns {passed, score}. CT then decides whether to recommend the
    promotion, and applies it only when auto_promote is on (default OFF: CT proposes, a human
    approves). Pure control flow, so the policy is tested without any infrastructure."""
    report = CTReport(triggered=bool(trigger_fired), reasons=list(reasons))
    if not trigger_fired:
        report.notes.append("no trigger fired; nothing to retrain this cycle")
        return report

    t = train() or {}
    report.steps.append("train")
    report.baseline_score = t.get("baseline_score")
    report.candidate_score = t.get("candidate_score")
    report.safety_passed = t.get("safety_passed")
    report.candidate_path = t.get("candidate_path", "")
    if t.get("note"):
        report.notes.append(t["note"])

    g = gate() or {}
    report.steps.append("gate")
    report.gate_passed = g.get("passed")

    report.promote_recommended = decide_promotion(
        baseline_score=report.baseline_score, candidate_score=report.candidate_score,
        min_gain=min_gain, gate_passed=report.gate_passed, safety_passed=report.safety_passed)

    if report.promote_recommended and auto_promote:
        report.promoted = True
        report.notes.append("auto-promoted: candidate beat baseline with gate and safety green")
    elif report.promote_recommended:
        report.notes.append(
            "promotion PROPOSED (human-gated): review {} then promote".format(
                report.candidate_path or "the candidate"))
    else:
        report.notes.append("no promotion: candidate did not clear the bar; baseline kept")
    return report
