"""Continuous Training (CT): the scheduled, gated loop that keeps the assistant current as new
labeled data and drift arrive, without ever auto-shipping an unsafe change.

CI, CD, and CT are three different loops, kept distinct on purpose:

  CI  runs on every push: lint, tests, and the eval gate -> is the CODE correct?  (ci.yml)
  CD  ships once CI is green -> is the code DEPLOYED?
  CT  runs on a schedule, on measured DRIFT, or on enough new human-verified data: it RE-OPTIMIZES
      the prompt against the ground-truth eval, then GATES the candidate on held-out accuracy AND
      safety. CT only ever PROPOSES a promotion; a human approves the deploy. Nothing
      retrains-and-ships itself, which is the drift and reward-hacking guardrail (a candidate that
      games the metric never reaches production).

"Training" here is not gradient descent on model weights (the LLM is a hosted Groq model): it is the
data-and-prompt layer we DO own, retrained on a cadence. Three assets are retrainable and each is
versioned and gated like a model: the router/answer PROMPTS (the OPRO loop in mlops.prompt_opt,
which the CT cycle runs), the retrieval index (re-embed new reviews and descriptions, incrementally
via run_ingest.py --only), and the governed enrichment features (recompute consensus on new
reviews). The wired scheduled cycle re-optimizes the prompt and gates it; the index and feature
refresh are the flywheel and batch jobs on the same cadence, which CT reads drift and new-data
signals from. That split is deliberate: the prompt loop runs offline in CI, the re-index needs the
live embedder and store.

This module is the pure decision core (trigger policy + promotion policy + the report), so it is
unit-testable offline with no infrastructure. scripts/run_ct.py wires the real drift report, the
review-queue flywheel, the prompt-optimization loop, the eval gate, and the MLflow sink.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CTReport:
    """The outcome of one CT cycle: what fired it, what training produced, and whether a promotion
    is RECOMMENDED. CT never marks a promotion approved and never edits served code; a human always
    approves the deploy. Serialized to evaluation/reports/ct_report.json and logged to MLflow so
    every cycle is auditable. `promoted` is retained for schema stability and is always False."""
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


# Which drift monitors a prompt re-optimization can actually move (quality signals) vs the ones it
# cannot (a data/index or intent-mix problem). Auto-opening an experiment on the latter is the
# noise-experiment anti-pattern: drift is a trigger for INVESTIGATION, not automatic retraining.
_QUALITY_MONITORS = ("confidence", "grounding", "feedback_rate")
_DATA_MONITORS = ("query_embedding", "retrieval_score")


def classify_signals(drift_report: dict, *, new_labeled: int = 0,
                     min_new_labeled: int = 0) -> dict:
    """Decide what a signal WARRANTS: opening a prompt experiment, or stopping at NOTIFY. A prompt
    retrain can move confidence, grounding, and the feedback rate, so those (or enough new verified
    answers) warrant a candidate experiment for human review. Query-embedding and retrieval-score
    drift indicate an intent-mix or index/data issue a prompt cannot fix, so those STOP at notify
    with a proposed human action and never auto-open an experiment."""
    monitors = drift_report.get("monitors", {}) if drift_report else {}

    def drifted(name: str) -> bool:
        return bool((monitors.get(name) or {}).get("drift"))

    quality = [m for m in _QUALITY_MONITORS if drifted(m)]
    data = [m for m in _DATA_MONITORS if drifted(m)]
    enough_new = min_new_labeled > 0 and new_labeled >= min_new_labeled
    warranted = bool(quality) or enough_new
    if warranted:
        action = "register a candidate prompt experiment for human review"
    elif data:
        action = ("NOTIFY only: {} drift is an intent-mix or index/data issue a prompt cannot fix; "
                  "propose a human investigate retrieval/ingest, no experiment opened".format(
                      ", ".join(data)))
    else:
        action = "no action: no signal warrants a change"
    return {"experiment_warranted": warranted, "quality_signals": quality, "data_signals": data,
            "enough_new_labeled": enough_new, "action": action}


def run_ct_cycle(*, trigger_fired: bool, reasons, train, gate, min_gain: float = 0.01,
                 experiment_warranted: bool = True) -> CTReport:
    """Run one CT cycle over injected steps. `train()` retrains the prompt against the ground truth
    and returns {baseline_score, candidate_score, safety_passed, candidate_path, note}; `gate()`
    runs the regression gate and returns {passed, score}. CT then decides whether to RECOMMEND the
    promotion; a human always approves the deploy (there is no auto-promote path). CT only records
    the recommendation and never edits served code. When a trigger fired but the signal does not
    warrant an experiment (data/index drift a prompt cannot fix), CT stops at NOTIFY without opening
    one. Pure control flow, so the policy is tested without infra."""
    report = CTReport(triggered=bool(trigger_fired), reasons=list(reasons))
    if not trigger_fired:
        report.notes.append("no trigger fired; nothing to retrain this cycle")
        return report
    if not experiment_warranted:
        report.notes.append("signal did not warrant an experiment; stopping at NOTIFY. A prompt "
                            "re-optimization cannot fix data/index drift; proposed a human "
                            "investigate retrieval and ingest instead.")
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

    if report.promote_recommended:
        report.notes.append(
            "promotion PROPOSED (human-gated): review {} then promote. There is no auto-promote "
            "path; a person always approves the deploy.".format(
                report.candidate_path or "the candidate"))
    else:
        report.notes.append("no promotion: candidate did not clear the bar; baseline kept")
    return report
