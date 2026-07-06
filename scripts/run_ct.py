#!/usr/bin/env python3
"""Run one Continuous Training (CT) cycle: check the triggers, retrain and gate, then PROPOSE a
promotion (human-gated by default). Writes an auditable CT report and logs the cycle to MLflow.

CT is scheduled or triggered, NOT per-push (that is CI). See mlops/ct.py for the policy and
docs/model-lifecycle-and-operations.md / showcase/mlops.qmd for how CI, CD, and CT fit together.

  make ct                        # a real cycle: drift + new-data triggers, retrain, gate, propose
  make ct CT_ARGS="--scheduled"  # force the scheduled cadence (what the cron workflow runs)
  make ct CT_ARGS="--skip-train" # fast gate-only health check (no Groq key needed)

Resilient by design: with no Groq key the prompt-optimization training step is skipped and CT runs
the eval gate as a health check, so the cycle still produces a report instead of failing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from adapters.config import get_settings
from evaluation.ci_gate import load_gate, run_gate
from mlops.ct import evaluate_trigger, run_ct_cycle
from rag.hitl import ReviewQueue

_TRACES = os.getenv("TRACE_PATH", "traces/requests.jsonl")
_PROMPT_OPT_REPORT = "evaluation/reports/prompt_opt.json"
_CT_REPORT = "evaluation/reports/ct_report.json"


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _drift_signal() -> tuple[float | None, str]:
    """Number of drift monitors flagged between an earlier and the current traffic window, or None
    when there is not enough traffic to split two windows. Uses a bounded recency window (the last
    800 traces) so an old one-time regime shift does not re-trigger CT forever, and honors the
    authoritative `drifted` flag, which folds in per-language drift the overall monitors miss."""
    from mlops.drift import drift_report
    traces = sorted(_read_jsonl(_TRACES), key=lambda t: t.get("ts", 0.0))[-800:]
    if len(traces) < 40:
        return None, "not enough traffic to measure drift ({} traces)".format(len(traces))
    cut = len(traces) // 2
    rep = drift_report(traces[:cut], traces[cut:])
    flagged = sum(1 for m in rep["monitors"].values() if m.get("drift"))
    flagged += sum(1 for lang in rep["by_language"].values()
                   for m in lang.values() if m.get("drift"))
    score = float(max(flagged, 1 if rep.get("drifted") else 0))  # drifted always triggers
    return score, "{} drift monitor(s) flagged (drifted={})".format(flagged, rep.get("drifted"))


def _new_labeled(domain: str) -> int:
    """Human-verified answers closed since the last CT watermark (the flywheel's new data)."""
    try:
        rq = ReviewQueue(get_settings().review_queue_db)
        return len(rq.closed_since(rq.flywheel_watermark(domain), domain=domain))
    except Exception:
        return 0


def _train() -> dict:
    """The training step: run the safety-gated prompt-optimization loop and read its report. If it
    cannot run (no Groq key), skip it so CT falls back to a gate-only health check."""
    try:
        proc = subprocess.run([sys.executable, "scripts/run_prompt_opt.py"],
                              capture_output=True, text=True, env=dict(os.environ), timeout=1800)
    except Exception as exc:  # noqa: BLE001 - any launch failure degrades to a health check
        return {"note": "training skipped ({}: {})".format(type(exc).__name__, exc)}
    if proc.returncode != 0 or not os.path.exists(_PROMPT_OPT_REPORT):
        tail = (proc.stderr or "").strip()[-300:]
        return {"note": "training skipped: prompt-opt produced no report (missing Groq key?)"
                + (" | stderr: " + tail if tail else "")}
    with open(_PROMPT_OPT_REPORT, encoding="utf-8") as f:
        rep = json.load(f)
    # Read the candidate path the loop ACTUALLY wrote (or ""), not a file that may be stale from a
    # prior run. A promotable score is exposed only when a candidate exists, so noise on the small
    # held-out split can never recommend a no-op promotion; safety comes from the loop's own gate.
    candidate_path = rep.get("candidate_path", "")
    return {
        "baseline_score": rep.get("baseline_test"),
        "candidate_score": (rep.get("candidate_test") if candidate_path
                            else rep.get("baseline_test")),
        "safety_passed": rep.get("safety_passed"),
        "candidate_path": candidate_path,
        "note": "retrained the tie-break prompt; held-out {} vs baseline {}{}".format(
            rep.get("candidate_test"), rep.get("baseline_test"),
            "" if candidate_path else " (no promotable candidate written)"),
    }


def _gate() -> dict:
    """The regression gate: the same offline eval gate CI runs, as CT's promotion gate."""
    trace_path = os.path.join(tempfile.mkdtemp(), "gate.jsonl")
    res = run_gate(load_gate(os.getenv("GATE_FIXTURES", "evaluation/fixtures/gate.json")),
                   min_score=float(os.getenv("GATE_MIN_SCORE", "1.0")), trace_path=trace_path)
    return {"passed": res["passed"], "score": res["score"]}


def _log_mlflow(d: dict) -> None:
    """Best-effort: log the cycle as an MLflow run so promotions are auditable. Never fails CT."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("skein-ct")
        with mlflow.start_run(run_name="ct-cycle"):
            mlflow.log_params({"triggered": d["triggered"], "domain": d.get("domain"),
                               "promote_recommended": d["promote_recommended"],
                               "promoted": d["promoted"]})
            for k in ("baseline_score", "candidate_score", "gain"):
                if d.get(k) is not None:
                    mlflow.log_metric(k, float(d[k]))
    except Exception:  # noqa: BLE001 - observability must never break the pipeline
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduled", action="store_true", help="run the scheduled cadence (cron)")
    ap.add_argument("--force", action="store_true", help="force a cycle regardless of triggers")
    ap.add_argument("--min-new-labeled", type=int,
                    default=int(os.getenv("CT_MIN_NEW_LABELED", "10")))
    ap.add_argument("--drift-threshold", type=float,
                    default=float(os.getenv("CT_DRIFT_THRESHOLD", "1")))
    ap.add_argument("--min-gain", type=float, default=float(os.getenv("CT_MIN_GAIN", "0.01")))
    ap.add_argument("--auto-promote", action="store_true",
                    help="apply a recommended promotion (default OFF: propose, a human approves)")
    ap.add_argument("--skip-train", action="store_true",
                    help="gate-only health check, no retraining (fast, no Groq key needed)")
    ap.add_argument("--out", default=_CT_REPORT)
    args = ap.parse_args()

    domain = get_settings().domain
    drift_score, drift_note = _drift_signal()
    new_labeled = _new_labeled(domain)
    fired, reasons = evaluate_trigger(
        drift_score=drift_score, drift_threshold=args.drift_threshold,
        new_labeled=new_labeled, min_new_labeled=args.min_new_labeled,
        scheduled=args.scheduled, forced=args.force)

    train = (lambda: {"note": "training skipped (--skip-train)"}) if args.skip_train else _train
    report = run_ct_cycle(trigger_fired=fired, reasons=reasons, train=train, gate=_gate,
                          min_gain=args.min_gain, auto_promote=args.auto_promote)

    d = report.to_dict()
    d["domain"] = domain
    d["signals"] = {"drift_score": drift_score, "drift_note": drift_note,
                    "new_labeled": new_labeled, "min_new_labeled": args.min_new_labeled}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    _log_mlflow(d)

    print("CT cycle: {}".format("TRIGGERED" if report.triggered else "no trigger this cycle"))
    print("  signals: drift={} ({}), new_verified={} (min {})".format(
        drift_score, drift_note, new_labeled, args.min_new_labeled))
    for r in report.reasons:
        print("  trigger:", r)
    if report.triggered:
        print("  baseline {} -> candidate {} | gate_passed={} | safety={}".format(
            report.baseline_score, report.candidate_score, report.gate_passed,
            report.safety_passed))
        print("  promote_recommended={} | promoted={}".format(
            report.promote_recommended, report.promoted))
    for n in report.notes:
        print("  note:", n)
    print("wrote", args.out)
    # go red when the regression gate actually failed, so a real regression surfaces in the workflow
    # status instead of hiding in a green run's artifact; a quiet or skipped cycle is fine.
    return 1 if (report.triggered and report.gate_passed is False) else 0


if __name__ == "__main__":
    sys.exit(main())
