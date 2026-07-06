#!/usr/bin/env python3
"""Report drift between an earlier reference window of traffic and the current one, across the
four monitors, by language. Reads the traces and feedback the app already writes. Run: make drift
"""
from __future__ import annotations

import json
import os
import sys

from adapters.factory import make_embedder
from evaluation.monitoring import read_jsonl
from mlops.drift import drift_report
from pipeline.answer import DEFAULT_TRACE_PATH

_FEEDBACK_PATH = "traces/feedback.jsonl"


def _log_mlflow(report: dict) -> None:
    """Best-effort: log the drift run to MLflow so the data-drift pillar is auditable alongside the
    eval and CT runs, not just printed to a terminal. Never fails the drift job."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("skein-drift")
        with mlflow.start_run(run_name="drift"):
            mlflow.log_param("drifted", report.get("drifted"))
            for name, monitor in (report.get("monitors") or {}).items():
                for key in ("psi", "distance", "current", "reference"):
                    val = monitor.get(key)
                    if isinstance(val, (int, float)):
                        mlflow.log_metric("{}_{}".format(name, key), float(val))
    except Exception:  # noqa: BLE001 - observability must never break the pipeline
        pass


def main() -> int:
    traces = read_jsonl(DEFAULT_TRACE_PATH)
    if len(traces) < 40:
        print("not enough traffic to split a reference and current window ({} traces); ask more "
              "questions first".format(len(traces)), file=sys.stderr)
        return 0
    feedback = read_jsonl(_FEEDBACK_PATH)
    # split both traces and feedback at one timestamp cutoff, so the windows line up
    ordered = sorted(traces, key=lambda t: t.get("ts", 0.0))
    cutoff = ordered[len(ordered) // 2].get("ts", 0.0)
    ref = [t for t in ordered if t.get("ts", 0.0) < cutoff]
    cur = [t for t in ordered if t.get("ts", 0.0) >= cutoff]
    fb_ref = [f for f in feedback if f.get("ts", 0.0) < cutoff]
    fb_cur = [f for f in feedback if f.get("ts", 0.0) >= cutoff]
    report = drift_report(ref, cur, feedback_ref=fb_ref, feedback_cur=fb_cur,
                          embedder=make_embedder())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    _log_mlflow(report)
    return 1 if report["drifted"] else 0  # non-zero so a scheduled job can alert on drift


if __name__ == "__main__":
    sys.exit(main())
