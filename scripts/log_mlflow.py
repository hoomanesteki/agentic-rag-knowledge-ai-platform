#!/usr/bin/env python3
"""Log the request traces to MLflow so each request is a run with its metrics.

Run: make mlflow-log
Local default is a file store at ./mlruns (view with `mlflow ui`); set MLFLOW_TRACKING_URI to
point at the MLflow server from docker-compose instead. Re-running is safe (already-logged runs
are skipped).
"""
from __future__ import annotations

import sys

from adapters.config import get_settings
from evaluation.monitoring import read_jsonl
from mlops.mlflow_sink import log_traces
from pipeline.answer import DEFAULT_TRACE_PATH


def main() -> int:
    uri = get_settings().mlflow_url
    if not uri:
        uri = "./mlruns"
        print("note: MLFLOW_TRACKING_URI is unset, using the deprecated local file store at "
              "./mlruns. Run make up and set MLFLOW_TRACKING_URI=http://localhost:5000 for the "
              "proper backend.", file=sys.stderr)
    traces = read_jsonl(DEFAULT_TRACE_PATH)
    if not traces:
        print("no traces at {}; ask some questions first (make ask / the API)".format(
            DEFAULT_TRACE_PATH), file=sys.stderr)
        return 0
    result = log_traces(traces, tracking_uri=uri)
    print("logged {} run(s), skipped {} already-logged, to {}".format(
        result["logged"], result["skipped"], uri))
    return 0


if __name__ == "__main__":
    sys.exit(main())
