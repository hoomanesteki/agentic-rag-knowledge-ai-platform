"""M8.1 MLflow sink: route the per-request traces the app already writes into MLflow runs.

Tracing exists since M1.3; this just points it at MLflow so each request shows up as a run with
its route/model as params and latency, tokens, cost, grounding, and confidence as metrics. Runs
are tagged with a stable trace_id so re-running the sink does not duplicate. Local runs use the
file store (opt-out env set here); a real deployment points MLFLOW_TRACKING_URI at the server.
"""
from __future__ import annotations

import hashlib
import json
import os

_PARAM_KEYS = ("route", "model", "tier", "lang", "reranked", "metric", "graph", "conflict",
               "conflict_resolved")
_METRIC_KEYS = ("latency_ms", "prompt_tokens", "completion_tokens", "cost", "grounding",
                "confidence", "agent_steps")


def _trace_id(trace: dict) -> str:
    if trace.get("message_id"):
        return str(trace["message_id"])
    if trace.get("ts") is not None:
        return "ts:{}".format(trace["ts"])
    # last resort so an id-less trace is still deduped instead of re-logged forever
    return "sha1:" + hashlib.sha1(
        json.dumps(trace, sort_keys=True, default=str).encode()).hexdigest()


def _logged_trace_ids(client, experiment_id: str) -> set:
    seen: set = set()
    token = None
    while True:
        page = client.search_runs([experiment_id], max_results=1000, page_token=token)
        seen.update(run.data.tags.get("trace_id") for run in page)
        token = page.token
        if not token:
            return seen


def log_traces(traces: list[dict], *, tracking_uri: str, experiment: str = "skein-lite",
               client=None) -> dict:
    """Log each trace as an MLflow run, skipping ones already logged (by trace_id). Returns
    {logged, skipped}."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # the local file store is fine here
    from mlflow.tracking import MlflowClient

    client = client or MlflowClient(tracking_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    experiment_id = found.experiment_id if found else client.create_experiment(experiment)

    seen = _logged_trace_ids(client, experiment_id)
    logged = skipped = 0
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        tid = _trace_id(trace)
        if tid in seen:
            skipped += 1
            continue
        start = int(trace["ts"] * 1000) if isinstance(trace.get("ts"), (int, float)) else None
        run = client.create_run(experiment_id, start_time=start,
                                run_name=trace.get("message_id"))
        run_id = run.info.run_id
        for key in _PARAM_KEYS:
            value = trace.get(key)
            if value is not None:
                client.log_param(run_id, key, str(value)[:250])
        for key in _METRIC_KEYS:
            value = trace.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                client.log_metric(run_id, key, float(value))
        # tag last: a crash before this leaves an untagged run that a re-run redoes, rather than a
        # half-logged run that is skipped forever
        client.set_tag(run_id, "trace_id", tid)
        client.set_terminated(run_id)
        seen.add(tid)
        logged += 1
    return {"logged": logged, "skipped": skipped}
