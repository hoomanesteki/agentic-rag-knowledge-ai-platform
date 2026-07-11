#!/usr/bin/env python3
"""Human-triggered shadow replay: replay the last N real questions through the CHALLENGER (whatever
prompt/config is active in this run) and compare to the champion's recorded results, so a human has
champion-vs-challenger evidence BEFORE promoting. Logs the deltas to an skein-shadow MLflow
experiment and writes evaluation/reports/shadow_delta.json.

Set the challenger via env, e.g. shadow-test the gpt-oss workhorse:
    GROQ_MODEL_LARGE=openai/gpt-oss-120b make shadow N=50

Nothing here promotes anything; it is the evidence step of the human-gated champion/challenger flow.
Needs keys + an ingest for real numbers; a no-key run is a no-op.
"""
from __future__ import annotations

import json
import os
import sys
import time

from adapters.config import get_settings
from adapters.factory import make_embedder, make_llm, make_reranker, make_store
from ingest.naming import collection_name
from mlops.shadow import compare, load_champion_questions
from pipeline.answer import answer_question

_TRACES = os.getenv("TRACE_PATH", "traces/requests.jsonl")
_OUT = "evaluation/reports/shadow_delta.json"


def _log_mlflow(report: dict) -> None:
    settings = get_settings()
    uri = settings.mlflow_url or "./mlruns"
    if not uri.startswith(("http://", "https://")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("skein-shadow")
        with mlflow.start_run():
            mlflow.log_metric("n", report["n"])
            mlflow.log_metric("grounding_delta", report["grounding"]["delta"] or 0.0)
            mlflow.log_metric("cost_delta", report["cost"]["delta"] or 0.0)
            mlflow.log_metric("route_flips", report["route_flips"])
    except Exception as exc:
        print("mlflow logging skipped ({}: {})".format(type(exc).__name__, str(exc)[:100]),
              file=sys.stderr)


def main() -> int:
    settings = get_settings()
    n = int(os.getenv("N", "50"))
    if settings.vector_provider in ("memory", "fake", "") or not settings.groq_api_key:
        print("shadow replay needs keys + an ingest (VECTOR_PROVIDER=qdrant); skipped",
              file=sys.stderr)
        return 0
    items = load_champion_questions(_TRACES, n)
    if not items:
        print("no recorded champion traffic to replay yet", file=sys.stderr)
        return 0

    embedder, llm, reranker = make_embedder(), make_llm(), make_reranker()
    store = make_store(collection=collection_name(settings.domain, settings.embed_model))
    for it in items:
        r = answer_question(it["question"], embedder=embedder, store=store, llm=llm,
                            reranker=reranker)
        it["challenger"] = {"grounding": round(r.grounding, 3), "lane": r.trace.get("lane"),
                            "cost": r.trace.get("cost"), "tier": r.tier}

    report = compare(items)
    report["challenger_model"] = os.getenv("GROQ_MODEL_LARGE", "llama-3.3-70b-versatile")
    report["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nreview the deltas above, then a human promotes (or not). Nothing was deployed.")
    _log_mlflow(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
