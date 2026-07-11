#!/usr/bin/env python3
"""Drain the online faithfulness queue: score each sampled live answer with a DIFFERENT-family judge
(gpt-oss-120b by default, so it never grades the llama generator's own style up), append the results
to traces/faithfulness.jsonl, and log the aggregate to MLflow.

This is the offline half of the faithfulness ladder. The request path only appends candidates to the
queue after the answer is sent, so nothing here ever blocks a served turn. Run: make faithfulness
(needs a Groq key; a no-key run is a no-op).
"""
from __future__ import annotations

import os
import sys

from adapters.config import get_settings
from mlops.faithfulness import drain_queue


def _judge():
    settings = get_settings()
    if not settings.groq_api_key:
        return None
    from adapters.groq import GroqClient
    model = os.getenv("FAITHFULNESS_JUDGE_MODEL", "openai/gpt-oss-120b")
    return GroqClient(model=model)


def _log_mlflow(summary: dict) -> None:
    settings = get_settings()
    tracking_uri = settings.mlflow_url or "./mlruns"
    if not tracking_uri.startswith(("http://", "https://")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("skein-faithfulness")
        with mlflow.start_run():
            mlflow.log_metric("scored", summary["scored"])
            mlflow.log_metric("flagged", summary["flagged"])
            if summary["mean"] is not None:
                mlflow.log_metric("faithfulness_mean", summary["mean"])
    except Exception as exc:  # never let an MLflow hiccup hide the result
        print("mlflow logging skipped ({}: {})".format(type(exc).__name__, str(exc)[:100]),
              file=sys.stderr)


def main() -> int:
    judge = _judge()
    if judge is None:
        print("no Groq key; faithfulness scoring skipped (queue left intact)", file=sys.stderr)
        return 0
    summary = drain_queue(judge)
    print("faithfulness: scored {scored}, flagged {flagged} (< 0.8), mean {mean}".format(**summary))
    _log_mlflow(summary)
    # a low mean faithfulness is a hallucination signal a scheduled job can alert on
    return 1 if (summary["mean"] is not None and summary["mean"] < 0.8) else 0


if __name__ == "__main__":
    sys.exit(main())
