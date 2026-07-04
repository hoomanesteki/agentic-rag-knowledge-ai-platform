#!/usr/bin/env python3
"""Promote the current RAG configuration through MLflow stages, gated by an eval score.

A "model" here is the serving configuration (LLM, embeddings, reranker, prompt/brain). Promotion is
not automatic: the config is logged to MLflow and transitioned to Production only if it clears the
production bar, to Staging if it clears the staging bar, otherwise left as a rejected candidate.
This is the dev -> staging -> prod gate flow.

What the gate actually scores (be precise): `make promote` runs the offline **CI eval gate**
(`evaluation/ci_gate.py`) over recorded fixtures with deterministic fakes, so it proves the
retrieval/grounding/abstain **pipeline** is wired and behaving, not the live model's answer quality.
For a model-quality promotion, run `make ragas` first (RAGAS on the golden set with the real
providers) and gate on that score; this script is the offline, CI-runnable half of that gate. The
promoted config's real providers are still recorded as MLflow params for provenance.

Offline-safe: the gate needs no keys, and MLflow logs to ./mlruns by default. The full model
registry needs the MLflow server (docker compose, MLFLOW_TRACKING_URI); without it the stage is
recorded as a run tag and printed, so the gate decision still holds.

Run: make promote   (or: uv run python scripts/promote_model.py)
Exit code is 0 when promoted (Staging or Production), 1 when rejected, so CI can block on it.
"""
from __future__ import annotations

import os
import sys
import tempfile

from adapters.config import get_settings
from evaluation.ci_gate import load_gate, run_gate

MODEL_NAME = os.getenv("PROMOTE_MODEL_NAME", "skein-rag")
STAGING_MIN = float(os.getenv("PROMOTE_STAGING_MIN", "0.8"))
PROD_MIN = float(os.getenv("PROMOTE_PROD_MIN", "1.0"))


def _stage_for(score: float) -> str:
    if score >= PROD_MIN:
        return "Production"
    if score >= STAGING_MIN:
        return "Staging"
    return "None"


def _log_and_promote(name: str, params: dict, score: float, stage: str, tracking_uri: str) -> None:
    if not tracking_uri.startswith(("http://", "https://")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # dev-only local file store
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("skein-promotion")
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("gate_score", score)
        mlflow.set_tag("candidate_stage", stage)
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri)
            try:
                client.create_registered_model(name)
            except Exception:
                pass  # already exists
            version = client.create_model_version(
                name, run.info.artifact_uri, run.info.run_id).version
            if stage != "None":
                client.transition_model_version_stage(
                    name, version, stage, archive_existing_versions=True)
            print("registered {} v{} -> {}".format(name, version, stage))
        except Exception as exc:  # file store has no registry; the run tag still records the stage
            print("model registry unavailable ({}); stage recorded as a run tag. "
                  "Use the MLflow server (docker compose) for the full registry."
                  .format(type(exc).__name__))


def main() -> int:
    settings = get_settings()
    fixtures = os.getenv("GATE_FIXTURES", "evaluation/fixtures/gate.json")
    trace_path = os.path.join(tempfile.mkdtemp(), "gate.jsonl")  # never touch real traces
    result = run_gate(load_gate(fixtures), min_score=PROD_MIN, trace_path=trace_path)
    score = float(result["score"])
    stage = _stage_for(score)

    params = {
        "llm_provider": settings.llm_provider,
        "embed_provider": settings.embed_provider,
        "embed_model": settings.embed_model,
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model,
        "chat_brain": settings.chat_brain,
    }
    tracking_uri = settings.mlflow_url or "./mlruns"
    try:
        _log_and_promote(MODEL_NAME, params, score, stage, tracking_uri)
    except Exception as exc:  # never let an MLflow hiccup hide the gate decision
        print("mlflow logging skipped ({}: {})".format(type(exc).__name__, str(exc)[:120]))

    print("eval score {:.3f} -> stage {} (staging >= {}, prod >= {})".format(
        score, stage, STAGING_MIN, PROD_MIN))
    print(("PROMOTED to {}" if stage != "None" else "REJECTED (below staging bar)").format(stage))
    return 0 if stage != "None" else 1


if __name__ == "__main__":
    sys.exit(main())
