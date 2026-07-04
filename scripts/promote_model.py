#!/usr/bin/env python3
"""Promote the current RAG configuration through MLflow stages, gated by an eval score.

A "model" here is the serving configuration (LLM, embeddings, reranker, prompt/brain). Promotion is
not automatic: the config is logged to MLflow and transitioned to Production only if it clears the
production bar, to Staging if it clears the staging bar, otherwise left as a rejected candidate.
This is the dev -> staging -> prod gate flow.

How the gate works (two stages, in order):
  1. PRECONDITION: the offline CI eval gate (`evaluation/ci_gate.py`) must pass on recorded
     fixtures with deterministic fakes. This proves the retrieval/grounding/abstain PIPELINE is
     wired; it is not a quality score and can never by itself promote.
  2. QUALITY GATE: promotion is decided by the measured RAGAS answer-quality aggregate from the
     REAL providers, written to evaluation/ragas_scores.json by `make ragas`. A candidate reaches
     Staging at PROMOTE_STAGING_MIN and Production at PROMOTE_PROD_MIN, and a Production candidate
     must also beat the frozen champion (evaluation/ragas_baseline.json) or it is capped at Staging.

If no real RAGAS score exists (or it was produced offline with fakes), promotion is REJECTED with
guidance to run `make ragas` first: you cannot promote on a quality you never measured. This is the
fix for the old behavior, where the gate scored fake providers and always promoted to Production.

MLflow logs to ./mlruns by default. The full model registry needs the MLflow server (docker
compose, MLFLOW_TRACKING_URI); without it the stage is recorded as a run tag and printed.

Run: make promote   (or: uv run python scripts/promote_model.py)
Exit code is 0 when promoted (Staging or Production), 1 when rejected, so CI can block on it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

from adapters.config import get_settings
from evaluation.ci_gate import load_gate, run_gate

MODEL_NAME = os.getenv("PROMOTE_MODEL_NAME", "skein-rag")
# These bars apply to the measured RAGAS answer-quality aggregate (0..1), not the offline pipeline
# smoke score. Typical RAGAS aggregates land in 0.6..0.9, so the defaults are set for that range.
STAGING_MIN = float(os.getenv("PROMOTE_STAGING_MIN", "0.65"))
PROD_MIN = float(os.getenv("PROMOTE_PROD_MIN", "0.78"))
RAGAS_SCORES_PATH = os.getenv("RAGAS_SCORES_PATH", "evaluation/ragas_scores.json")
BASELINE_PATH = os.getenv("RAGAS_BASELINE_PATH", "evaluation/ragas_baseline.json")


def _load_quality() -> dict | None:
    """The RAGAS answer-quality record written by `make ragas` with the real providers. Returns None
    when it is missing or was produced offline (fake providers), because you cannot promote on a
    quality you never measured."""
    if not os.path.exists(RAGAS_SCORES_PATH):
        return None
    try:
        record = json.load(open(RAGAS_SCORES_PATH, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return None if record.get("offline") else record


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

    # 1) PRECONDITION: the offline pipeline smoke gate must pass (retrieval/grounding/abstain are
    # wired). This is a gate on the pipeline, not on answer quality, so it can never itself promote.
    fixtures = os.getenv("GATE_FIXTURES", "evaluation/fixtures/gate.json")
    trace_path = os.path.join(tempfile.mkdtemp(), "gate.jsonl")  # never touch real traces
    smoke = run_gate(load_gate(fixtures), min_score=1.0, trace_path=trace_path)
    if not smoke["passed"]:
        print("REJECTED: pipeline smoke gate failed (score {:.3f}); fix the pipeline before "
              "promoting.".format(float(smoke["score"])))
        return 1

    # 2) QUALITY GATE: promote on the measured RAGAS answer-quality score from the REAL providers.
    quality = _load_quality()
    if quality is None:
        print("REJECTED: no measured answer-quality score. The pipeline is wired (smoke gate "
              "passed), but promotion requires a real RAGAS run: set the real providers, "
              "`make ingest`, then `make ragas`, and re-run `make promote`.")
        return 1

    # The score must have been measured on the SAME serving config we are about to promote, or it is
    # stale: promoting config B on config A's number would ship an unmeasured change. Fail closed.
    measured = quality.get("config", {}) or {}
    serving = {"llm_provider": settings.llm_provider, "embed_provider": settings.embed_provider,
               "embed_model": settings.embed_model, "rerank_provider": settings.rerank_provider,
               "rerank_model": settings.rerank_model, "chat_brain": settings.chat_brain}
    drift = {k: (measured.get(k), v) for k, v in serving.items() if measured.get(k) != v}
    if drift:
        print("REJECTED: the RAGAS score was measured on a different config than the one being "
              "promoted, so it is stale. Mismatches (measured -> current): {}. Re-run `make ragas` "
              "on the current config first.".format(drift))
        return 1

    score = quality.get("aggregate")
    if not isinstance(score, (int, float)):
        print("REJECTED: the persisted RAGAS score is missing or malformed; re-run `make ragas`.")
        return 1
    score = float(score)
    stage = _stage_for(score)

    # A Production candidate must also beat the frozen champion, so quality never silently slips.
    # With no champion committed yet, cap at Staging rather than promote to Production unchecked:
    # the first Production promotion should always be deliberate, compared against a baseline.
    champion = None
    if stage == "Production" and not os.path.exists(BASELINE_PATH):
        print("note: no champion baseline ({}) yet, so capping at Staging. Freeze this score as "
              "ragas_baseline.json to enable Production promotion.".format(BASELINE_PATH))
        stage = "Staging"
    if stage == "Production" and os.path.exists(BASELINE_PATH):
        champion = float(json.load(open(BASELINE_PATH, encoding="utf-8")).get("aggregate", 0.0))
        if score <= champion:
            stage = "Staging"  # good enough to stage, but not better than the current champion

    params = {
        "llm_provider": settings.llm_provider,
        "embed_provider": settings.embed_provider,
        "embed_model": settings.embed_model,
        "rerank_provider": settings.rerank_provider,
        "rerank_model": settings.rerank_model,
        "chat_brain": settings.chat_brain,
        "ragas_aggregate": score,
        "smoke_score": float(smoke["score"]),
    }
    tracking_uri = settings.mlflow_url or "./mlruns"
    try:
        _log_and_promote(MODEL_NAME, params, score, stage, tracking_uri)
    except Exception as exc:  # never let an MLflow hiccup hide the gate decision
        print("mlflow logging skipped ({}: {})".format(type(exc).__name__, str(exc)[:120]))

    champ_note = "" if champion is None else " (champion {:.3f})".format(champion)
    print("ragas quality {:.3f} -> stage {} (staging >= {}, prod >= {}){}".format(
        score, stage, STAGING_MIN, PROD_MIN, champ_note))
    print(("PROMOTED to {}" if stage != "None" else "REJECTED (below staging bar)").format(stage))
    return 0 if stage != "None" else 1


if __name__ == "__main__":
    sys.exit(main())
