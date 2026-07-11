#!/usr/bin/env python3
"""RAGAS-style answer-quality eval on the golden set: run each question through the pipeline, then
score faithfulness, answer relevance, and context precision/recall with an LLM judge.

Run: make ragas   (needs keys, Qdrant up, and an ingest for real numbers)
The judge is the app LLM here; in production use a separate, stronger judge model.
"""
from __future__ import annotations

import json
import os
import sys
import time
from statistics import mean

from adapters.config import get_settings
from adapters.factory import make_embedder, make_llm, make_reranker, make_store
from evaluation.golden import load_golden
from evaluation.ragas_eval import evaluate_ragas
from ingest.naming import collection_name
from pipeline.answer import answer_question

# Where the measured quality score is persisted so `make promote` can gate on a real answer-quality
# number (not the offline pipeline smoke test), and the frozen champion to beat. RAGAS_MIN is the
# floor; a run below it (or meaningfully below the baseline) exits non-zero so a scheduled job or CI
# can block a quality regression.
SCORES_PATH = os.getenv("RAGAS_SCORES_PATH", "evaluation/ragas_scores.json")
BASELINE_PATH = os.getenv("RAGAS_BASELINE_PATH", "evaluation/ragas_baseline.json")
RAGAS_MIN = float(os.getenv("RAGAS_MIN", "0.6"))
REGRESSION_EPS = 0.02  # tolerate this much noise below the baseline before failing
# Core metrics that a real golden run must actually measure. Each judge metric is a separate LLM
# call, so a rate limit or malformed verdict can null one out; the aggregate would then silently
# average only the survivors and could pass with faithfulness (the hallucination guard) never
# measured. Requiring these to be present distinguishes "not applicable" from "failed to measure".
# context_recall is intentionally NOT required: it needs a ground_truth, which a golden item may
# legitimately lack, so its absence is not a measurement failure.
REQUIRED_METRICS = ("faithfulness", "answer_relevance", "context_precision")


def _aggregate(report: dict) -> float:
    """A single quality number: the mean of the overall RAGAS metrics (0.0 if none scored)."""
    overall = report.get("overall", {})
    return round(mean(overall.values()), 4) if overall else 0.0


def _log_mlflow(record: dict) -> None:
    settings = get_settings()
    tracking_uri = settings.mlflow_url or "./mlruns"
    if not tracking_uri.startswith(("http://", "https://")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("skein-ragas")
        with mlflow.start_run():
            mlflow.log_params({k: record["config"][k] for k in record["config"]})
            mlflow.log_metric("ragas_aggregate", record["aggregate"])
            for metric, value in record["overall"].items():
                mlflow.log_metric(metric, value)
    except Exception as exc:  # never let an MLflow hiccup hide the score
        print("mlflow logging skipped ({}: {})".format(type(exc).__name__, str(exc)[:100]),
              file=sys.stderr)


def main() -> int:
    settings = get_settings()
    pack = os.path.join("domains", settings.domain)
    offline = settings.vector_provider in ("memory", "fake", "")
    if offline:
        print("warning: VECTOR_PROVIDER is offline; scores will be near zero and are NOT gated or "
              "used for promotion. Set VECTOR_PROVIDER=qdrant and run make ingest.",
              file=sys.stderr)

    embedder, llm, reranker = make_embedder(), make_llm(), make_reranker()
    store = make_store(collection=collection_name(settings.domain, settings.embed_model))

    # an independent judge (canonical RAGAS): set JUDGE_MODEL, else the app LLM judges itself
    if settings.judge_model and settings.llm_provider == "groq":
        from adapters.groq import GroqClient
        judge = GroqClient(model=settings.judge_model)
    else:
        judge = llm
        print("note: the judge is the app LLM (set JUDGE_MODEL for an independent judge)",
              file=sys.stderr)

    items = []
    for g in load_golden(pack):
        # refusals are correct behavior, not an answer-quality case, so they are not scored
        if g.get("type") in ("out_of_domain", "unanswerable"):
            continue
        result = answer_question(g["question"], embedder=embedder, store=store, llm=llm,
                                 reranker=reranker, lang=g.get("lang"))
        items.append({
            "question": g["question"], "answer": result.answer, "abstained": result.abstained,
            "contexts": [c["text"] for c in result.contexts],
            "ground_truth": "; ".join(g.get("expected_answer_contains", []) or []),
            "lang": g.get("lang", "unknown"), "difficulty": g.get("difficulty", "untagged")})

    report = evaluate_ragas(items, judge, embedder)
    aggregate = _aggregate(report)
    record = {
        "aggregate": aggregate,
        "overall": report.get("overall", {}),
        "by_language": report.get("by_language", {}),
        "count": report.get("count", 0),
        "offline": offline,
        "ts": time.time(),
        "config": {
            "llm_provider": settings.llm_provider, "embed_provider": settings.embed_provider,
            "embed_model": settings.embed_model, "rerank_provider": settings.rerank_provider,
            "rerank_model": settings.rerank_model,
            "chat_brain": settings.chat_brain, "judge_model": settings.judge_model or "self",
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("ragas aggregate: {:.4f}".format(aggregate))

    # Persist the measured score so `make promote` can gate on it, and log the run to MLflow.
    os.makedirs(os.path.dirname(SCORES_PATH) or ".", exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    _log_mlflow(record)

    if offline:
        return 0  # offline scores are meaningless; do not gate a fake-provider run

    # Coverage gate: a core metric that was never measured (e.g. faithfulness rate-limited to None
    # for every item) must fail, not silently drop out of the average. Otherwise a run could pass on
    # an incomplete score with the hallucination guard never actually computed.
    measured = report.get("overall", {})
    missing = [m for m in REQUIRED_METRICS if m not in measured]
    if missing:
        print("FAIL: required ragas metric(s) not measured: {}".format(", ".join(missing)),
              file=sys.stderr)
        return 1

    # Gate: fail on a score below the floor or meaningfully below the frozen baseline champion.
    if aggregate < RAGAS_MIN:
        print("FAIL: ragas aggregate {:.4f} < RAGAS_MIN {:.2f}".format(aggregate, RAGAS_MIN),
              file=sys.stderr)
        return 1
    if os.path.exists(BASELINE_PATH):
        base = json.load(open(BASELINE_PATH, encoding="utf-8")).get("aggregate", 0.0)
        if aggregate < base - REGRESSION_EPS:
            print("FAIL: ragas aggregate {:.4f} regressed below baseline {:.4f}".format(
                aggregate, base), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
