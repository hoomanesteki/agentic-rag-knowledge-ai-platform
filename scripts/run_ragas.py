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

from adapters.config import get_settings
from adapters.factory import make_embedder, make_llm, make_reranker, make_store
from evaluation.golden import load_golden
from evaluation.ragas_eval import evaluate_ragas
from ingest.naming import collection_name
from pipeline.answer import answer_question


def main() -> int:
    settings = get_settings()
    pack = os.path.join("domains", settings.domain)
    if settings.vector_provider in ("memory", "fake", ""):
        print("warning: VECTOR_PROVIDER is offline; scores will be near zero. "
              "Set VECTOR_PROVIDER=qdrant and run make ingest.", file=sys.stderr)

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
            "lang": g.get("lang", "unknown")})
    print(json.dumps(evaluate_ragas(items, judge, embedder), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
