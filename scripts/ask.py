#!/usr/bin/env python3
"""Ask the active domain a question. Retrieves hybrid, grounds, answers with citations, or
abstains, and writes a trace.

Run: make ask q="What do customers say about sizing?"
Needs keys in .env, Qdrant up (make up), and an ingest done (make ingest).
"""
from __future__ import annotations

import os
import sys

from adapters.config import get_settings
from adapters.factory import make_embedder, make_graph, make_llm, make_reranker, make_store
from data.metrics import MetricResolver
from ingest.naming import collection_name
from pipeline.answer import answer_question
from retrieval.graph import make_graph_retriever


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or os.getenv("q", "").strip()
    if not query:
        print('usage: make ask q="your question"')
        return 2

    settings = get_settings()
    if settings.vector_provider in ("memory", "fake", ""):
        print("warning: VECTOR_PROVIDER is offline and the in-memory store is empty. "
              "Set VECTOR_PROVIDER=qdrant and run make ingest for real answers.",
              file=sys.stderr)

    store = make_store(collection=collection_name(settings.domain, settings.embed_model))
    resolver = MetricResolver(settings.domain, os.getenv("LAKEHOUSE_DB", "lakehouse.duckdb"))
    try:
        graph_retriever = make_graph_retriever(settings.domain, make_graph())
    except Exception as exc:  # noqa: BLE001 - graph is optional; degrade to vector + metric
        print("note: graph unavailable ({}); answering without graph facts".format(exc),
              file=sys.stderr)
        graph_retriever = None
    components = {"embedder": make_embedder(), "store": store, "llm": make_llm(),
                  "reranker": make_reranker(), "metric_resolver": resolver,
                  "graph_retriever": graph_retriever}
    try:
        result = answer_question(query, **components)  # the one gated pipeline
    except RuntimeError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        print("hint: is Qdrant up (make up) and ingested (make ingest), and are keys set in .env?",
              file=sys.stderr)
        return 1

    print("\nTier: {}   Confidence: {:.2f}   Grounding: {:.2f}\n".format(
        result.tier, result.confidence, result.grounding))
    print(result.answer + "\n")
    if result.citations:
        print("Sources:")
        for c in result.citations:
            print("  [{}] {} ({})".format(c["n"], c["id"], c.get("doc_type") or "doc"))
    if result.tier == "auto" and result.grounding < 0.5:
        print("note: parts of this answer may be weakly grounded (grounding {:.2f})".format(
            result.grounding), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
