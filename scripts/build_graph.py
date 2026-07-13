#!/usr/bin/env python3
"""Load the active domain's knowledge graph from the gold lakehouse into the configured graph
store (Neo4j in real use, the in-memory fake offline).

Run: make graph-load
Needs a lakehouse (make lakehouse) and, for real use, Neo4j up (make up) with GRAPH_PROVIDER=neo4j.
"""
from __future__ import annotations

import json
import os
import sys

from adapters.config import get_settings
from adapters.factory import make_graph, make_llm
from knowledge.entity_linking import link_mentions
from knowledge.graph_loader import load_graph

_REVIEW_QUEUE = os.getenv("ENTITY_REVIEW_PATH", "traces/entity_link_review.jsonl")


def main() -> int:
    settings = get_settings()
    db = os.getenv("LAKEHOUSE_DB", "lakehouse.duckdb")
    if not os.path.exists(db):
        print("no lakehouse at {}; run make lakehouse first".format(db), file=sys.stderr)
        return 1
    if settings.graph_provider in ("memory", "fake", ""):
        print("warning: GRAPH_PROVIDER is offline; the graph is built in memory and discarded. "
              "Set GRAPH_PROVIDER=neo4j and run make up to persist it.", file=sys.stderr)

    store = make_graph(timeout=60)  # a batch load may take longer than a request-path read
    try:
        counts = load_graph(settings.domain, db, store)
    except (RuntimeError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        print("hint: is Neo4j up (make up), keys/creds set in .env, and the lakehouse built?",
              file=sys.stderr)
        return 1

    nodes = sum(counts["nodes"].values())
    edges = sum(counts["edges"].values())
    print("loaded {} nodes and {} edges for {}".format(nodes, edges, settings.domain))
    for label, n in counts["nodes"].items():
        print("  node {}: {}".format(label, n))
    for etype, n in counts["edges"].items():
        print("  edge {}: {}".format(etype, n))

    # Entity linking (M5.2): link doc mentions to entities. Needs a real LLM; offline the fake
    # returns no JSON, so this is a no-op and reports zero.
    if settings.llm_provider in ("fake", ""):
        print("note: LLM_PROVIDER is offline, skipping entity linking (set LLM_PROVIDER=groq).",
              file=sys.stderr)
        return 0
    report = link_mentions(settings.domain, store, make_llm())
    print("entity linking: {} mention edge(s) over {} doc(s)".format(report.linked, report.docs))
    if report.review_list:
        os.makedirs(os.path.dirname(_REVIEW_QUEUE) or ".", exist_ok=True)
        with open(_REVIEW_QUEUE, "w", encoding="utf-8") as f:
            for row in report.review_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("  {} low-confidence link(s) to review -> {}".format(
            len(report.review_list), _REVIEW_QUEUE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
