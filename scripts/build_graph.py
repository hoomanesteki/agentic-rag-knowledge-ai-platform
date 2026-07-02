#!/usr/bin/env python3
"""Load the active domain's knowledge graph from the gold lakehouse into the configured graph
store (Neo4j in real use, the in-memory fake offline).

Run: make graph-load
Needs a lakehouse (make lakehouse) and, for real use, Neo4j up (make up) with GRAPH_PROVIDER=neo4j.
"""
from __future__ import annotations

import os
import sys

from adapters.config import get_settings
from adapters.factory import make_graph
from knowledge.graph_loader import load_graph


def main() -> int:
    settings = get_settings()
    db = os.getenv("LAKEHOUSE_DB", ".lakehouse.duckdb")
    if not os.path.exists(db):
        print("no lakehouse at {}; run make lakehouse first".format(db), file=sys.stderr)
        return 1
    if settings.graph_provider in ("memory", "fake", ""):
        print("warning: GRAPH_PROVIDER is offline; the graph is built in memory and discarded. "
              "Set GRAPH_PROVIDER=neo4j and run make up to persist it.", file=sys.stderr)

    store = make_graph()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
