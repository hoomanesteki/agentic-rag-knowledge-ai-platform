#!/usr/bin/env python3
"""Evaluate the active domain's retrieval and abstain gate against its golden set.

Run: make eval    (needs keys in .env, Qdrant up, and an ingest done for real numbers)
Offline (VECTOR_PROVIDER=memory) the store is empty, so scores are zero; that is expected.
"""
from __future__ import annotations

import os
import sys

import yaml

from adapters.config import get_settings
from adapters.factory import make_embedder, make_reranker, make_store
from evaluation.golden import load_golden
from evaluation.harness import evaluate, format_scorecard
from ingest.naming import collection_name


def _entity_fields(pack: str) -> list[str]:
    with open(os.path.join(pack, "domain.yaml"), encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    fields = set()
    for src in (manifest.get("sources", {}) or {}).get("unstructured", []) or []:
        fields.update(src.get("entity_ref", []) or [])
    return sorted(fields)


def main() -> int:
    settings = get_settings()
    pack = os.path.join("domains", settings.domain)
    if not os.path.isfile(os.path.join(pack, "eval", "golden.jsonl")):
        print("no golden set at {}/eval/golden.jsonl".format(pack))
        return 1
    if settings.vector_provider in ("memory", "fake", ""):
        print("warning: VECTOR_PROVIDER is offline and the in-memory store is empty, so "
              "retrieval will be zero and abstain_recall will trivially read 1.0. "
              "Set VECTOR_PROVIDER=qdrant and run make ingest.", file=sys.stderr)

    no_rerank = "--no-rerank" in sys.argv  # ablation switch: score hybrid without reranking
    reranker = None if no_rerank else make_reranker()
    label = "{} rerank={}".format(
        settings.domain, "off" if reranker is None else settings.rerank_provider)

    golden = load_golden(pack)
    store = make_store(collection=collection_name(settings.domain, settings.embed_model))
    try:
        scorecard = evaluate(golden, embedder=make_embedder(), store=store,
                             reranker=reranker, entity_fields=_entity_fields(pack))
    except RuntimeError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        print("hint: is Qdrant up (make up) and ingested (make ingest), and are the hosted "
              "APIs (Voyage/Groq) reachable and not rate-limited?", file=sys.stderr)
        return 1

    print(format_scorecard(scorecard, label=label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
