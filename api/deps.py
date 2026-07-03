"""Build the retrieval/answer components once from config. Tests override this dependency
with offline fakes and a seeded store."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from adapters.config import get_settings
from adapters.factory import (
    make_embedder,
    make_graph,
    make_llm,
    make_reranker,
    make_store,
    make_transcriber,
)
from api.resilience import CachingEmbedder, CachingReranker, ResilientEmbedder, ResilientLLM
from data.metrics import MetricResolver
from ingest.naming import collection_name
from rag.hitl import ReviewQueue
from retrieval.graph import make_graph_retriever

_log = logging.getLogger("skein.api")


def _build_graph_retriever(domain: str):
    # Builds a name index by scanning the graph, so a graph that is down or unbuilt must not take
    # the whole app down; degrade to no graph evidence and log it.
    try:
        return make_graph_retriever(domain, make_graph())
    except Exception as exc:  # noqa: BLE001 - any backend error should degrade, not crash chat
        _log.warning("graph retriever unavailable (%s); answers will not use the graph", exc)
        return None


@lru_cache
def get_components() -> dict:
    settings = get_settings()
    lakehouse_db = os.getenv("LAKEHOUSE_DB", "lakehouse.duckdb")
    # Chat still works vector-only without a lakehouse, but a missing one silently disables the
    # metric layer. Log it once so the loss is visible instead of a mystery in production.
    if not os.path.exists(lakehouse_db):
        _log.warning("lakehouse not found at %s; metric answers are disabled. Run make lakehouse.",
                     lakehouse_db)
    reranker = make_reranker()  # may be None when RERANK_PROVIDER=none
    return {
        # cache query embeds (fewer metered Voyage calls), then retry transient failures
        "embedder": CachingEmbedder(ResilientEmbedder(make_embedder())),
        "store": make_store(collection=collection_name(settings.domain, settings.embed_model)),
        "llm": ResilientLLM(make_llm()),
        "reranker": CachingReranker(reranker) if reranker is not None else None,
        "metric_resolver": MetricResolver(settings.domain, lakehouse_db),
        "graph_retriever": _build_graph_retriever(settings.domain),
        "transcriber": make_transcriber(),
        "review_queue": ReviewQueue(
            settings.review_queue_db,
            verified_path=os.getenv("VERIFIED_PATH", "traces/verified_answers.jsonl")),
        "domain": settings.domain,
    }
