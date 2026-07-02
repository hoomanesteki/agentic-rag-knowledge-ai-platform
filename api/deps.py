"""Build the retrieval/answer components once from config. Tests override this dependency
with offline fakes and a seeded store."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from adapters.config import get_settings
from adapters.factory import make_embedder, make_llm, make_reranker, make_store
from api.resilience import ResilientEmbedder
from data.metrics import MetricResolver
from ingest.naming import collection_name

_log = logging.getLogger("skein.api")


@lru_cache
def get_components() -> dict:
    settings = get_settings()
    lakehouse_db = os.getenv("LAKEHOUSE_DB", ".lakehouse.duckdb")
    # Chat still works vector-only without a lakehouse, but a missing one silently disables the
    # metric layer. Log it once so the loss is visible instead of a mystery in production.
    if not os.path.exists(lakehouse_db):
        _log.warning("lakehouse not found at %s; metric answers are disabled. Run make lakehouse.",
                     lakehouse_db)
    return {
        "embedder": ResilientEmbedder(make_embedder()),
        "store": make_store(collection=collection_name(settings.domain, settings.embed_model)),
        "llm": make_llm(),
        "reranker": make_reranker(),
        "metric_resolver": MetricResolver(settings.domain, lakehouse_db),
    }
