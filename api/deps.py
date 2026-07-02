"""Build the retrieval/answer components once from config. Tests override this dependency
with offline fakes and a seeded store."""
from __future__ import annotations

from functools import lru_cache

from adapters.config import get_settings
from adapters.factory import make_embedder, make_llm, make_reranker, make_store
from api.resilience import ResilientEmbedder
from ingest.naming import collection_name


@lru_cache
def get_components() -> dict:
    settings = get_settings()
    return {
        "embedder": ResilientEmbedder(make_embedder()),
        "store": make_store(collection=collection_name(settings.domain, settings.embed_model)),
        "llm": make_llm(),
        "reranker": make_reranker(),
    }
