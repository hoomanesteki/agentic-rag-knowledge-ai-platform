"""Runtime configuration, read once from the environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    domain: str
    llm_provider: str
    embed_provider: str
    rerank_provider: str
    vector_provider: str
    groq_api_key: str
    voyage_api_key: str
    qdrant_url: str


@lru_cache
def get_settings() -> Settings:
    # Providers default to offline ("fake"/"memory") so a fresh checkout runs with no keys.
    return Settings(
        domain=os.getenv("DOMAIN", "apparel_ecommerce"),
        llm_provider=os.getenv("LLM_PROVIDER", "fake"),
        embed_provider=os.getenv("EMBED_PROVIDER", "fake"),
        rerank_provider=os.getenv("RERANK_PROVIDER", "fake"),
        vector_provider=os.getenv("VECTOR_PROVIDER", "memory"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
