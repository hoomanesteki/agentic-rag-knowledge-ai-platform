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
    embed_model: str
    rerank_provider: str
    rerank_model: str
    vector_provider: str
    graph_provider: str
    groq_api_key: str
    voyage_api_key: str
    qdrant_url: str
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str
    rate_limit: str
    jwt_secret: str
    turnstile_secret: str
    auth_db_path: str
    demo_username: str
    demo_password: str
    chat_brain: str
    review_queue_db: str


@lru_cache
def get_settings() -> Settings:
    # Providers default to offline ("fake"/"memory") so a fresh checkout runs with no keys.
    return Settings(
        domain=os.getenv("DOMAIN", "apparel_ecommerce"),
        llm_provider=os.getenv("LLM_PROVIDER", "fake"),
        embed_provider=os.getenv("EMBED_PROVIDER", "fake"),
        embed_model=os.getenv("EMBED_MODEL", "voyage-3-large"),
        rerank_provider=os.getenv("RERANK_PROVIDER", "none"),
        rerank_model=os.getenv("RERANK_MODEL", "rerank-2.5"),
        vector_provider=os.getenv("VECTOR_PROVIDER", "memory"),
        graph_provider=os.getenv("GRAPH_PROVIDER", "memory"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        neo4j_url=os.getenv("NEO4J_URL", "http://localhost:7474"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "skein_password"),
        rate_limit=os.getenv("RATE_LIMIT", "30/minute"),
        jwt_secret=os.getenv("JWT_SECRET", "dev-insecure-change-me"),
        turnstile_secret=os.getenv("TURNSTILE_SECRET_KEY", ""),
        auth_db_path=os.getenv("AUTH_DB_PATH", ".auth.db"),
        demo_username=os.getenv("DEMO_USERNAME", "demo"),
        demo_password=os.getenv("DEMO_PASSWORD", "skein-demo-2026"),
        # linear streams tokens (the proven default); agent runs the full M6 brain (supervisor,
        # gate, escalation to the review queue) as a buffered response.
        chat_brain=os.getenv("CHAT_BRAIN", "linear"),
        review_queue_db=os.getenv("REVIEW_QUEUE_DB", ".review_queue.db"),
    )
