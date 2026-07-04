"""Runtime configuration, read once from the environment (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str
    demo_readonly: bool
    domain: str
    llm_provider: str
    transcribe_provider: str
    whisper_model: str
    embed_provider: str
    embed_model: str
    rerank_provider: str
    rerank_model: str
    vector_provider: str
    graph_provider: str
    groq_api_key: str
    voyage_api_key: str
    cohere_api_key: str
    cohere_api_key_fallback: str
    qdrant_url: str
    qdrant_api_key: str
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str
    rate_limit: str
    jwt_secret: str
    turnstile_secret: str
    auth_db_path: str
    demo_username: str
    demo_password: str
    demo_customer_name: str
    demo_customer_email: str
    admin_username: str
    admin_password: str
    gate_username: str
    gate_password: str
    chat_brain: str
    review_queue_db: str
    mlflow_url: str
    langfuse_url: str
    judge_model: str
    tts_provider: str
    elevenlabs_api_key: str
    elevenlabs_model: str
    elevenlabs_voice_id: str
    elevenlabs_agent_voice_id: str


@lru_cache
def get_settings() -> Settings:
    # Providers default to offline ("fake"/"memory") so a fresh checkout runs with no keys.
    return Settings(
        # "production" hard-enforces a real JWT secret; dev/test only warn (see api/app.py).
        app_env=os.getenv("SKEIN_ENV", "dev").strip().lower(),
        # On a public demo, block the mutating admin endpoints so the documented default admin
        # password cannot be used to drain paid APIs (flywheel re-embed) or corrupt the store.
        demo_readonly=os.getenv("DEMO_READONLY", "false").strip().lower() == "true",
        domain=os.getenv("DOMAIN", "apparel_ecommerce"),
        llm_provider=os.getenv("LLM_PROVIDER", "fake"),
        transcribe_provider=os.getenv("TRANSCRIBE_PROVIDER", "fake"),
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
        embed_provider=os.getenv("EMBED_PROVIDER", "fake"),
        embed_model=os.getenv("EMBED_MODEL", "embed-v4.0"),
        rerank_provider=os.getenv("RERANK_PROVIDER", "none"),
        rerank_model=os.getenv("RERANK_MODEL", "rerank-v3.5"),
        vector_provider=os.getenv("VECTOR_PROVIDER", "memory"),
        graph_provider=os.getenv("GRAPH_PROVIDER", "memory"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        # Optional second Cohere key (e.g. a paid Production key) used only when the primary key
        # returns 429. Lets a free Trial key be the first layer and a paid key the backstop.
        cohere_api_key_fallback=os.getenv("COHERE_API_KEY_FALLBACK", ""),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),  # required by Qdrant Cloud, empty locally
        neo4j_url=os.getenv("NEO4J_URL", "http://localhost:7474"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "skein_password"),
        rate_limit=os.getenv("RATE_LIMIT", "300/minute"),
        jwt_secret=os.getenv("JWT_SECRET", "dev-insecure-change-me"),
        turnstile_secret=os.getenv("TURNSTILE_SECRET_KEY", ""),
        auth_db_path=os.getenv("AUTH_DB_PATH", ".auth.db"),
        demo_username=os.getenv("DEMO_USERNAME", "demo"),
        demo_password=os.getenv("DEMO_PASSWORD", "Canada54321"),
        # The demo shopper's account identity (deployment config, blank by default so no name is
        # baked into engine code). When set, a logged-in shopper is greeted by name and unlocks
        # their OWN orders without re-typing name+email; the login already proved who they are.
        demo_customer_name=os.getenv("DEMO_CUSTOMER_NAME", ""),
        demo_customer_email=os.getenv("DEMO_CUSTOMER_EMAIL", ""),
        admin_username=os.getenv("ADMIN_USERNAME", "admin"),
        admin_password=os.getenv("ADMIN_PASSWORD", "skein-admin-2026"),
        # The public landing page gates the demo behind this one shared credential (emailed to a
        # reviewer). Empty by default, so the gate is open in local dev unless you set it.
        gate_username=os.getenv("GATE_USERNAME", ""),
        gate_password=os.getenv("GATE_PASSWORD", ""),
        # linear streams tokens (the proven default); agent runs the full M6 brain (supervisor,
        # gate, escalation to the review queue) as a buffered response.
        chat_brain=os.getenv("CHAT_BRAIN", "linear"),
        review_queue_db=os.getenv("REVIEW_QUEUE_DB", ".review_queue.db"),
        mlflow_url=os.getenv("MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_URI", ""),
        # the Langfuse UI, shown in the admin console only when tracing is fully configured (both
        # keys); defaults to the Langfuse cloud host, which is where the SDK sends by default
        langfuse_url=(os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com")
        if (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")) else "",
        judge_model=os.getenv("JUDGE_MODEL", ""),  # an independent RAGAS judge; empty = the app LLM
        # Premium voice for the spoken assistant. "none" (default) means the browser's built-in
        # speechSynthesis is used, so voice still works with no key; set to "elevenlabs" for a real,
        # human-sounding voice. The key stays server-side (the browser calls /api/tts).
        tts_provider=os.getenv("TTS_PROVIDER", "none").strip().lower(),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        elevenlabs_model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
        # Premade voices (usable on the free tier; Voice-Library voices need a paid plan): "Sarah",
        # a warm, natural young-adult woman for the assistant (Aria), and "Jessica", a friendly,
        # conversational woman for the human specialist (Sara). Both overridable with any voice id.
        elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
        elevenlabs_agent_voice_id=os.getenv("ELEVENLABS_AGENT_VOICE_ID", "cgSgspJ2msm6clMCkdW9"),
    )
