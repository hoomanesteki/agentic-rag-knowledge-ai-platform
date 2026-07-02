"""Choose adapter implementations from config.

Defaults to the offline fakes so the app runs with no API keys. Real providers (voyage,
groq, qdrant) get wired in at M1.2 and M1.3 and raise a clear message until then.
"""
from __future__ import annotations

from . import fakes
from .base import Embedder, LLMClient, VectorStore
from .config import get_settings


def make_embedder(provider: str | None = None) -> Embedder:
    provider = provider or get_settings().embed_provider
    if provider in ("fake", ""):
        return fakes.HashEmbedder()
    if provider == "voyage":
        raise NotImplementedError(
            "voyage embedder arrives at M1.2; set EMBED_PROVIDER=fake to run offline")
    raise ValueError("unknown EMBED_PROVIDER: {}".format(provider))


def make_llm(provider: str | None = None) -> LLMClient:
    provider = provider or get_settings().llm_provider
    if provider in ("fake", ""):
        return fakes.EchoLLM()
    if provider == "groq":
        raise NotImplementedError(
            "groq client arrives at M1.3; set LLM_PROVIDER=fake to run offline")
    raise ValueError("unknown LLM_PROVIDER: {}".format(provider))


def make_vector_store(provider: str | None = None) -> VectorStore:
    provider = provider or "memory"
    if provider in ("memory", "fake", ""):
        return fakes.InMemoryVectorStore()
    if provider == "qdrant":
        raise NotImplementedError(
            "qdrant store arrives at M1.2; use the in-memory store to run offline")
    raise ValueError("unknown vector store: {}".format(provider))
