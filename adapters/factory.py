"""Choose adapter implementations from config.

Defaults to offline fakes so the app runs with no API keys. Real providers (voyage, qdrant,
groq) are config swaps, not code changes.
"""
from __future__ import annotations

from . import fakes
from .base import Embedder, HybridStore, LLMClient, VectorStore
from .config import get_settings


def make_embedder(provider: str | None = None) -> Embedder:
    provider = provider or get_settings().embed_provider
    if provider in ("fake", ""):
        return fakes.HashEmbedder()
    if provider == "voyage":
        from .voyage import VoyageEmbedder
        return VoyageEmbedder()
    raise ValueError("unknown EMBED_PROVIDER: {}".format(provider))


def make_llm(provider: str | None = None) -> LLMClient:
    provider = provider or get_settings().llm_provider
    if provider in ("fake", ""):
        return fakes.EchoLLM()
    if provider == "groq":
        from .groq import GroqClient
        return GroqClient()
    raise ValueError("unknown LLM_PROVIDER: {}".format(provider))


def make_store(provider: str | None = None, collection: str | None = None) -> HybridStore:
    """The hybrid (dense + sparse) store. Offline uses an in-memory fake, real uses Qdrant."""
    provider = provider or get_settings().vector_provider
    if provider in ("memory", "fake", ""):
        return fakes.InMemoryHybridStore()
    if provider == "qdrant":
        if not collection:
            raise ValueError("make_store('qdrant') needs a collection name")
        from .qdrant_store import QdrantStore
        return QdrantStore(collection=collection)
    raise ValueError("unknown vector provider: {}".format(provider))


def make_vector_store(provider: str | None = None) -> VectorStore:
    """Simple dense-only store (offline fake). For hybrid retrieval use make_store."""
    provider = provider or "memory"
    if provider in ("memory", "fake", ""):
        return fakes.InMemoryVectorStore()
    raise ValueError(
        "unknown vector store: {} (use make_store for the qdrant hybrid path)".format(provider))
