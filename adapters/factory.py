"""Choose adapter implementations from config.

Defaults to offline fakes so the app runs with no API keys. Real providers (voyage, qdrant,
groq) are config swaps, not code changes.
"""
from __future__ import annotations

from . import fakes
from .base import Embedder, GraphStore, HybridStore, LLMClient, Reranker, Transcriber, VectorStore
from .config import get_settings


def make_embedder(provider: str | None = None) -> Embedder:
    provider = provider or get_settings().embed_provider
    if provider in ("fake", ""):
        return fakes.HashEmbedder()
    if provider == "voyage":
        from .voyage import VoyageEmbedder
        return VoyageEmbedder()
    if provider == "cohere":
        from .cohere import CohereEmbedder
        return CohereEmbedder()
    raise ValueError("unknown EMBED_PROVIDER: {}".format(provider))


def make_llm(provider: str | None = None) -> LLMClient:
    provider = provider or get_settings().llm_provider
    if provider in ("fake", ""):
        return fakes.EchoLLM()
    if provider == "groq":
        from .groq import GroqClient
        return GroqClient()
    raise ValueError("unknown LLM_PROVIDER: {}".format(provider))


def make_transcriber(provider: str | None = None) -> Transcriber:
    """Speech to text. Offline uses a fixed-transcript fake, real uses Groq hosted Whisper."""
    provider = provider or get_settings().transcribe_provider
    if provider in ("fake", ""):
        return fakes.FakeTranscriber()
    if provider == "groq":
        from .groq_whisper import GroqWhisper
        return GroqWhisper()
    raise ValueError("unknown TRANSCRIBE_PROVIDER: {}".format(provider))


def make_reranker(provider: str | None = None) -> Reranker | None:
    provider = provider or get_settings().rerank_provider
    if provider in ("none", ""):
        return None
    if provider == "fake":
        return fakes.LexicalReranker()
    if provider == "voyage":
        from .voyage_rerank import VoyageReranker
        return VoyageReranker()
    if provider == "cohere":
        from .cohere import CohereReranker
        return CohereReranker()
    raise ValueError("unknown RERANK_PROVIDER: {}".format(provider))


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


def make_graph(provider: str | None = None) -> GraphStore:
    """The knowledge graph. Offline uses an in-memory fake, real uses Neo4j over its HTTP API."""
    provider = provider or get_settings().graph_provider
    if provider in ("memory", "fake", ""):
        return fakes.InMemoryGraphStore()
    if provider == "neo4j":
        from .neo4j_store import Neo4jGraphStore
        return Neo4jGraphStore()
    raise ValueError("unknown GRAPH_PROVIDER: {}".format(provider))


def make_vector_store(provider: str | None = None) -> VectorStore:
    """Simple dense-only store (offline fake). For hybrid retrieval use make_store."""
    provider = provider or "memory"
    if provider in ("memory", "fake", ""):
        return fakes.InMemoryVectorStore()
    raise ValueError(
        "unknown vector store: {} (use make_store for the qdrant hybrid path)".format(provider))
