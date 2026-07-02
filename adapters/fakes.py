"""Offline, in-memory adapter implementations.

They need no API keys and no network, so tests and a first end-to-end run work anywhere.
The hosted implementations (Voyage, Groq, Qdrant) arrive at M1.2 and M1.3 behind the same
interfaces, and swapping them in is a config change.
"""
from __future__ import annotations

import hashlib
import math

from .base import Chunk


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashEmbedder:
    """Deterministic hashing bag-of-words embedder. Crude, but real vectors with no network."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self._dim
            for token in text.lower().split():
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
                vec[bucket] += 1.0
            out.append(_l2(vec))
        return out


class InMemoryVectorStore:
    """Cosine-similarity search over vectors held in memory, with metadata filtering."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        replacement = {c.id: (c, list(v)) for c, v in zip(chunks, vectors)}
        self._items = [(c, v) for c, v in self._items if c.id not in replacement]
        self._items.extend(replacement.values())

    def search(self, vector: list[float], top_k: int = 8,
               where: dict | None = None) -> list[Chunk]:
        hits = []
        for chunk, vec in self._items:
            if where and any(chunk.metadata.get(k) != val for k, val in where.items()):
                continue
            score = sum(a * b for a, b in zip(vector, vec))
            hits.append(Chunk(id=chunk.id, text=chunk.text, score=score,
                              metadata=chunk.metadata))
        hits.sort(key=lambda c: c.score, reverse=True)
        return hits[:top_k]


class EchoLLM:
    """Offline placeholder LLM. Returns a fixed string so the seam is testable without keys."""

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 512) -> str:
        return "offline-fake-response"
