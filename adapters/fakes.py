"""Offline, in-memory adapter implementations.

They need no API keys and no network, so tests and a first end-to-end run work anywhere.
The hosted implementations (Voyage, Qdrant) live behind the same interfaces, and swapping
them in is a config change.
"""
from __future__ import annotations

import hashlib
import math

from .base import Chunk, LLMResult

_RRF_K = 60


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are already L2-normalized


def _sparse_dot(q: dict, p: dict) -> float:
    pv = dict(zip(p.get("indices", []), p.get("values", [])))
    qi, qv = q.get("indices", []), q.get("values", [])
    return sum(val * pv.get(idx, 0.0) for idx, val in zip(qi, qv))


def _matches(payload: dict, where: dict | None) -> bool:
    return not where or all(payload.get(k) == v for k, v in where.items())


class HashEmbedder:
    """Deterministic hashing bag-of-words embedder. Crude, but real vectors with no network."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self._dim
            for token in text.lower().split():
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
                vec[bucket] += 1.0
            out.append(_l2(vec))
        return out


class InMemoryVectorStore:
    """Cosine-similarity search over dense vectors held in memory, with metadata filtering."""

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
            if not _matches(chunk.metadata, where):
                continue
            hits.append(Chunk(id=chunk.id, text=chunk.text, score=_cosine(vector, vec),
                              metadata=chunk.metadata))
        hits.sort(key=lambda c: c.score, reverse=True)
        return hits[:top_k]


class InMemoryHybridStore:
    """Offline HybridStore: dense cosine plus sparse dot, fused with reciprocal-rank fusion.
    Mirrors the shape QdrantStore returns so retrieval code is identical for both."""

    def __init__(self) -> None:
        self._points: dict[str, dict] = {}

    def ensure_collection(self, dense_dim: int) -> None:
        pass

    def upsert(self, points: list[dict]) -> None:
        for p in points:
            self._points[p["id"]] = p

    def hybrid_search(self, dense_query: list[float], sparse_query: dict,
                      top_k: int = 8, where: dict | None = None) -> list[dict]:
        candidates = [p for p in self._points.values() if _matches(p["payload"], where)]
        dense_ranked = sorted(candidates, key=lambda p: _cosine(dense_query, p["dense"]),
                              reverse=True)
        sparse_ranked = sorted(candidates, key=lambda p: _sparse_dot(sparse_query, p["sparse"]),
                               reverse=True)
        fused: dict[str, float] = {}
        for ranked in (dense_ranked, sparse_ranked):
            for rank, point in enumerate(ranked):
                fused[point["id"]] = fused.get(point["id"], 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_id = {p["id"]: p for p in candidates}
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            {"id": pid, "score": score,
             "payload": {**by_id[pid]["payload"], "text": by_id[pid]["text"], "chunk_id": pid}}
            for pid, score in top
        ]


class EchoLLM:
    """Offline placeholder LLM. Returns a fixed string plus rough token counts so the seam
    (and tracing) is testable without keys."""

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 512) -> LLMResult:
        return LLMResult(text="offline-fake-response",
                         prompt_tokens=max(len(prompt) // 4, 1),
                         completion_tokens=3, model="fake")
