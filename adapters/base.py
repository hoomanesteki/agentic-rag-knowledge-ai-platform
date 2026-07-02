"""Adapter interfaces. These are the seams the whole engine is built on."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Chunk:
    """A retrievable unit of text with its metadata and (after search) a score."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 512) -> str: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], top_k: int = 8,
               where: dict | None = None) -> list[Chunk]: ...
