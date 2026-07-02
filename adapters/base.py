"""Adapter interfaces. These are the seams the whole engine is built on."""
from __future__ import annotations

from collections.abc import Iterator
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

    # input_type lets asymmetric models embed queries and documents differently.
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]: ...


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    # Returns text plus token usage so every request can be traced and costed.
    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 512) -> LLMResult: ...

    # Yields answer text incrementally for streaming responses.
    def stream(self, prompt: str, *, system: str | None = None,
               max_tokens: int = 512) -> Iterator[str]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Simple dense store (the offline fake path)."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], top_k: int = 8,
               where: dict | None = None) -> list[Chunk]: ...


@runtime_checkable
class Reranker(Protocol):
    # Returns (original_index, score) pairs, highest score first, at most top_n of them.
    def rerank(self, query: str, documents: list[str],
               top_n: int = 8) -> list[tuple[int, float]]: ...


@dataclass
class GraphNode:
    """A node in the knowledge graph: a label (Product, Supplier, ...), its primary-key
    property name and value, and the rest of its properties."""

    label: str
    key: str
    id: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphNeighbor:
    """A node reached from an anchor, with the edge that connected them. direction is relative
    to the anchor: "out" means anchor -> node, "in" means node -> anchor."""

    edge_type: str
    direction: str
    node: GraphNode


@runtime_checkable
class GraphStore(Protocol):
    """A property graph the loader writes and the retriever traverses. The Neo4j impl turns
    these typed calls into parameterized, allowlisted Cypher; the in-memory fake runs the same
    traversals over dicts, so the retrieval code is written once and tested with no database.
    """

    def reset(self) -> None: ...

    # Ontology constraints (from the pack's ontology.cypher). The fake treats them as a no-op.
    def apply_schema(self, statements: list[str]) -> None: ...

    # rows are property dicts that each include the key property; the id is row[key].
    def upsert_nodes(self, label: str, key: str, rows: list[dict]) -> int: ...

    def upsert_edges(self, edge_type: str, from_label: str, from_key: str,
                     to_label: str, to_key: str, pairs: list[tuple[str, str]]) -> int: ...

    def get_node(self, label: str, key: str, value: str) -> "GraphNode | None": ...

    # All nodes of a label, optionally filtered by exact property matches. Powers entity-linking
    # candidate lookup (M5.2) and graph-first relational queries (M5.3).
    def find_nodes(self, label: str, where: dict | None = None,
                   limit: int = 1000) -> "list[GraphNode]": ...

    def neighbors(self, label: str, key: str, value: str, *, edge_type: str | None = None,
                  direction: str = "both", to_label: str | None = None,
                  limit: int = 50) -> "list[GraphNeighbor]": ...


@runtime_checkable
class HybridStore(Protocol):
    """Dense + sparse store with reciprocal-rank fusion. Implemented by the in-memory fake
    and by Qdrant, so retrieval code is written once and the backend is a config swap.

    A point is a dict: {id, text, payload, dense: [...], sparse: {indices, values}}.
    A hit is a dict: {id, score, payload} (payload carries text and metadata).
    """

    def ensure_collection(self, dense_dim: int) -> None: ...

    def upsert(self, points: list[dict]) -> None: ...

    def hybrid_search(self, dense_query: list[float], sparse_query: dict,
                      top_k: int = 8, where: dict | None = None,
                      dense_only: bool = False) -> list[dict]: ...
