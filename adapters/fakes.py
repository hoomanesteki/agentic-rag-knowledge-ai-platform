"""Offline, in-memory adapter implementations.

They need no API keys and no network, so tests and a first end-to-end run work anywhere.
The hosted implementations (Voyage, Qdrant) live behind the same interfaces, and swapping
them in is a config change.
"""
from __future__ import annotations

import hashlib
import math

from .base import Chunk, GraphNeighbor, GraphNode, LLMResult

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
                      top_k: int = 8, where: dict | None = None,
                      dense_only: bool = False) -> list[dict]:
        candidates = [p for p in self._points.values() if _matches(p["payload"], where)]

        def out(pid, score):
            p = by_id[pid]
            return {"id": pid, "score": score,
                    "payload": {**p["payload"], "text": p["text"], "chunk_id": pid}}

        by_id = {p["id"]: p for p in candidates}
        if dense_only:
            scored = sorted(((p["id"], _cosine(dense_query, p["dense"])) for p in candidates),
                            key=lambda kv: kv[1], reverse=True)[:top_k]
            return [out(pid, score) for pid, score in scored]

        dense_ranked = sorted(candidates, key=lambda p: _cosine(dense_query, p["dense"]),
                              reverse=True)
        sparse_ranked = sorted(candidates, key=lambda p: _sparse_dot(sparse_query, p["sparse"]),
                               reverse=True)
        fused: dict[str, float] = {}
        for ranked in (dense_ranked, sparse_ranked):
            for rank, point in enumerate(ranked):
                fused[point["id"]] = fused.get(point["id"], 0.0) + 1.0 / (_RRF_K + rank + 1)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [out(pid, score) for pid, score in top]


class LexicalReranker:
    """Offline reranker: scores documents by word overlap with the query. Crude, but it
    reorders deterministically with no network, mirroring a cross-encoder's role."""

    def rerank(self, query: str, documents: list[str],
               top_n: int = 8) -> list[tuple[int, float]]:
        q = set(query.lower().split())
        scored = [(i, float(len(q & set(doc.lower().split())))) for i, doc in enumerate(documents)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]


class InMemoryGraphStore:
    """Offline GraphStore: nodes in a dict, edges in a list, traversals in Python. Mirrors the
    Neo4j impl's behavior so the loader and retriever are tested without a database."""

    def __init__(self) -> None:
        self._nodes: dict[tuple[str, str], GraphNode] = {}
        self._edges: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
        self._edge_sigs: set[tuple] = set()     # dedup across calls, like Neo4j MERGE
        self._label_keys: dict[str, str] = {}   # label -> its key property, to fill neighbor ids

    def reset(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._edge_sigs.clear()
        self._label_keys.clear()

    def apply_schema(self, statements: list[str]) -> None:
        pass  # uniqueness constraints are a real-database concern; the dict store is exact

    def upsert_nodes(self, label: str, key: str, rows: list[dict]) -> int:
        self._label_keys[label] = key
        for row in rows:
            node_id = str(row[key])
            self._nodes[(label, node_id)] = GraphNode(
                label=label, key=key, id=node_id, properties=dict(row))
        return len(rows)

    def upsert_edges(self, edge_type: str, from_label: str, from_key: str,
                     to_label: str, to_key: str, pairs: list[tuple[str, str]]) -> int:
        added = 0
        for from_id, to_id in pairs:
            sig = (edge_type, from_label, str(from_id), to_label, str(to_id))
            if sig in self._edge_sigs:
                continue
            self._edge_sigs.add(sig)
            self._edges.append(
                (edge_type, (from_label, str(from_id)), (to_label, str(to_id))))
            added += 1
        return added

    def get_node(self, label: str, key: str, value: str) -> GraphNode | None:
        # mirror Neo4j: matching on a key this label was not loaded with finds nothing
        if label in self._label_keys and self._label_keys[label] != key:
            return None
        return self._nodes.get((label, str(value)))

    def find_nodes(self, label: str, where: dict | None = None,
                   limit: int = 1000) -> list[GraphNode]:
        out = []
        for (node_label, _id), node in self._nodes.items():
            if node_label != label:
                continue
            if where and any(node.properties.get(k) != v for k, v in where.items()):
                continue
            out.append(node)
            if len(out) >= limit:
                break
        return out

    def neighbors(self, label: str, key: str, value: str, *, edge_type: str | None = None,
                  direction: str = "both", to_label: str | None = None,
                  limit: int = 50) -> list[GraphNeighbor]:
        anchor = (label, str(value))
        out: list[GraphNeighbor] = []
        for etype, src, dst in self._edges:
            if edge_type and etype != edge_type:
                continue
            if src == anchor and direction in ("out", "both"):
                far, seen_dir = dst, "out"
            elif dst == anchor and direction in ("in", "both"):
                far, seen_dir = src, "in"
            else:
                continue
            if to_label and far[0] != to_label:
                continue
            node = self._nodes.get(far)
            if node is None:
                continue  # edge points at a node that was not loaded; skip it
            out.append(GraphNeighbor(edge_type=etype, direction=seen_dir, node=node))
            if len(out) >= limit:
                break
        return out


class EchoLLM:
    """Offline placeholder LLM. Returns a fixed string plus rough token counts so the seam
    (and tracing) is testable without keys."""

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 512) -> LLMResult:
        return LLMResult(text="offline-fake-response",
                         prompt_tokens=max(len(prompt) // 4, 1),
                         completion_tokens=3, model="fake")

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int = 512):
        for word in ("offline-fake-response",):
            yield word
