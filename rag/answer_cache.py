"""An in-process semantic cache for the answers lane: a repeat FAQ ("what is your return policy")
skips retrieval and generation entirely and is served from a near-identical earlier answer.

Scope is deliberately narrow for safety: only the generic answers lane, only anonymous turns, and
only grounded, non-abstained answers are ever cached, so nothing personalized or order-specific can
be stored or served. Entries expire on a TTL and the whole cache is invalidated on a re-index, so a
verified answer is never served after the knowledge behind it changed.

Similarity is cosine over the same embedder the pipeline already uses, thresholded high (0.95) so
only genuine restatements hit. In-process and bounded for demo scale; a shared store (Redis, etc.)
is the scale-up.
"""
from __future__ import annotations

import math
import time


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


class SemanticAnswerCache:
    def __init__(self, embedder, *, threshold: float = 0.95, ttl_s: float = 3600.0,
                 max_entries: int = 512, clock=time.monotonic) -> None:
        self._embedder = embedder
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._entries: list[dict] = []
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> list[float]:
        vec = self._embedder.embed([text])[0]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]  # unit vector, so cosine is a dot product

    def get(self, query: str) -> dict | None:
        if not self._entries:
            self.misses += 1
            return None
        qv = self._embed(_norm(query))
        now = self._clock()
        best, best_sim = None, 0.0
        for e in self._entries:
            if now - e["ts"] > self.ttl_s:
                continue
            sim = sum(a * b for a, b in zip(qv, e["vec"]))
            if sim > best_sim:
                best, best_sim = e, sim
        if best is not None and best_sim >= self.threshold:
            self.hits += 1
            return {"answer": best["answer"], "citations": best["citations"],
                    "grounding": best["grounding"], "confidence": best["confidence"],
                    "similarity": round(best_sim, 4)}
        self.misses += 1
        return None

    def put(self, query: str, answer: str, citations, *, grounding=None, confidence=None) -> None:
        if not answer:
            return
        self._entries.append({"vec": self._embed(_norm(query)), "answer": answer,
                              "citations": citations or [], "grounding": grounding,
                              "confidence": confidence, "ts": self._clock()})
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]  # evict oldest

    def invalidate(self) -> None:
        """Drop everything. Call on a re-index so a stale verified answer is never served."""
        self._entries.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}
