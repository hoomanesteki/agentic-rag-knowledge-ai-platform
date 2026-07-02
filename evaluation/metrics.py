"""Small, pure retrieval metrics. Each takes a list of relevance flags in rank order."""
from __future__ import annotations


def hit_at_k(relevance_flags: list[bool]) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0 (a.k.a. recall@k for one target)."""
    return 1.0 if any(relevance_flags) else 0.0


def reciprocal_rank(relevance_flags: list[bool]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none is relevant."""
    for rank, relevant in enumerate(relevance_flags, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
