"""A BM25-style sparse encoder for hybrid search.

This is classic lexical weighting (term frequency times inverse document frequency), not a
neural model, so it needs no download and no GPU. It gives Qdrant the sparse half of hybrid
retrieval; Voyage gives the dense half. Tokens are hashed into a fixed-size index space.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


class Bm25SparseEncoder:
    def __init__(self, dim: int = 2 ** 16, k1: float = 1.5, b: float = 0.75) -> None:
        self.dim = dim
        self.k1 = k1
        self.b = b
        self._idf: dict[str, float] = {}
        self._avg_len = 0.0
        self._fitted = False

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def fit(self, corpus: list[str]) -> "Bm25SparseEncoder":
        doc_freq: dict[str, int] = {}
        total_len = 0
        for text in corpus:
            tokens = tokenize(text)
            total_len += len(tokens)
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        n_docs = max(len(corpus), 1)
        self._avg_len = total_len / n_docs
        self._idf = {
            token: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for token, df in doc_freq.items()
        }
        self._fitted = True
        return self

    def encode(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        freq: dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        doc_len = len(tokens) or 1
        avg_len = self._avg_len or doc_len
        default_idf = math.log(2)  # unseen term gets a small positive weight
        weights: dict[int, float] = {}
        for token, tf in freq.items():
            idf = self._idf.get(token, default_idf)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            score = idf * (tf * (self.k1 + 1)) / denom
            bucket = self._bucket(token)
            weights[bucket] = weights.get(bucket, 0.0) + score
        indices = sorted(weights)
        return SparseVector(indices=indices, values=[weights[i] for i in indices])
