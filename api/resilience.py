"""Resilience helpers for hosted-API calls: retry transient failures, and a wrapper that
makes the embedder retry. The chat endpoint degrades (rather than 500s) when these are
exhausted. Full retry policy on the LLM stream is a mid-stream problem left to M6."""
from __future__ import annotations

import re
import time
from collections.abc import Callable

_HTTP_CODE = re.compile(r"HTTP (\d{3})")


def is_transient(err: Exception) -> bool:
    """Transient = worth retrying/degrading: HTTP 429 or 5xx, or a connection-level failure.
    Classify by the status code in the message (not arbitrary body text) to avoid treating a
    permanent 4xx whose body happens to say 'failed' as transient."""
    text = str(err)
    match = _HTTP_CODE.search(text)
    if match:
        code = int(match.group(1))
        return code == 429 or 500 <= code < 600
    return "failed:" in text  # URLError / normalized mid-stream connection error


def with_retry(fn: Callable, *, attempts: int = 3, base_delay: float = 0.2,
               sleep: Callable = time.sleep):
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except RuntimeError as exc:
            last = exc
            if is_transient(exc) and i < attempts - 1:
                sleep(base_delay * (2 ** i))
                continue
            raise
    raise last  # pragma: no cover


class ResilientEmbedder:
    """Wraps an embedder so embed() retries transient failures."""

    def __init__(self, inner, attempts: int = 3) -> None:
        self.inner = inner
        self.attempts = attempts

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        return with_retry(lambda: self.inner.embed(texts, input_type=input_type),
                          attempts=self.attempts)


class CachingEmbedder:
    """Wraps an embedder with a small per-text LRU cache. At runtime the app only embeds queries,
    and demos reuse the same starter prompts, so caching turns repeated questions into zero API
    calls. That matters directly on a metered embedder (Voyage's free tier is 3 requests/minute):
    clicking a suggestion twice, or asking the same thing, no longer spends the budget."""

    def __init__(self, inner, maxsize: int = 512) -> None:
        self.inner = inner
        self.maxsize = maxsize
        self._cache: dict[tuple[str, str], list[float]] = {}
        self._order: list[tuple[str, str]] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        miss_idx, miss_txt = [], []
        for i, t in enumerate(texts):
            hit = self._cache.get((input_type, t))
            if hit is not None:
                out[i] = hit
            else:
                miss_idx.append(i)
                miss_txt.append(t)
        if miss_txt:
            vecs = self.inner.embed(miss_txt, input_type=input_type)
            for j, i in enumerate(miss_idx):
                key = (input_type, texts[i])
                self._cache[key] = vecs[j]
                self._order.append(key)
                out[i] = vecs[j]
            while len(self._order) > self.maxsize:  # evict oldest
                self._cache.pop(self._order.pop(0), None)
        return [v for v in out if v is not None]


class CachingReranker:
    """Caches rerank results by (query, documents, top_n). A repeated identical query retrieves the
    same candidates, so the rerank is a cache hit and spends no metered call. This is the second
    Voyage call per question (after the query embed), so caching it too is what makes clicking the
    same suggestion repeatedly cost nothing."""

    def __init__(self, inner, maxsize: int = 256) -> None:
        self.inner = inner
        self.maxsize = maxsize
        self._cache: dict[tuple, list[tuple[int, float]]] = {}
        self._order: list[tuple] = []

    def rerank(self, query: str, documents: list[str], top_n: int = 8) -> list[tuple[int, float]]:
        key = (query, tuple(documents), top_n)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self.inner.rerank(query, documents, top_n=top_n)
        self._cache[key] = out
        self._order.append(key)
        while len(self._order) > self.maxsize:
            self._cache.pop(self._order.pop(0), None)
        return out


class ResilientLLM:
    """Wraps an LLM client so generate() retries transient failures (Groq 429s, connection
    resets), instead of the chat immediately degrading on the first hiccup. stream() and other
    attributes pass through untouched (mid-stream retry is a separate problem)."""

    def __init__(self, inner, attempts: int = 3, base_delay: float = 0.5) -> None:
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay

    def __getattr__(self, name):  # model, stream, and anything else defer to the wrapped client
        return getattr(self.inner, name)

    def generate(self, *args, **kwargs):
        return with_retry(lambda: self.inner.generate(*args, **kwargs),
                          attempts=self.attempts, base_delay=self.base_delay)
