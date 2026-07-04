"""Resilience helpers for hosted-API calls: retry transient failures, and a wrapper that
makes the embedder retry. The chat endpoint degrades (rather than 500s) when these are
exhausted. Full retry policy on the LLM stream is a mid-stream problem left to M6."""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

_log = logging.getLogger("skein.resilience")
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


class ResilientStore:
    """Wraps a hybrid vector store so the read path (hybrid_search) retries transient failures.
    Retrieval has no fallback, so a single 429/503 from the store would otherwise lose the whole
    turn; a couple of retries ride out a blip. Writes (ensure_collection/upsert) are retried too, so
    ingestion survives a hiccup."""

    def __init__(self, inner, attempts: int = 3) -> None:
        self.inner = inner
        self.attempts = attempts

    def hybrid_search(self, *args, **kwargs):
        return with_retry(lambda: self.inner.hybrid_search(*args, **kwargs), attempts=self.attempts)

    def search(self, *args, **kwargs):
        return with_retry(lambda: self.inner.search(*args, **kwargs), attempts=self.attempts)

    def ensure_collection(self, *args, **kwargs):
        return with_retry(lambda: self.inner.ensure_collection(*args, **kwargs),
                          attempts=self.attempts)

    def upsert(self, *args, **kwargs):
        return with_retry(lambda: self.inner.upsert(*args, **kwargs), attempts=self.attempts)

    def __getattr__(self, name):  # delegate any other attribute/method to the wrapped store
        return getattr(self.inner, name)


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
        need: dict[str, list[float] | None] = {}  # unique cache-miss texts, deduped
        for i, t in enumerate(texts):
            hit = self._cache.get((input_type, t))
            if hit is not None:
                out[i] = hit
            elif t not in need:
                need[t] = None
        if need:
            uniq = list(need)
            vecs = self.inner.embed(uniq, input_type=input_type)  # each miss embedded once
            for t, v in zip(uniq, vecs):
                key = (input_type, t)
                if key not in self._cache:
                    self._order.append(key)
                self._cache[key] = v
                need[t] = v
            while len(self._order) > self.maxsize:  # evict oldest
                self._cache.pop(self._order.pop(0), None)
        for i, t in enumerate(texts):  # fill misses, preserving input order
            if out[i] is None:
                out[i] = need[t]
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


class ResilientReranker:
    """Retries transient rerank failures; if they persist, returns the identity ranking so the
    caller keeps the pre-rerank order and still answers, instead of the whole turn degrading over a
    rerank blip. Rerank is a precision boost, not a hard dependency, so falling back is safe."""

    def __init__(self, inner, attempts: int = 2, base_delay: float = 0.4) -> None:
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay

    def rerank(self, query: str, documents: list[str], top_n: int = 8) -> list[tuple[int, float]]:
        try:
            return with_retry(lambda: self.inner.rerank(query, documents, top_n=top_n),
                              attempts=self.attempts, base_delay=self.base_delay)
        except RuntimeError as exc:
            if is_transient(exc):  # keep the answer, just without the rerank re-ordering
                return [(i, 0.0) for i in range(min(top_n, len(documents)))]
            raise


class ResilientLLM:
    """Wraps an LLM client so generate() retries transient failures (Groq 429s, connection
    resets), instead of the chat immediately degrading on the first hiccup. stream() and other
    attributes pass through untouched (mid-stream retry is a separate problem).

    If a fallback client is given, a primary failure that survives the retries falls back to it (a
    cheaper, faster secondary model), so a bad minute on the main model degrades quality rather than
    losing the turn."""

    def __init__(self, inner, attempts: int = 3, base_delay: float = 0.5, fallback=None) -> None:
        self.inner = inner
        self.attempts = attempts
        self.base_delay = base_delay
        self.fallback = fallback

    def __getattr__(self, name):  # model, stream, and anything else defer to the wrapped client
        return getattr(self.inner, name)

    def generate(self, *args, **kwargs):
        try:
            return with_retry(lambda: self.inner.generate(*args, **kwargs),
                              attempts=self.attempts, base_delay=self.base_delay)
        except Exception:
            if self.fallback is None:
                raise
            _log.warning("primary LLM failed after retries, falling back to the secondary model")
            return self.fallback.generate(*args, **kwargs)
