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
