"""A tiny in-memory token-bucket rate limiter, keyed per client. Good enough for the demo;
swap for Redis behind the same interface at scale."""
from __future__ import annotations

import time

_UNITS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}


def parse_rate(spec: str) -> tuple[int, float]:
    """'30/minute' -> (30, 60.0)."""
    count, _, unit = spec.partition("/")
    return int(count), _UNITS.get(unit.strip().lower(), 60.0)


class RateLimiter:
    # Above this many tracked clients, drop the idle ones so a wide botnet cannot grow the dict
    # without bound. A bucket refilled to full capacity is identical to a fresh one, so evicting it
    # changes no decision.
    _MAX_BUCKETS = 20_000

    def __init__(self, spec: str = "30/minute") -> None:
        self.capacity, window = parse_rate(spec)
        self.refill_per_sec = self.capacity / window if window else float(self.capacity)
        self._buckets: dict[str, tuple[float, float]] = {}

    def _current_tokens(self, tokens: float, last: float, now: float) -> float:
        return min(self.capacity, tokens + (now - last) * self.refill_per_sec)

    def _prune(self, now: float) -> None:
        self._buckets = {
            k: (t, ts) for k, (t, ts) in self._buckets.items()
            if self._current_tokens(t, ts, now) < self.capacity}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if len(self._buckets) > self._MAX_BUCKETS:
            self._prune(now)
        tokens, last = self._buckets.get(key, (float(self.capacity), now))
        tokens = self._current_tokens(tokens, last, now)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1.0, now)
        return True
