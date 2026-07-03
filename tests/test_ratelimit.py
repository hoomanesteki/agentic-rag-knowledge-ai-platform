"""The token-bucket limiter: refill, per-key isolation, and bounded memory under many clients."""
from api.ratelimit import RateLimiter, parse_rate


def test_parse_rate():
    assert parse_rate("30/minute") == (30, 60.0)
    assert parse_rate("5/second") == (5, 1.0)


def test_allows_up_to_capacity_then_blocks():
    limiter = RateLimiter("3/minute")
    assert [limiter.allow("a", now=0.0) for _ in range(3)] == [True, True, True]
    assert limiter.allow("a", now=0.0) is False  # bucket empty


def test_refills_over_time():
    limiter = RateLimiter("60/minute")  # one token per second
    for _ in range(60):
        limiter.allow("a", now=0.0)
    assert limiter.allow("a", now=0.0) is False
    assert limiter.allow("a", now=2.0) is True  # ~2 tokens refilled after 2s


def test_keys_are_isolated():
    limiter = RateLimiter("1/minute")
    assert limiter.allow("a", now=0.0) is True
    assert limiter.allow("b", now=0.0) is True  # a different client has its own bucket
    assert limiter.allow("a", now=0.0) is False


def test_idle_buckets_are_pruned_so_memory_is_bounded():
    limiter = RateLimiter("30/minute")
    limiter._MAX_BUCKETS = 5
    # one active client (bucket drained now) plus many idle clients seen long ago
    limiter.allow("active", now=1000.0)
    for i in range(10):
        limiter._buckets["idle{}".format(i)] = (30.0, 0.0)  # full and stale
    limiter.allow("trigger", now=1000.0)  # crosses the threshold and prunes
    assert "active" in limiter._buckets  # the drained active client is kept
    assert not any(k.startswith("idle") for k in limiter._buckets)  # refilled idles dropped
