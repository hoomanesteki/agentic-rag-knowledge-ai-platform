"""The caching and retry wrappers cut repeated metered calls and ride out transient failures,
which is what keeps the demo working under a low free-tier rate limit."""
import pytest

from api.resilience import (
    CachingEmbedder,
    CachingReranker,
    ResilientLLM,
    ResilientReranker,
)


class _CountingEmbedder:
    def __init__(self):
        self.calls = 0
        self.dim = 3

    def embed(self, texts, input_type="document"):
        self.calls += 1
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class _CountingReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents, top_n=8):
        self.calls += 1
        return [(i, 1.0 / (i + 1)) for i in range(min(top_n, len(documents)))]


def test_embedder_cache_serves_repeats_without_recalling():
    inner = _CountingEmbedder()
    emb = CachingEmbedder(inner)
    a = emb.embed(["how much is the legging"], input_type="query")
    b = emb.embed(["how much is the legging"], input_type="query")
    assert a == b
    assert inner.calls == 1  # the second identical query hit the cache


def test_embedder_cache_only_calls_for_the_missing_texts():
    inner = _CountingEmbedder()
    emb = CachingEmbedder(inner)
    emb.embed(["one"], input_type="query")
    out = emb.embed(["one", "two"], input_type="query")  # only "two" is new
    assert len(out) == 2
    assert inner.calls == 2  # one call for "one", one for the "two" miss


def test_embedder_cache_separates_input_types():
    inner = _CountingEmbedder()
    emb = CachingEmbedder(inner)
    emb.embed(["x"], input_type="query")
    emb.embed(["x"], input_type="document")  # same text, different type is a distinct key
    assert inner.calls == 2


def test_reranker_cache_serves_repeats():
    inner = _CountingReranker()
    rr = CachingReranker(inner)
    docs = ["a", "b", "c"]
    first = rr.rerank("q", docs, top_n=2)
    second = rr.rerank("q", docs, top_n=2)
    assert first == second
    assert inner.calls == 1


class _FlakyLLM:
    """Fails with a transient 429 a set number of times, then succeeds."""

    def __init__(self, fails):
        self.fails = fails
        self.model = "test-model"

    def generate(self, prompt, **kwargs):
        if self.fails > 0:
            self.fails -= 1
            raise RuntimeError("groq -> HTTP 429: rate limited")
        return "ok"


def test_llm_retry_recovers_from_transient_429():
    llm = ResilientLLM(_FlakyLLM(fails=2), base_delay=0)
    assert llm.generate("hi") == "ok"


def test_llm_passthrough_exposes_inner_attributes():
    llm = ResilientLLM(_FlakyLLM(fails=0))
    assert llm.model == "test-model"  # non-generate attributes defer to the wrapped client


def test_embedder_cache_dedupes_duplicate_texts_in_one_call():
    inner = _CountingEmbedder()
    emb = CachingEmbedder(inner)
    out = emb.embed(["same", "same", "other"], input_type="query")
    assert out[0] == out[1]  # duplicates share the one vector, in input order
    assert inner.calls == 1  # a single batched call...
    # ...and that call embedded the two unique texts only, not the duplicate


class _FlakyReranker:
    def __init__(self, fails, error="rerank -> HTTP 429: rate limited"):
        self.fails = fails
        self.error = error
        self.calls = 0

    def rerank(self, query, documents, top_n=8):
        self.calls += 1
        if self.fails > 0:
            self.fails -= 1
            raise RuntimeError(self.error)
        return [(len(documents) - 1, 0.9)]  # last doc wins, to prove passthrough


def test_reranker_retries_then_passes_through():
    inner = _FlakyReranker(fails=1)
    rr = ResilientReranker(inner, base_delay=0)
    assert rr.rerank("q", ["a", "b", "c"], top_n=2) == [(2, 0.9)]


def test_reranker_falls_back_to_identity_on_persistent_transient():
    inner = _FlakyReranker(fails=99)  # never succeeds
    rr = ResilientReranker(inner, attempts=2, base_delay=0)
    # identity order over the first top_n, so the answer still returns un-reranked
    assert rr.rerank("q", ["a", "b", "c", "d"], top_n=2) == [(0, 0.0), (1, 0.0)]


def test_reranker_reraises_non_transient():
    inner = _FlakyReranker(fails=99, error="rerank -> HTTP 400: bad request")
    rr = ResilientReranker(inner, attempts=2, base_delay=0)
    with pytest.raises(RuntimeError):
        rr.rerank("q", ["a"], top_n=1)
