"""The caching and retry wrappers cut repeated metered calls and ride out transient failures,
which is what keeps the demo working under a low free-tier rate limit."""
from api.resilience import CachingEmbedder, CachingReranker, ResilientLLM


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
