"""M4.3 metric routing: LLM slot-fill to the governed metric layer, and pipeline evidence."""
from adapters.base import LLMResult
from adapters.factory import make_embedder, make_store
from data.lakehouse import build_lakehouse
from data.metrics import MetricResolver
from pipeline.answer import answer_question, stream_answer
from retrieval.metric_router import route_metric
from retrieval.sparse import SparseEncoder

_RATE_Q = "what is the return rate for size M"


class JsonLLM:
    """Returns a fixed slot-fill JSON for any prompt."""

    model = "fake"

    def __init__(self, payload):
        self.payload = payload

    def generate(self, prompt, *, system=None, max_tokens=512):
        return LLMResult(text=self.payload, prompt_tokens=1, completion_tokens=1, model="fake")


class SmartFakeLLM:
    """Slot-fills when asked about metrics, otherwise answers citing the first source."""

    model = "fake"

    def _text(self, prompt):
        if "Available metrics:" in prompt:
            return '{"metric": "return_rate_by_size", "params": {"size": "M"}}'
        return "The return rate for size M is 0.5 [1]."

    def generate(self, prompt, *, system=None, max_tokens=512):
        return LLMResult(text=self._text(prompt), prompt_tokens=1, completion_tokens=1,
                         model="fake")

    def stream(self, prompt, *, system=None, max_tokens=512, usage_out=None):
        yield self._text(prompt)


def _resolver(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    return MetricResolver("apparel_ecommerce", db)


def test_route_metric_resolves_from_json(tmp_path):
    resolver = _resolver(tmp_path)
    llm = JsonLLM('{"metric": "return_rate_by_size", "params": {"size": "M"}}')
    result = route_metric(_RATE_Q, llm, resolver)
    assert result is not None
    assert result.name == "return_rate_by_size"
    assert 0.5 in result.rows[0]


def test_route_metric_handles_code_fenced_json(tmp_path):
    resolver = _resolver(tmp_path)
    fenced = '```json\n{"metric": "return_rate_by_size", "params": {"size": "M"}}\n```'
    assert route_metric(_RATE_Q, JsonLLM(fenced), resolver) is not None


def test_route_metric_none_when_model_declines_or_malformed(tmp_path):
    resolver = _resolver(tmp_path)
    assert route_metric(_RATE_Q, JsonLLM('{"metric": null, "params": {}}'), resolver) is None
    assert route_metric(_RATE_Q, JsonLLM("not json at all"), resolver) is None
    # a non-string metric name must not crash (hashing an unhashable value)
    assert route_metric(_RATE_Q, JsonLLM('{"metric": ["x"], "params": {}}'), resolver) is None


def test_route_metric_pregate_skips_llm_for_non_metric(tmp_path):
    resolver = _resolver(tmp_path)

    class CountingLLM:
        model = "fake"
        calls = 0

        def generate(self, prompt, *, system=None, max_tokens=512):
            CountingLLM.calls += 1
            return LLMResult(text="{}", prompt_tokens=1, completion_tokens=1, model="fake")

    assert route_metric("does the legging run small", CountingLLM(), resolver) is None
    assert CountingLLM.calls == 0  # pre-gate avoided the slot-fill call entirely


def test_route_metric_none_when_lakehouse_missing(tmp_path):
    resolver = MetricResolver("apparel_ecommerce", str(tmp_path / "never-built.duckdb"))
    llm = JsonLLM('{"metric": "return_rate_by_size", "params": {"size": "M"}}')
    assert route_metric(_RATE_Q, llm, resolver) is None


def test_pipeline_uses_metric_block_and_cites_it(tmp_path):
    resolver = _resolver(tmp_path)
    embedder = make_embedder("fake")
    store = make_store("memory")  # empty vector store: the answer must come from the metric
    result = answer_question("what is the return rate for size M", embedder=embedder, store=store,
                             llm=SmartFakeLLM(), metric_resolver=resolver,
                             trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "auto"                       # metric evidence -> not abstain
    assert result.trace["metric"] is True
    metric_ctx = result.contexts[0]
    assert metric_ctx["doc_type"] == "metric"
    assert "0.5" in metric_ctx["text"]
    assert result.citations[0]["id"].startswith("metric:")  # the metric block is cited


def test_streaming_path_uses_metric_block(tmp_path):
    resolver = _resolver(tmp_path)
    events = list(stream_answer(_RATE_Q, embedder=make_embedder("fake"),
                                store=make_store("memory"), llm=SmartFakeLLM(),
                                metric_resolver=resolver, trace_path=str(tmp_path / "t.jsonl")))
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] == "auto"
    assert any(c["id"].startswith("metric:") for c in final["citations"])


def test_value_less_metric_does_not_suppress_abstain(tmp_path):
    # A return rate for a size we do not sell resolves to zero rows: it must not be treated as
    # authoritative, so with no vector evidence the pipeline abstains instead of answering.
    resolver = _resolver(tmp_path)
    llm = JsonLLM('{"metric": "return_rate_by_size", "params": {"size": "ZZZ"}}')
    result = answer_question(_RATE_Q, embedder=make_embedder("fake"), store=make_store("memory"),
                             llm=llm, metric_resolver=resolver,
                             trace_path=str(tmp_path / "t.jsonl"))
    assert result.trace["metric"] is False
    assert result.tier == "abstain"


def test_pipeline_without_resolver_is_unchanged(tmp_path):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    docs = [{"id": "R1", "text": "the flow legging runs small", "payload": {"doc_type": "review"}}]
    dense = embedder.embed([d["text"] for d in docs])
    store.upsert([{**docs[0], "dense": dense[0],
                   "sparse": {"indices": encoder.encode(docs[0]["text"]).indices,
                              "values": encoder.encode(docs[0]["text"]).values}}])
    result = answer_question("does the legging run small", embedder=embedder, store=store,
                             llm=SmartFakeLLM(), trace_path=str(tmp_path / "t.jsonl"))
    assert result.trace["metric"] is False
