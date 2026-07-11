"""A streamed turn must be metered, not zeroed. The stream fills a usage dict (Groq's
stream_options.include_usage), and the cost function prices it, billing cached input cheaper so the
prompt-caching win shows up in the trace once the workhorse is a caching model."""
from adapters.fakes import EchoLLM
from pipeline.answer import _estimate_cost


def test_stream_fills_usage_out():
    usage: dict = {}
    text = "".join(EchoLLM().stream("two words here", usage_out=usage))
    assert text
    assert usage["prompt_tokens"] == 3 and usage["completion_tokens"] == 1
    assert usage["model"]  # so the trace can attribute cost to the real model/tier


def test_stream_without_usage_out_is_backward_compatible():
    # a caller that does not want metering still gets a plain token stream
    assert "".join(EchoLLM().stream("hello there")) != ""


def test_estimate_cost_bills_cached_input_cheaper():
    model = "llama-3.3-70b-versatile"
    # a turn with 500 fresh + 1000 cached input costs LESS than 1500 all-fresh, because Groq bills a
    # cache hit at 0.5x input. The streamed path passes fresh = total - cached, no double-count.
    all_fresh = _estimate_cost(model, 1500, 250)
    with_cache = _estimate_cost(model, 500, 250, cache_read_tokens=1000)
    assert all_fresh is not None and with_cache is not None
    assert with_cache < all_fresh
    assert _estimate_cost("unknown-model", 100, 100) is None  # never pretend an unknown cost is 0
