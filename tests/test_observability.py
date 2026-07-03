"""T3 Langfuse tracing seam. In tests Langfuse is off (conftest clears the keys), so every hook is
a no-op passthrough: the decorator returns the function unchanged, the span is a null context, and
the update/flush calls do nothing. This proves tracing never changes engine behavior offline."""
from adapters import observability


def test_disabled_in_tests():
    assert observability.enabled() is False


def test_observe_is_a_transparent_passthrough_when_disabled():
    def double(x):
        return x * 2

    assert observability.observe(as_type="generation")(double) is double
    assert observability.observe(double) is double  # bare form


def test_decorated_function_still_returns_its_value():
    @observability.observe(as_type="generation", name="unit")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_request_span_is_a_usable_context_manager():
    with observability.request_span("chat", input="hello"):
        pass  # must not raise and must be a real context manager


def test_updates_and_flush_are_harmless_noops():
    observability.update_generation(model="x", input="p", output="o",
                                    usage_details={"input": 1, "output": 2})
    observability.update_span(output="o")
    observability.flush()  # none of these raise when disabled
