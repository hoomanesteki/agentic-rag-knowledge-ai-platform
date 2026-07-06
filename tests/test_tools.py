"""The lane tools are thin adapters over the specialists, so the tests pin the safe degenerate
paths (missing deps return a not-found Finding, never an exception) and the one new helper."""
from rag import tools


def test_governed_metric_without_a_resolver_is_not_found():
    f = tools.get_governed_metric("what is the average price", llm=None, metric_resolver=None)
    assert f.found is False and f.kind == "metric"


def test_graph_facts_without_a_graph_is_not_found():
    f = tools.graph_facts("which supplier makes it", graph_retriever=None)
    assert f.found is False and f.kind == "graph"


def test_get_profile_shapes_notes_into_a_brief():
    assert tools.get_profile(None) == {}
    assert tools.get_profile("   ") == {}
    assert tools.get_profile("shops for her mum, likes yoga") == {
        "brief": "shops for her mum, likes yoga"}


def test_tools_expose_no_privileged_order_fetch():
    # order lookups must stay on the gated retrieve() path; there is no direct customer-orders tool
    assert not any("order" in name or "customer" in name for name in tools.__all__)
