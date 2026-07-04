"""M5.2 entity linking: link doc mentions to canonical graph entities, offline on the fakes.

The real pass calls Groq; here a fake LLM echoes shortlisted candidate ids with a chosen
confidence, so the linking logic (shortlist, threshold, review list, edge creation) is tested
without a model. Proves it is domain agnostic and that low-confidence links are queued, not
dropped or silently linked.
"""
import json
import re

from adapters.base import LLMResult
from adapters.factory import make_graph
from data.lakehouse import build_lakehouse
from knowledge.entity_linking import link_mentions
from knowledge.graph_loader import load_graph


class FakeLinkLLM:
    """Returns the first shortlisted candidate (parsed from the prompt) at a fixed confidence."""

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.calls = 0

    def generate(self, prompt, *, system=None, max_tokens=512):
        self.calls += 1
        ids = re.findall(r"id=(\S+)", prompt)
        payload = json.dumps([{"id": ids[0], "confidence": self.confidence}]) if ids else "[]"
        return LLMResult(text=payload, prompt_tokens=1, completion_tokens=1, model="fake")


class RaisingLLM:
    def generate(self, prompt, *, system=None, max_tokens=512):
        raise AssertionError("LLM must not be called when there is nothing to link")


def _graph_for(domain, tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse(domain, db)
    store = make_graph("memory")
    load_graph(domain, db, store)
    return store


def test_high_confidence_creates_mention_edge(tmp_path):
    store = _graph_for("apparel_ecommerce", tmp_path)
    report = link_mentions("apparel_ecommerce", store, FakeLinkLLM(0.9))
    assert report.linked > 0 and not report.review_list
    # R001 is about a Flow Legging, so it must carry a MENTIONS edge to a Product.
    mentions = store.neighbors("Review", "id", "R001", edge_type="MENTIONS", direction="out")
    assert mentions and mentions[0].node.label == "Product"


def test_low_confidence_goes_to_review_list_not_graph(tmp_path):
    store = _graph_for("apparel_ecommerce", tmp_path)
    report = link_mentions("apparel_ecommerce", store, FakeLinkLLM(0.3), threshold=0.6)
    assert report.linked == 0
    assert report.review_list  # queued for a human instead of dropped or linked
    assert not store.neighbors("Review", "id", "R001", edge_type="MENTIONS", direction="out")


def test_unparseable_llm_output_is_queued_not_dropped(tmp_path):
    store = _graph_for("apparel_ecommerce", tmp_path)

    class JunkLLM:
        def generate(self, prompt, *, system=None, max_tokens=512):
            return LLMResult(text="not json", prompt_tokens=1, completion_tokens=1, model="fake")

    report = link_mentions("apparel_ecommerce", store, JunkLLM())
    assert report.linked == 0
    # a doc we could not resolve must reach the review list, never vanish silently
    assert report.review_list
    assert all(r["reason"] == "unparsed_llm_output" for r in report.review_list)


def test_empty_list_is_a_real_no_match_not_queued(tmp_path):
    store = _graph_for("apparel_ecommerce", tmp_path)

    class NoMatchLLM:
        def generate(self, prompt, *, system=None, max_tokens=512):
            return LLMResult(text="[]", prompt_tokens=1, completion_tokens=1, model="fake")

    report = link_mentions("apparel_ecommerce", store, NoMatchLLM())
    assert report.linked == 0 and not report.review_list  # [] is a valid "no mention"


def test_string_confidence_is_accepted(tmp_path):
    store = _graph_for("apparel_ecommerce", tmp_path)

    class StringConfLLM:
        def generate(self, prompt, *, system=None, max_tokens=512):
            ids = re.findall(r"id=(\S+)", prompt)
            payload = json.dumps([{"id": ids[0], "confidence": "0.9"}]) if ids else "[]"
            return LLMResult(text=payload, prompt_tokens=1, completion_tokens=1, model="fake")

    report = link_mentions("apparel_ecommerce", store, StringConfLLM())
    assert report.linked > 0  # "0.9" as a string must still count


def test_no_candidates_means_no_llm_call(tmp_path):
    # An empty graph has no target nodes, so nothing can be shortlisted and the LLM is untouched.
    store = make_graph("memory")
    build_lakehouse("apparel_ecommerce", str(tmp_path / "lh.duckdb"))
    report = link_mentions("apparel_ecommerce", store, RaisingLLM())
    assert report.linked == 0
