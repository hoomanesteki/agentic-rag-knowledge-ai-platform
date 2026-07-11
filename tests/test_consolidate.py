"""Consolidation proposes a knowledge pack for a human to approve; it never mutates memory itself,
and candidate chunks come only from human-verified answers. These pin the proposal logic offline."""
from mlops.consolidate import propose_pack


def test_candidate_chunks_come_only_from_human_verified_answers():
    closed = [
        {"question": "how do I export my data", "answer": "Settings, Data, Export.", "lang": "en"},
        {"question": "how do I export my data", "answer": "dup", "lang": "en"},  # de-duplicated
    ]
    pack = propose_pack([], [], closed)
    assert pack["counts"]["chunks"] == 1  # deduped by question
    assert pack["candidate_chunks"][0]["source"] == "hitl-consolidated"
    assert pack["candidate_eval_rows"][0]["question"] == "how do I export my data"


def test_gaps_surface_abstains_and_thumbs_down_not_answers():
    traces = [
        {"query": "do you sell titanium widgets", "tier": "abstain", "message_id": "m1"},
        {"query": "do you sell titanium widgets", "tier": "abstain", "message_id": "m2"},
        {"query": "what is your refund policy", "tier": "auto", "message_id": "m3"},
    ]
    feedback = [{"message_id": "m3", "verdict": "down"}]  # a thumbs-down on an answered turn
    pack = propose_pack(traces, feedback, [])
    gaps = {g["query"]: g["count"] for g in pack["knowledge_gaps"]}
    assert gaps.get("do you sell titanium widgets") == 2  # two abstains cluster
    assert gaps.get("what is your refund policy") == 1    # the thumbs-down turn is a gap
    assert pack["candidate_chunks"] == []                 # gaps are not auto-answered


def test_frequent_repeats_are_flagged_over_the_threshold():
    traces = [{"query": "where is my order", "tier": "auto", "message_id": "m{}".format(i)}
              for i in range(4)]
    pack = propose_pack(traces, [], [], min_repeats=3)
    assert pack["frequent_queries"][0]["query"] == "where is my order"
    assert pack["frequent_queries"][0]["count"] == 4


def test_proposal_says_it_is_human_gated():
    pack = propose_pack([], [], [])
    assert any("PROPOSED only" in n for n in pack["notes"])
    assert any("never an LLM rewrite" in n for n in pack["notes"])
