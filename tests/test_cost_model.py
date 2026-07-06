"""The cost model is a reproducible, deterministic estimate. These tests pin the ordering that the
business case rests on (cheap tiers below frontier below human) so a bad edit to an assumption or a
price cannot silently invert the story."""
from mlops.cost_model import build, text_turn_cost, voice_turn_cost


def test_tier_cost_ordering_is_sane():
    pt = build()["per_turn"]
    assert pt["text_8b"] < pt["text_70b"] < pt["text_sonnet5"] < pt["text_opus"] < pt["human_agent"]


def test_voice_costs_more_than_text_and_routing_is_near_free():
    assert voice_turn_cost()["total"] > text_turn_cost()["total"]  # speech in and out add cost
    b = text_turn_cost()
    assert b["routing"] < b["generation"]  # routing is essentially free next to generation


def test_human_agent_is_orders_of_magnitude_over_the_ai():
    assert build()["ratios_vs_text_70b"]["human_agent"] > 100
