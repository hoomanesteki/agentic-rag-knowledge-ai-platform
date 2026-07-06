"""The router is the orchestrator's cheap-first decision. The deterministic layers (0 and 1) are
pinned here offline; the Layer 2 small-model tie-break is exercised in integration, not unit tests,
so these run with no model and no network."""
from rag.router import LANES, route


def test_explicit_human_request_routes_to_escalation_at_layer_0():
    for q in ("can I talk to a human", "please connect me with a representative",
              "I want to speak to someone", "get me a real person"):
        d = route(q)
        assert d.lane == "escalation" and d.layer == 0, q


def test_escalation_does_not_fire_on_a_plain_product_mention_of_person():
    d = route("I need a jacket for a tall person")
    assert d.lane != "escalation"


def test_complaint_intent_routes_to_complaint_at_layer_1():
    d = route("you charged me twice and I am furious")
    assert d.lane == "complaint" and d.layer == 1


def test_own_account_question_routes_to_care():
    d = route("where is my order")
    assert d.lane == "care" and d.layer == 1


def test_recommendation_request_routes_to_stylist():
    d = route("can you recommend a gift for my mum")
    assert d.lane == "stylist" and d.layer == 1


def test_two_intents_become_a_complaint_first_multitask_plan():
    d = route("my order never arrived, and can you suggest a replacement")
    assert d.lane == "complaint"  # complaint leads
    assert d.tasks and d.tasks[0] == "complaint" and "stylist" in d.tasks


def test_unmatched_general_question_falls_back_to_answers():
    d = route("what is your return window")
    assert d.lane == "answers"


def test_every_decision_names_a_known_lane():
    for q in ("hi", "", "where is my order", "recommend a coat", "talk to a human"):
        assert route(q).lane in LANES


def test_route_never_calls_a_model_when_a_layer_decides():
    # a stub that would raise if used proves the deterministic path pays nothing
    class Boom:
        def generate(self, *a, **k):
            raise AssertionError("router used the model on a deterministic turn")
    for q in ("where is my order", "recommend a coat", "talk to a human"):
        assert route(q, small_llm=Boom()).lane in LANES
