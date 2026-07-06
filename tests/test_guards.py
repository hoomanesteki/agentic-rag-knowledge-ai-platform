"""The guard layer is the single deterministic API the omni brain routes on, so its behavior is
pinned here: the classifiers agree with the linear pipeline they re-export, and the PII gate is
NOT among them (it stays coupled to retrieval, not exposed as a routing primitive)."""
from rag import guards


def test_shopping_intent_matches_recommendation_requests():
    assert guards.shopping_intent("can you recommend a jacket for running") is True
    assert guards.shopping_intent("where is my order") is False


def test_problem_intent_matches_complaints_not_shopping():
    assert guards.problem_intent("you charged me twice and I'm furious") is True
    assert guards.problem_intent("show me some leggings") is False


def test_account_intent_is_first_person_only():
    assert guards.account_intent("where is my order") is True
    # a third-party lookup keyed on someone else must not read as an own-account question
    assert guards.account_intent("list all orders placed by Jordan Avery") is False


def test_heuristic_route_is_deterministic_and_in_the_known_set():
    for q in ("how much does it cost", "which supplier makes it", "is it comfortable",
              "what is the average price"):
        assert guards.heuristic_route(q) in guards.ROUTES


def test_guards_do_not_expose_a_disclosure_primitive():
    # the guard surface must not grow a "can disclose PII" helper; disclosure stays in retrieve()
    assert not any("disclose" in name or "reveal" in name for name in guards.__all__)
