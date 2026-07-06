"""A comparison or superlative follow-up ("which is warmest", "the cheaper one") must be answered
over the items the assistant JUST recommended, not a fresh, different set. These lock in the
detector and the prior-product extraction that anchor retrieval on the shown products."""
from pipeline.answer import _COMPARE_FOLLOWUP, _catalog_names, _prior_products

DOMAIN = "apparel_ecommerce"


def test_compare_followup_detects_superlatives_and_comparisons():
    for q in ["which is warmest?", "the cheaper one", "which of those is best",
              "compare those two", "the softest one", "which one is lightest",
              "the warmest of these"]:
        assert _COMPARE_FOLLOWUP.search(q), q


def test_compare_followup_ignores_plain_requests():
    for q in ["show me a warm jacket", "I want the blue hoodie", "a gift for my dad",
              "do you have leggings", "something cozy for the house"]:
        assert not _COMPARE_FOLLOWUP.search(q), q


def test_catalog_names_load_from_the_pack():
    names = _catalog_names(DOMAIN)
    assert names and all(isinstance(n, str) and n for n in names)
    # longest-first, so a full name matches before a shorter one it contains
    assert len(names[0]) >= len(names[-1])


def test_prior_products_extracts_named_items_from_the_last_bot_turn():
    names = _catalog_names(DOMAIN)
    a, b = names[0], names[1]  # two real, distinct catalog product names
    history = [{"role": "user", "content": "a gift for my dad who hikes"},
               {"role": "assistant", "content": "I'd suggest the {} or the {}.".format(a, b)},
               {"role": "user", "content": "which is warmest"}]
    got = _prior_products(history, DOMAIN)
    assert a in got and b in got


def test_prior_products_empty_without_a_prior_bot_turn():
    assert _prior_products(None, DOMAIN) == []
    assert _prior_products([{"role": "user", "content": "which is warmest"}], DOMAIN) == []
