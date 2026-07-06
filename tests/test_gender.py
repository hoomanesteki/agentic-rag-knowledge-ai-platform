"""Gender is a hard facet: when a shopper states it (directly or via a gendered relative noun), the
assistant must never surface an opposite-gender SKU. These lock in the recipient-noun precedence
(a gift "for her husband" is for a man), the natural direct phrasings ("shorts for men"), and the
retrieval hard-filter's fallback when a product chunk carries no gender tag.
"""
import pytest

from pipeline.answer import (
    _explicit_gender,
    _gender_filter,
    _product_genders,
    _redact_contexts_by_gender,
    _sticky_gender,
)

DOMAIN = "apparel_ecommerce"


@pytest.mark.parametrize("query,expected", [
    # a gendered relative noun names the RECIPIENT and wins over the possessor's pronoun
    ("a warm jacket for her husband", "men"),
    ("something for her boyfriend", "men"),
    ("a gift for her dad", "men"),
    ("leggings for his girlfriend", "women"),
    ("a present for his wife", "women"),
    # possessive relatives without "for"
    ("my husband needs a rain shell", "men"),
    ("shopping for my sister", "women"),
    # natural direct phrasings that used to slip through as None
    ("shorts for men", "men"),
    ("a jacket for men", "men"),
    ("leggings for women", "women"),
    ("do you have leggings for women", "women"),
    ("I am a man, show me leggings", "men"),
    ("I'm a woman looking for shorts", "women"),
    # bare pronoun with no relative noun still sets the recipient's gender
    ("a jacket for her", "women"),
    ("a cap for him", "men"),
    # possessive forms
    ("men's training tops", "men"),
    ("women's yoga gear", "women"),
    # an explicit PRODUCT cue outranks an incidental opposite relative noun in the same turn
    ("men's leggings, my wife recommended them", "men"),
    ("women's jacket, my husband said to get one", "women"),
    ("shorts for men, my girlfriend loves the brand", "men"),
    # the RECIPIENT decides the section, so an opposite-gender relative beats the shopper's own
    # stated gender: a woman shopping for her husband must see the men's section, not the women's
    ("I'm a woman, get something for my husband", "men"),
    ("I am a man buying a gift for my wife", "women"),
    # a bare recipient pronoun also outranks the shopper's own stated gender
    ("I'm a woman, a gift for him", "men"),
])
def test_explicit_gender(query, expected):
    assert _explicit_gender(query) == expected


@pytest.mark.parametrize("query", [
    "a jacket for the gym",           # no gender stated
    "my wife and her brother both run",  # mixed recipients -> no hard filter
    "something cozy for lounging",
])
def test_no_gender_stated(query):
    assert _explicit_gender(query) is None


def _womens_and_other_names():
    pairs = list(_product_genders(DOMAIN))
    womens = [n for n, g in pairs if g == "women"]
    other = [n for n, g in pairs if g == "men"]
    return womens, other


def test_hard_filter_drops_untagged_womens_product_for_a_man():
    womens, _ = _womens_and_other_names()
    assert womens, "fixture needs a women's product in the manifest"
    name = womens[0]
    # a product chunk whose payload has NO gender tag but whose text names a women's SKU
    hit = {"id": "x", "payload": {"doc_type": "product",
                                  "text": "The {} is a great pick.".format(name)}}
    kept = _gender_filter([hit], "men", DOMAIN)
    assert kept == [], "an untagged women's SKU must not survive the hard filter for a male shopper"


def test_hard_filter_keeps_untagged_mens_product_for_a_man():
    _, mens = _womens_and_other_names()
    if not mens:
        pytest.skip("no men's product in the manifest to test the keep path")
    hit = {"id": "y", "payload": {"doc_type": "product",
                                  "text": "The {} is a great pick.".format(mens[0])}}
    assert _gender_filter([hit], "men", DOMAIN) == [hit]


def test_hard_filter_keeps_non_product_and_unisex():
    guide = {"id": "g", "payload": {"doc_type": "guide", "text": "Layering for cold runs."}}
    unisex = {"id": "u", "payload": {"doc_type": "product", "text": "A unisex beanie in black."}}
    kept = _gender_filter([guide, unisex], "men", DOMAIN)
    assert guide in kept and unisex in kept


# --- gender carried across conversation turns (the "gift for my father" then "$100?" bug) ---

def _hist(*user_turns):
    """A conversation history from the shopper's turns; each is followed by a neutral assistant turn
    so the lookback must skip assistant turns to find the recipient's gender."""
    hist = []
    for u in user_turns:
        hist.append({"role": "user", "content": u})
        hist.append({"role": "assistant", "content": "Here are a few options."})
    return hist


@pytest.mark.parametrize("query,history,expected", [
    # the recipient is named once; a follow-up that does NOT restate gender still filters
    ("something around $100", _hist("a gift for my father"), "men"),
    ("which is the warmest layer", _hist("a present for my mom", "she likes yoga"), "women"),
    ("in black if you have it", _hist("shopping for her", "she's into the gym"), "women"),
    ("the warmest one?", _hist("i'm shopping for him"), "men"),
    # the CURRENT turn always wins over an earlier, different recipient
    ("actually a dress for my wife", _hist("a gift for my father"), "women"),
    ("men's leggings please", _hist("a present for my sister"), "men"),
    # the most recent prior turn that named a gender wins when the current turn names none
    ("around $100", _hist("a gift for my father", "actually make it for my mother"), "women"),
])
def test_sticky_gender_carries_and_current_turn_wins(query, history, expected):
    assert _sticky_gender(query, history) == expected


def test_sticky_gender_none_when_never_stated():
    assert _sticky_gender("something around $100",
                          _hist("what do you recommend for the gym")) is None
    assert _sticky_gender("hello", None) is None


def test_sticky_gender_ignores_the_assistants_words():
    # only the shopper's own turns set the recipient: the assistant asking "for her or him?" must
    # not flip the recipient the shopper actually named
    hist = [{"role": "user", "content": "a gift for my father"},
            {"role": "assistant", "content": "Sure! Is this for her or for him?"}]
    assert _sticky_gender("around $100", hist) == "men"


def test_redact_drops_an_opposite_gender_category_phrase():
    # a metric or category aggregate can name "women's bottoms" with no branded SKU; for a male
    # recipient that clause must be scrubbed so the model cannot recommend it (the "women's bottoms
    # for your father" leak), while the men's clause in the same context is kept
    ctx = [{"n": 1, "doc_type": "metric",
            "text": "Popular picks: Aster women's bottoms priced from $100. "
                    "Aster Coastal Hoodie for men at $104."}]
    out = _redact_contexts_by_gender(ctx, "men", DOMAIN)
    assert "women's bottoms" not in out[0]["text"].lower()
    assert "coastal hoodie" in out[0]["text"].lower()


def test_redact_keeps_a_unisex_mention():
    # a genuinely unisex clause naming both genders must survive (no over-scrub into an abstain)
    ctx = [{"n": 1, "doc_type": "guide", "text": "The beanie suits women and men alike."}]
    out = _redact_contexts_by_gender(ctx, "men", DOMAIN)
    assert "beanie" in out[0]["text"].lower()
