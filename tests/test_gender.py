"""Gender is a hard facet: when a shopper states it (directly or via a gendered relative noun), the
assistant must never surface an opposite-gender SKU. These lock in the recipient-noun precedence
(a gift "for her husband" is for a man), the natural direct phrasings ("shorts for men"), and the
retrieval hard-filter's fallback when a product chunk carries no gender tag.
"""
import pytest

from pipeline.answer import _explicit_gender, _gender_filter, _product_genders

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
