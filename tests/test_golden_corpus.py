"""The graded golden set is the substrate every cost flip is validated against, so its integrity
is enforced in CI, not trusted. A 20-item flat set cannot see a 5% regression; a stratified set
can, but only if the strata stay populated and honestly labelled. These guard that contract for
the apparel pack (the graded pack). Correctness of each answerable label is enforced upstream by
the grounded builder that derives every expected field from the real seed data."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "domains" / "apparel_ecommerce" / "eval" / "golden.jsonl"

pytestmark = pytest.mark.skipif(not GOLDEN.is_file(), reason="apparel golden set absent")


def _rows():
    return [json.loads(ln) for ln in GOLDEN.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_corpus_is_large_enough_to_see_a_regression():
    # a handful of items averages away small regressions; the graded set is sized to catch them
    assert len(_rows()) >= 100


def test_every_item_is_graded_common_semi_or_edge():
    for r in _rows():
        assert r.get("difficulty") in ("common", "semi", "edge"), r["id"]


def test_each_difficulty_stratum_is_populated():
    counts: dict[str, int] = {}
    for r in _rows():
        counts[r["difficulty"]] = counts.get(r["difficulty"], 0) + 1
    for stratum in ("common", "semi", "edge"):
        assert counts.get(stratum, 0) >= 15, "thin {} stratum: {}".format(stratum, counts)


def test_both_languages_and_all_types_present():
    rows = _rows()
    assert {"en", "fr"} <= {r["lang"] for r in rows}
    assert {"answerable", "unanswerable", "out_of_domain"} <= {r["type"] for r in rows}


def test_adversarial_items_expect_a_refusal_not_an_answer():
    # a false-premise / third-party-PII / injection probe must be labelled to abstain, never
    # answerable, so scoring it rewards refusal rather than a confident wrong answer
    for r in _rows():
        if r.get("adversarial"):
            assert r["type"] in ("unanswerable", "out_of_domain"), r["id"]


def test_measurable_answerable_items_name_their_expected_entities():
    # a qualitative item is only retrieval-scorable if it declares the entity it should surface
    for r in _rows():
        if r["type"] == "answerable" and r.get("route") == "qualitative":
            assert r.get("expected_entities"), r["id"]


def test_ids_and_questions_are_unique():
    rows = _rows()
    ids = [r["id"] for r in rows]
    questions = [r["question"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate id"
    assert len(questions) == len(set(questions)), "duplicate question"
