"""Lock the README's hand-typed headline numbers to the committed site_stats.json.

The showcase pages read their numbers from site_stats.json via {python} chunks, so they cannot
drift. The README is plain markdown and types the same figures by hand, so this test fails the
build if any of them fall out of sync with the source of truth. Run `make site-stats` after an eval
regenerates a report, then update the README to match; this test is the guard that you did.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_headline_numbers_match_site_stats():
    stats = json.loads((ROOT / "evaluation/reports/site_stats.json").read_text())
    readme = (ROOT / "README.md").read_text()
    routing = stats["routing"]
    cost = stats["cost"]
    catalog = stats["catalog"]

    expected = {
        "tests": str(stats["tests"]),
        "golden": str(stats["golden_items"]),
        "routing set": str(stats["routing_eval_set"]),
        "deterministic %": "{:g}%".format(routing["deterministic_pct"]),
        "8b tie-break %": "{:g}%".format(routing["tiebreak_8b_pct"]),
        "70b tie-break %": "{:g}%".format(routing["tiebreak_70b_pct"]),
        "escalation recall %": "{:g}%".format(routing["tiebreak_8b_escalation_recall_pct"]),
        "text turn $": "${:.4f}".format(cost["text_turn_usd"]),
        "human turn $": "${:.4f}".format(cost["human_turn_usd"]),
        "cost ratio": "{:.0f}x".format(cost["human_vs_ai_ratio"]),
        "products": str(catalog["products"]),
        "variants": str(catalog["variants"]),
    }
    missing = {label: value for label, value in expected.items() if value not in readme}
    assert not missing, "README numbers out of sync with site_stats.json: {}".format(missing)
