#!/usr/bin/env python3
"""Emit the site's headline numbers to evaluation/reports/site_stats.json from the committed
artifacts and the code, so the README and the Quarto pages READ them instead of hand-typing a
figure that drifts. One source of truth; run this after an eval regenerates a report.

Run: make site-stats   (or: python scripts/build_site_stats.py)
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys

_COLLECT_LINE = re.compile(r"^tests/\S+\.py:\s*(\d+)\s*$")

_REPORTS = "evaluation/reports"
_OUT = os.path.join(_REPORTS, "site_stats.json")


def _read(name: str) -> dict:
    path = os.path.join(_REPORTS, name)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _routing() -> dict:
    cards = _read("routing_eval.json").get("scorecards", [])
    out: dict = {}
    for c in cards:
        mode = (c.get("mode") or "").lower()
        pct = round(c.get("accuracy", 0.0) * 100, 1)
        esc = round((c.get("escalation") or {}).get("recall", 0.0) * 100, 1)
        if "deterministic" in mode:
            out["deterministic_pct"] = pct
            out["deterministic_escalation_recall_pct"] = esc
        elif "8b" in mode:
            out["tiebreak_8b_pct"] = pct
            out["tiebreak_8b_escalation_recall_pct"] = esc
        elif "70b" in mode:
            out["tiebreak_70b_pct"] = pct
    return out


def _cost() -> dict:
    c = _read("cost_model.json")
    ratios = c.get("ratios_vs_text_70b", {})
    per_turn = c.get("per_turn", {})
    return {"human_vs_ai_ratio": ratios.get("human_agent"),
            "text_turn_usd": per_turn.get("text_70b"),
            "human_turn_usd": per_turn.get("human_agent")}


def _test_count() -> int | None:
    # `pytest --collect-only -q` prints one "path/test_x.py: N" line per file; sum the N. Fast (no
    # execution) and reproducible, so the headline test count is never hand-typed.
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                              capture_output=True, text=True, timeout=300)
        total = 0
        for ln in proc.stdout.splitlines():
            m = _COLLECT_LINE.match(ln.strip())
            if m:
                total += int(m.group(1))
        return total or None
    except Exception:
        return None


def _golden_size() -> int:
    path = "domains/apparel_ecommerce/eval/golden.jsonl"
    if not os.path.exists(path):
        return 0
    return sum(1 for ln in open(path, encoding="utf-8") if ln.strip())


def _catalog() -> dict:
    path = "domains/apparel_ecommerce/seed/structured/products.csv"
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {"products": len({r["name"] for r in rows}), "variants": len(rows)}


def _prompt_opt() -> dict:
    # The promoted tie-break prompt's held-out scores, read from the registry candidate record so
    # the OPRO gain the pages quote is derived, never hand-typed. (prompt_opt.json is the LATEST
    # loop run, which found no further gain against the now-served prompt; the served prompt's own
    # promotion numbers are here.)
    path = "mlops/prompt_registry/tiebreak_system.candidate.json"
    d = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            d = {}
    base, cand = d.get("baseline_test"), d.get("candidate_test")
    if base is None or cand is None:
        return {}
    return {"baseline_pct": round(base * 100, 1),
            "candidate_pct": round(cand * 100, 1),
            "delta_pts": round((cand - base) * 100, 1)}


def _gate_fixtures() -> dict:
    # The CI eval-gate fixtures, split by expectation, so the page never hand-types the count.
    path = "evaluation/fixtures/gate.json"
    fx: list = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                fx = json.load(f).get("fixtures", [])
        except (OSError, json.JSONDecodeError):
            fx = []
    if not fx:
        return {}
    return {"total": len(fx),
            "retrieves": sum(1 for f in fx if f.get("expect") == "retrieves"),
            "abstain": sum(1 for f in fx if f.get("expect") == "abstain")}


def build() -> dict:
    stats = {
        "tests": _test_count(),
        "golden_items": _golden_size(),
        "routing": _routing(),
        "cost": _cost(),
        "catalog": _catalog(),
        "routing_eval_set": _read("routing_eval.json").get("scorecards", [{}])[0].get("n"),
        "prompt_opt": _prompt_opt(),
        "gate": _gate_fixtures(),
    }
    return {k: v for k, v in stats.items() if v not in (None, {}, [])}


def main() -> int:
    stats = build()
    os.makedirs(_REPORTS, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("wrote {}:".format(_OUT))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
