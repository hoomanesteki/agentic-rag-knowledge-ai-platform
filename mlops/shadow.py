"""Human-triggered shadow replay: champion vs challenger on real traffic, offline, zero user risk.

Before a human promotes a candidate prompt or config, they run it here. The last N real questions
are replayed through the challenger and its answers compared to the champion's recorded results on
grounding, route, and cost. The human reads the delta and THEN decides; nothing auto-promotes, this
only produces the evidence to approve with. Offline replay over historical requests is the
right-sized local-first pattern; a live canary on a traffic slice is the documented scale-up.

Shadow-then-canary is the mature 2025-26 rollout sequence (champion/challenger through the registry
with a human controlling promotion); this is the shadow half, decoupled from serving.
"""
from __future__ import annotations

import json
import os


def load_champion_questions(trace_path: str, n: int) -> list[dict]:
    """The last N answered questions and the champion's recorded result per question."""
    rows = []
    if not os.path.exists(trace_path):
        return rows
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("query"):
                rows.append(r)
    rows = sorted(rows, key=lambda r: r.get("ts", 0.0))[-n:]
    return [{"question": r.get("query", ""), "message_id": r.get("message_id"),
             "champion": {"grounding": r.get("grounding"), "lane": r.get("lane"),
                          "cost": r.get("cost"), "tier": r.get("tier")}}
            for r in rows]


def _mean(values) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(nums) / len(nums), 6) if nums else None


def compare(items: list[dict]) -> dict:
    """Aggregate champion-vs-challenger deltas over replayed items. Each item carries a `champion`
    and a `challenger` dict with grounding / cost / tier / lane. The recommendation is always to let
    a human decide: this produces evidence, it does not promote."""
    champ_g = _mean([i["champion"].get("grounding") for i in items])
    chall_g = _mean([i.get("challenger", {}).get("grounding") for i in items])
    champ_c = _mean([i["champion"].get("cost") for i in items])
    chall_c = _mean([i.get("challenger", {}).get("cost") for i in items])
    route_flips = sum(1 for i in items
                      if i["champion"].get("lane") != i.get("challenger", {}).get("lane"))
    abstain_delta = (sum(1 for i in items if i.get("challenger", {}).get("tier") == "abstain")
                     - sum(1 for i in items if i["champion"].get("tier") == "abstain"))

    def delta(a, b):
        return round((b or 0) - (a or 0), 6) if (a is not None or b is not None) else None

    return {
        "n": len(items),
        "grounding": {"champion": champ_g, "challenger": chall_g, "delta": delta(champ_g, chall_g)},
        "cost": {"champion": champ_c, "challenger": chall_c, "delta": delta(champ_c, chall_c)},
        "route_flips": route_flips,
        "abstain_delta": abstain_delta,
        "recommendation": "evidence only: a human reads these deltas and decides, no auto-promote.",
    }
