"""Staleness monitoring and a human-triggered refresh PROPOSAL.

The flywheel stamps each verified chunk with `indexed_at`. Over time some of those answers outlive
the data behind them (an enrichment consensus flips, a policy changes). This flags the chunks older
than a threshold and proposes a refresh a person approves. It never expires or re-indexes anything
on its own, so the corpus is never silently mutated. Pure and time-injected, so it tests offline.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _epoch(ts: str | None) -> float | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return None


def staleness_report(records: list[dict], *, now_ts: float, max_age_days: int = 90) -> dict:
    """How much of the verified knowledge has aged past max_age_days. `records` are chunk payloads
    that may carry `indexed_at`; undated chunks are counted separately, not assumed fresh."""
    cutoff = now_ts - max_age_days * 86400
    dated = [(r, _epoch(r.get("indexed_at"))) for r in records]
    dated = [(r, t) for r, t in dated if t is not None]
    stale = [r for r, t in dated if t < cutoff]
    ages = [(now_ts - t) / 86400 for _, t in dated]
    return {
        "total": len(records),
        "dated": len(dated),
        "undated": len(records) - len(dated),
        "stale": len(stale),
        "max_age_days": max_age_days,
        "oldest_days": round(max(ages), 1) if ages else None,
        "stale_ids": [r.get("chunk_id") or r.get("id") for r in stale][:50],
    }


def refresh_proposal(report: dict, *, new_reviews: int = 0) -> dict:
    """A proposed, human-triggered refresh plan. Never auto-run: a person reads it and decides."""
    actions = []
    if report.get("stale"):
        actions.append("review {} verified chunk(s) older than {} days for re-index/expiry".format(
            report["stale"], report["max_age_days"]))
    if new_reviews:
        actions.append("index {} new verified answer(s) since the last refresh".format(new_reviews))
    if not actions:
        actions.append("nothing to refresh: no stale chunks and no new verified answers")
    return {"proposed_actions": actions, "human_gated": True,
            "note": "PROPOSED only: a person approves; nothing expires or re-indexes automatically"}
