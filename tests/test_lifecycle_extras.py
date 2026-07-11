"""Staleness flags aged verified chunks and proposes a human-approved refresh (never mutating), and
the learning curve records quality over CT cycles so self-learning is provable. Offline, pure."""
from datetime import datetime, timezone

from mlops.learning_curve import append_point, read_curve, trend
from mlops.staleness import refresh_proposal, staleness_report

_NOW = 1_800_000_000.0  # a fixed 'now' so the tests are deterministic
_DAY = 86400.0


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_staleness_report_counts_dated_stale_and_undated():
    records = [
        {"chunk_id": "a", "indexed_at": _iso(_NOW - 200 * _DAY)},  # old -> stale
        {"chunk_id": "b", "indexed_at": _iso(_NOW - 10 * _DAY)},   # recent -> fresh
        {"chunk_id": "c"},                                         # undated, not assumed fresh
    ]
    rep = staleness_report(records, now_ts=_NOW, max_age_days=90)
    assert rep["total"] == 3 and rep["dated"] == 2 and rep["undated"] == 1
    assert rep["stale"] == 1 and rep["stale_ids"] == ["a"]
    assert rep["oldest_days"] >= 199


def test_refresh_proposal_is_human_gated():
    prop = refresh_proposal({"stale": 2, "max_age_days": 90}, new_reviews=5)
    assert prop["human_gated"] is True and "PROPOSED only" in prop["note"]
    assert any("older than 90 days" in a for a in prop["proposed_actions"])
    assert any("5 new verified" in a for a in prop["proposed_actions"])


def test_learning_curve_append_read_and_trend(tmp_path):
    p = str(tmp_path / "curve.jsonl")
    append_point(p, at="2026-02-01T00:00:00Z", score=0.85, n_examples=25, version=2)
    append_point(p, at="2026-01-01T00:00:00Z", score=0.80, n_examples=10, version=1)
    curve = read_curve(p)
    assert [c["score"] for c in curve] == [0.80, 0.85]  # sorted by timestamp
    t = trend(curve)
    assert t["improving"] is True and t["delta"] == 0.05 and t["points"] == 2


def test_trend_is_unknown_with_a_single_point(tmp_path):
    p = str(tmp_path / "curve.jsonl")
    append_point(p, at="2026-01-01T00:00:00Z", score=0.8, n_examples=10)
    assert trend(read_curve(p))["improving"] is None
