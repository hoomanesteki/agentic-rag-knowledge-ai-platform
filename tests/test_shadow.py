"""Shadow replay produces champion-vs-challenger evidence for a human to approve with, and never
promotes anything itself. These pin the champion loader and the delta comparison, offline."""
import json

from mlops.shadow import compare, load_champion_questions


def test_load_champion_questions_takes_the_recent_answered_traffic(tmp_path):
    p = tmp_path / "traces.jsonl"
    rows = [{"query": "q{}".format(i), "message_id": "m{}".format(i), "ts": float(i),
             "grounding": 0.9, "lane": "answers", "cost": 0.003, "tier": "auto"}
            for i in range(5)]
    rows.append({"ts": 9.0})  # a non-question trace line is skipped
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    items = load_champion_questions(str(p), n=3)
    assert [i["question"] for i in items] == ["q2", "q3", "q4"]  # the most recent 3
    assert items[0]["champion"]["grounding"] == 0.9


def test_compare_reports_deltas_and_never_promotes():
    items = [
        {"champion": {"grounding": 0.9, "cost": 0.003, "lane": "answers", "tier": "auto"},
         "challenger": {"grounding": 0.8, "cost": 0.001, "lane": "stylist", "tier": "auto"}},
        {"champion": {"grounding": 1.0, "cost": 0.003, "lane": "care", "tier": "auto"},
         "challenger": {"grounding": 1.0, "cost": 0.001, "lane": "care", "tier": "abstain"}},
    ]
    report = compare(items)
    assert report["n"] == 2
    assert report["grounding"]["delta"] == round((0.8 + 1.0) / 2 - (0.9 + 1.0) / 2, 6)
    assert report["cost"]["delta"] < 0  # the challenger is cheaper here
    assert report["route_flips"] == 1  # answers -> stylist on the first item
    assert report["abstain_delta"] == 1  # the challenger abstained once more
    assert "human" in report["recommendation"].lower()  # evidence only, no auto-promotion
