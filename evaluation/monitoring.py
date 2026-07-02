"""Aggregate the per-request traces and thumbs feedback into back-office views (M7).

Everything is derived from the trace and feedback JSONL that every request already writes, so the
dashboards read real traffic with no extra store. Trace fields are generic (tier, grounding,
route, lang, cost, latency), so this stays domain agnostic. M7.2 uses aggregate_quality; M7.5
uses aggregate_health.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict, deque

_THROUGHPUT_WINDOW = 900.0  # seconds; throughput is measured over the most recent window

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    """Read a JSONL trace file, newest last. With a limit, only the last `limit` rows are kept in
    memory (a deque), so a large file cannot blow up the admin endpoint. Missing file is empty; a
    bad line is skipped rather than failing the whole view."""
    if not os.path.isfile(path):
        return []
    rows: deque | list = deque(maxlen=limit) if limit else []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


def _lang(trace: dict) -> str:
    return trace.get("lang") or "unknown"


def aggregate_quality(traces: list[dict], feedback: list[dict]) -> dict:
    """Answer quality by language: tier mix, escalation and abstain rates, average grounding of
    answered turns, and thumbs. Thumbs are joined to a turn's language by message_id."""
    def bucket() -> dict:
        return {"total": 0, "tiers": Counter(), "grounding_sum": 0.0, "grounded": 0,
                "thumbs_up": 0, "thumbs_down": 0}

    overall = bucket()
    by_language: dict[str, dict] = defaultdict(bucket)
    lang_of: dict[str, str] = {}

    for trace in traces:
        lang = _lang(trace)
        message_id = trace.get("message_id")
        if message_id:
            lang_of[message_id] = lang
        for target in (overall, by_language[lang]):
            target["total"] += 1
            target["tiers"][trace.get("tier", "unknown")] += 1
            grounding = trace.get("grounding")
            if trace.get("tier") == "auto" and isinstance(grounding, (int, float)):
                target["grounding_sum"] += grounding
                target["grounded"] += 1

    unmatched_feedback = 0
    for entry in feedback:
        key = {"up": "thumbs_up", "down": "thumbs_down"}.get(entry.get("verdict"))
        if not key:
            continue
        overall[key] += 1
        lang = lang_of.get(entry.get("message_id"))
        if lang is not None:
            by_language[lang][key] += 1  # only real trace languages get a bucket
        else:
            unmatched_feedback += 1  # a thumb on a turn outside the window; not a phantom language

    def finalize(b: dict) -> dict:
        # rates are over turns the system actually served (exclude infra degraded/error), so an
        # outage does not flatter the escalation rate
        served = b["total"] - b["tiers"].get("degraded", 0) - b["tiers"].get("error", 0)
        denom = served or 1
        avg_grounding = round(b["grounding_sum"] / b["grounded"], 3) if b["grounded"] else None
        return {
            "total": b["total"],
            "served": served,
            "tiers": dict(b["tiers"]),
            "escalation_rate": round(b["tiers"].get("escalate", 0) / denom, 3),
            "abstain_rate": round(b["tiers"].get("abstain", 0) / denom, 3),
            "avg_grounding": avg_grounding,
            "thumbs_up": b["thumbs_up"],
            "thumbs_down": b["thumbs_down"],
        }

    overall_out = finalize(overall)
    overall_out["unmatched_feedback"] = unmatched_feedback
    return {"overall": overall_out,
            "by_language": {lang: finalize(b) for lang, b in sorted(by_language.items())}}


def _p95(values: list) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)  # nearest-rank
    return ordered[index]


def _throughput_per_min(timestamps: list) -> float | None:
    """Requests per minute over the most recent window, so buckets are comparable and it reflects
    current load rather than the age of the whole trace file."""
    if len(timestamps) < 2:
        return None
    latest = max(timestamps)
    start = max(min(timestamps), latest - _THROUGHPUT_WINDOW)
    minutes = (latest - start) / 60
    if minutes <= 0:
        return None
    return round(sum(1 for t in timestamps if t >= start) / minutes, 2)


def _grounding_trend(traces: list[dict]) -> dict:
    """Average grounding of the earlier half vs the recent half of answered turns (ordered by
    time, since traces are written at completion and can interleave), so a drift in retrieval
    quality is visible rather than hidden in one flat average."""
    grounded = sorted(
        ((t.get("ts", 0.0), t["grounding"]) for t in traces
         if t.get("tier") == "auto" and isinstance(t.get("grounding"), (int, float))),
        key=lambda pair: pair[0])
    values = [g for _, g in grounded]
    if len(values) < 4:
        return {"early": None, "recent": None, "delta": None}
    mid = len(values) // 2
    early = round(sum(values[:mid]) / mid, 3)
    recent = round(sum(values[mid:]) / (len(values) - mid), 3)
    return {"early": early, "recent": recent, "delta": round(recent - early, 3)}


def aggregate_health(traces: list[dict]) -> dict:
    """Operational health from the request traces: throughput, p95 latency, error rate, average
    cost per request, and grounding, overall and by language, plus a retrieval-quality trend."""
    def bucket() -> dict:
        return {"total": 0, "latencies": [], "auto_latencies": [], "errors": 0, "cost_sum": 0.0,
                "costed": 0, "grounding_sum": 0.0, "grounded": 0, "timestamps": []}

    overall = bucket()
    by_language: dict[str, dict] = defaultdict(bucket)
    for trace in traces:
        tier = trace.get("tier")
        for target in (overall, by_language[_lang(trace)]):
            target["total"] += 1
            latency = trace.get("latency_ms")
            if isinstance(latency, (int, float)):
                target["latencies"].append(latency)
                if tier == "auto":
                    target["auto_latencies"].append(latency)
            if tier in ("error", "degraded"):
                target["errors"] += 1
            cost = trace.get("cost")
            if isinstance(cost, (int, float)):
                target["cost_sum"] += cost
                target["costed"] += 1
            grounding = trace.get("grounding")
            if tier == "auto" and isinstance(grounding, (int, float)):
                target["grounding_sum"] += grounding
                target["grounded"] += 1
            ts = trace.get("ts")
            if isinstance(ts, (int, float)):
                target["timestamps"].append(ts)

    def finalize(b: dict) -> dict:
        total = b["total"] or 1
        return {
            "total": b["total"],
            "p95_latency_ms": _p95(b["latencies"]),
            # auto-only so slow answers are not masked by fast abstains
            "p95_latency_auto_ms": _p95(b["auto_latencies"]),
            "error_rate": round(b["errors"] / total, 3),
            "avg_cost": round(b["cost_sum"] / b["costed"], 6) if b["costed"] else None,
            "costed_turns": b["costed"],  # denominator is explicit; streaming omits cost (M8)
            "avg_grounding": (round(b["grounding_sum"] / b["grounded"], 3)
                              if b["grounded"] else None),
            "throughput_per_min": _throughput_per_min(b["timestamps"]),
        }

    result = {"overall": finalize(overall),
              "by_language": {lang: finalize(b) for lang, b in sorted(by_language.items())}}
    result["overall"]["grounding_trend"] = _grounding_trend(traces)
    return result


def aggregate_gaps(traces: list[dict], limit: int = 50) -> list[dict]:
    """The knowledge gaps: questions the system could not answer well (abstained or escalated),
    most frequent first. Case-insensitive so the same question does not split; emails are masked
    and the text is capped, since a user query can carry PII even in an admin view."""
    counts: Counter = Counter()
    original: dict[str, str] = {}
    langs: dict[str, Counter] = defaultdict(Counter)
    for trace in traces:
        if trace.get("tier") not in ("abstain", "escalate"):
            continue
        question = (trace.get("query") or "").strip()
        if not question:
            continue
        key = question.casefold()
        counts[key] += 1
        original.setdefault(key, question)
        langs[key][trace.get("lang") or "unknown"] += 1
    return [{"question": _EMAIL.sub("<email>", original[key])[:200], "count": count,
             "lang": langs[key].most_common(1)[0][0]}
            for key, count in counts.most_common(limit)]
