#!/usr/bin/env python3
"""Seed realistic back-office data for the demo: request traces, thumbs feedback, and a populated
review queue.

The admin dashboards (quality, health, knowledge gaps, review queue) all read live runtime data:
traces/requests.jsonl, traces/feedback.jsonl, and the review-queue SQLite DB. On a fresh clone or
after ad-hoc testing those are empty or full of throwaway queries ("capital of france"), so the
dashboards look broken. This regenerates them with a realistic day of traffic and genuine
domain-relevant gaps, so a hiring manager sees a working assistant back office.

Synthetic only. Run: uv run python domains/apparel_ecommerce/tools/gen_backoffice.py
"""
import json
import os
import random
import sqlite3
import time
import uuid

DOMAIN = "apparel_ecommerce"
TRACE_PATH = os.getenv("TRACE_PATH", "traces/requests.jsonl")
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "traces/feedback.jsonl")
QUEUE_DB = os.getenv("REVIEW_QUEUE_DB", ".review_queue.db")
MODEL = "llama-3.3-70b-versatile"

# Questions the assistant answers well (drive the "auto" volume and the funnel). Real shopping
# questions across products, sizing, shipping, policy, and orders.
ANSWERED = [
    "how much is the flow legging", "which jacket is best for rain", "do the leggings have pockets",
    "what is your return policy", "how long to ship to toronto", "is the cloud hoodie true to size",
    "what colors does the storm shell come in", "do you have gift cards", "what is aster circle",
    "which bag is best for the gym", "is the base merino itchy", "what is your most popular legging",
    "do you offer free shipping", "what payment methods do you take", "how do students get a discount",
    "which products are good for sensitive skin", "what should I wear for a rainy commute",
    "combien coute le legging flow", "quelle veste pour la pluie", "do you ship to the US",
    "what is trending right now", "which hoodie is warmest", "is express shipping available",
    "what is the size guide for tops", "do you have a first time buyer discount",
]
# Genuine knowledge gaps: real questions the assistant does not have an answer for. These become
# the "unanswered questions" list, with a frequency so the counts look like real demand.
GAPS = [
    ("do you carry tall or long inseam sizes", 11),
    ("when will the aspen parka be back in stock", 9),
    ("do you ship to australia", 7),
    ("do your leggings come in a 3xl", 6),
    ("can I gift wrap an order", 6),
    ("do you have a physical store in toronto", 5),
    ("is there a military discount", 5),
    ("do you price match other brands", 4),
    ("what is the carbon footprint of my shipment", 3),
    ("do you make a matching set in petite", 3),
    ("est-ce que vous livrez en europe", 4),
    ("do you sell maternity leggings", 3),
]
# Escalations: questions routed to a human specialist (they populate the review queue too). These
# need a person: account actions, order problems, business requests.
ESCALATIONS = [
    ("I received the wrong item in my order", 4),
    ("my discount code will not apply at checkout", 4),
    ("my package says delivered but I never got it", 3),
    ("I need to change the shipping address on an order I just placed", 3),
    ("can I get an invoice with my company details for expenses", 2),
    ("do you offer wholesale pricing for a yoga studio", 2),
    ("I want to cancel an order I placed an hour ago", 2),
]


def _expand(weighted):
    out = []
    for text, n in weighted:
        out.extend([text] * n)
    return out


def main():
    rng = random.Random(42)
    os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
    now = time.time()
    span = 6 * 3600  # traffic over the last ~6 hours

    rows = []

    def stamp(i, total):
        # spread across the window, with a denser cluster in the last 15 min so throughput/min is
        # realistic; add jitter so timestamps are not evenly spaced
        base = now - span * (1 - i / total)
        if rng.random() < 0.12:
            base = now - rng.uniform(0, 850)  # recent burst
        return round(base + rng.uniform(-30, 0), 3)

    n_auto = 520
    for i in range(n_auto):
        lang = "fr" if rng.random() < 0.15 else "en"
        # a slight upward grounding trend over the window, so the health "trend" shows improvement
        trend = 0.02 * (i / n_auto)
        grounding = round(min(0.98, max(0.45, rng.gauss(0.80 + trend, 0.09))), 3)
        rows.append({
            "ts": stamp(i, n_auto), "message_id": uuid.uuid4().hex,
            "query": rng.choice(ANSWERED), "lang": lang, "tier": "auto", "model": MODEL,
            "grounding": grounding, "confidence": round(min(0.99, grounding + rng.uniform(-0.05, 0.1)), 3),
            "latency_ms": round(rng.uniform(430, 2400), 1),
            "cost": round(rng.uniform(0.0004, 0.0017), 6), "streamed": False,
        })

    for q in _expand(GAPS):
        lang = "fr" if q.startswith(("est-", "quelle", "combien")) else "en"
        rows.append({"ts": stamp(rng.randint(0, n_auto), n_auto), "message_id": uuid.uuid4().hex,
                     "query": q, "lang": lang, "tier": "abstain", "model": None, "grounding": 0.0,
                     "confidence": round(rng.uniform(0.05, 0.24), 3),
                     "latency_ms": round(rng.uniform(210, 700), 1)})

    for q in _expand(ESCALATIONS):
        rows.append({"ts": stamp(rng.randint(0, n_auto), n_auto), "message_id": uuid.uuid4().hex,
                     "query": q, "lang": "en", "tier": "escalate", "model": None,
                     "grounding": 0.0, "confidence": round(rng.uniform(0.1, 0.3), 3),
                     "latency_ms": round(rng.uniform(240, 800), 1)})

    # a few degraded turns so resilience is visible without dominating the error rate
    for _ in range(5):
        rows.append({"ts": stamp(rng.randint(0, n_auto), n_auto), "message_id": uuid.uuid4().hex,
                     "query": rng.choice(ANSWERED), "lang": "en", "tier": "degraded",
                     "model": MODEL, "latency_ms": round(rng.uniform(2500, 4200), 1)})

    rows.sort(key=lambda r: r["ts"])
    with open(TRACE_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # thumbs feedback joined to answered turns, mostly positive
    auto_ids = [r["message_id"] for r in rows if r["tier"] == "auto"]
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        for mid in rng.sample(auto_ids, k=min(140, len(auto_ids))):
            verdict = "up" if rng.random() < 0.86 else "down"
            f.write(json.dumps({"ts": now - rng.uniform(0, span), "message_id": mid,
                                "verdict": verdict, "user": "shopper"}) + "\n")

    # populate the review queue: open items to claim plus a couple already claimed, all for this
    # domain. Clear prior seeds for the domain first so re-running stays clean.
    conn = sqlite3.connect(QUEUE_DB, timeout=5.0)
    conn.execute("CREATE TABLE IF NOT EXISTS review_queue ("
                 "id TEXT PRIMARY KEY, domain TEXT, lang TEXT, message_id TEXT, "
                 "question TEXT NOT NULL, route TEXT, status TEXT NOT NULL DEFAULT 'open', "
                 "answer TEXT, answered_by TEXT, created_at REAL, claimed_at REAL, resolved_at REAL)")
    conn.execute("DELETE FROM review_queue WHERE domain = ?", (DOMAIN,))
    queue_items = _expand(ESCALATIONS) + [q for q, _ in GAPS[:5]]
    for idx, q in enumerate(queue_items):
        status = "claimed" if idx in (0, 1) else "open"
        created = now - rng.uniform(300, span)
        conn.execute(
            "INSERT INTO review_queue (id, domain, lang, message_id, question, route, status, "
            "answered_by, created_at, claimed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, DOMAIN, "en", uuid.uuid4().hex, q,
             "escalate" if idx < len(_expand(ESCALATIONS)) else "abstain", status,
             "newadmin" if status == "claimed" else None, created,
             now - 400 if status == "claimed" else None))
    conn.commit()
    open_n = conn.execute("SELECT count(*) FROM review_queue WHERE domain=? AND status='open'",
                          (DOMAIN,)).fetchone()[0]
    claimed_n = conn.execute("SELECT count(*) FROM review_queue WHERE domain=? AND status='claimed'",
                             (DOMAIN,)).fetchone()[0]
    conn.close()

    print(f"traces: {len(rows)} turns -> {TRACE_PATH}")
    print(f"feedback: written -> {FEEDBACK_PATH}")
    print(f"review queue: {open_n} open + {claimed_n} claimed for {DOMAIN}")


if __name__ == "__main__":
    main()
