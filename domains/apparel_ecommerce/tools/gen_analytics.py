#!/usr/bin/env python3
"""Simulate about 1,200 shopper sessions and aggregate them into the metrics a store manager
cares about: traffic, top searches and questions, most-viewed products, category mix, a
conversion funnel, and revenue. Writes seed/analytics.json, which the admin API serves.

Lives in the domain pack, so its brand vocabulary does not trip the engine leak linter.
Run: python domains/apparel_ecommerce/tools/gen_analytics.py
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

random.seed(7)

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
OUT = os.path.join(PACK, "seed", "analytics.json")

N = 1240
DAYS = 30

# weighted pools, so the aggregates look like a real store's mix
SEARCHES = (["leggings"] * 9 + ["jacket"] * 8 + ["hoodie"] * 7 + ["shorts"] * 5 + ["sports bra"] * 5
            + ["storm shell"] * 4 + ["flow legging"] * 4 + ["tops"] * 4 + ["joggers"] * 3
            + ["bags"] * 3 + ["merino"] * 3 + ["rain jacket"] * 3 + ["tank"] * 2 + ["beanie"] * 2
            + ["sale"] * 3 + ["gift"] * 2 + ["black leggings"] * 3 + ["men jacket"] * 3)
QUESTIONS = (["how long to ship to toronto"] * 7 + ["what is the return policy"] * 7
             + ["is it in stock in my size"] * 6 + ["do the leggings run small"] * 5
             + ["which jacket for rain"] * 5 + ["gift for my girlfriend"] * 4
             + ["what should I wear for summer"] * 4 + ["do you ship to the us"] * 4
             + ["how much is shipping"] * 4 + ["price range for hoodies"] * 3
             + ["student discount"] * 3 + ["how do I wash merino"] * 2
             + ["do you have kids products"] * 2 + ["can I talk to a human"] * 2
             + ["what colors do you have"] * 3 + ["free shipping"] * 4)
PRODUCTS = (["Flow Legging"] * 9 + ["Storm Shell Jacket"] * 8 + ["Vent Tech Tee"] * 6
            + ["Cloud Hoodie"] * 6 + ["Studio Sports Bra"] * 5 + ["Trailhead Puffer"] * 4
            + ["Momentum Short"] * 4 + ["Everywhere Jogger"] * 3 + ["Base Merino Long Sleeve"] * 3
            + ["Commute Tote"] * 3 + ["Peak Beanie"] * 2 + ["Daytrip Belt Bag"] * 2)
CATEGORIES = (["leggings"] * 8 + ["jackets"] * 7 + ["tops"] * 7 + ["hoodies"] * 5 + ["shorts"] * 4
              + ["bras"] * 4 + ["bottoms"] * 4 + ["bags"] * 3 + ["accessories"] * 2)
DEVICES = ["Mobile"] * 60 + ["Desktop"] * 32 + ["Tablet"] * 8
COUNTRIES = ["Canada"] * 62 + ["United States"] * 33 + ["Other"] * 5
MONTHS = ["Jun", "Jul"]


def main():
    searches, questions, products, cats, devices, countries = (Counter() for _ in range(6))
    by_day = Counter()
    funnel = {"visit": 0, "engaged": 0, "viewed": 0, "cart": 0, "order": 0}
    revenue, orders, search_events, question_events = 0.0, 0, 0, 0

    for _ in range(N):
        funnel["visit"] += 1
        by_day[random.randint(0, DAYS - 1)] += 1
        devices[random.choice(DEVICES)] += 1
        countries[random.choice(COUNTRIES)] += 1
        engaged = random.random() < 0.82
        if not engaged:
            continue
        funnel["engaged"] += 1
        for _ in range(random.choice([0, 1, 1, 2, 3])):
            searches[random.choice(SEARCHES)] += 1
            search_events += 1
        for _ in range(random.choice([0, 0, 1, 1, 2])):
            questions[random.choice(QUESTIONS)] += 1
            question_events += 1
        if random.random() < 0.74:
            funnel["viewed"] += 1
            for _ in range(random.choice([1, 1, 2, 3])):
                products[random.choice(PRODUCTS)] += 1
                cats[random.choice(CATEGORIES)] += 1
            if random.random() < 0.42:
                funnel["cart"] += 1
                if random.random() < 0.48:
                    funnel["order"] += 1
                    orders += 1
                    revenue += random.choice([68, 88, 98, 108, 118, 146, 178, 206, 228, 256])

    def top(counter, key, n=8):
        return [{key: k, "count": c} for k, c in counter.most_common(n)]

    def pct(counter):
        total = sum(counter.values()) or 1
        return [{"label": k, "pct": round(100 * c / total)} for k, c in counter.most_common()]

    summary = {
        "generated_sessions": N,
        "kpis": {
            "sessions": N,
            "visitors": int(N * 0.79),
            "searches": search_events,
            "questions": question_events,
            "add_to_cart": funnel["cart"],
            "orders": orders,
            "revenue": round(revenue),
            "conversion": round(100 * orders / N, 1),
            "aov": round(revenue / orders) if orders else 0,
        },
        "sessions_by_day": [
            {"day": "{} {:02d}".format(MONTHS[d // 15], (d % 15) + 1), "count": by_day.get(d, 0)}
            for d in range(DAYS)
        ],
        "top_searches": top(searches, "term"),
        "top_questions": top(questions, "question", 8),
        "top_products": top(products, "name"),
        "by_category": pct(cats),
        "by_device": pct(devices),
        "by_country": pct(countries),
        "funnel": [
            {"step": "Visits", "count": funnel["visit"]},
            {"step": "Engaged", "count": funnel["engaged"]},
            {"step": "Viewed a product", "count": funnel["viewed"]},
            {"step": "Added to bag", "count": funnel["cart"]},
            {"step": "Ordered", "count": funnel["order"]},
        ],
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote analytics for {} sessions to {}".format(N, os.path.relpath(OUT, PACK + "/..")))


if __name__ == "__main__":
    main()
