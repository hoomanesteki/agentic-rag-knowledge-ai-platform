#!/usr/bin/env python3
"""Generate the apparel catalog: products.csv (structured) and products_catalog.jsonl (marketing
copy). Deterministic (seeded), so re-running reproduces the same catalog.

Lives inside the domain pack, not scripts/, so the brand vocabulary here does not trip the engine
leak linter. The first 20 products (P001-P020) are preserved exactly, because reviews, sales, and
the eval golden reference them by id, price, and name.

Run: python domains/apparel_ecommerce/tools/gen_catalog.py
"""
from __future__ import annotations

import csv
import json
import os
import random

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
STRUCT = os.path.join(PACK, "seed", "structured")
UNSTRUCT = os.path.join(PACK, "seed", "unstructured")

# product_id, name, category, gender, size, price, color, colors, weather, stock, supplier_id.
# Colors is a pipe list of available colorways; the store shows swatches, the row keeps a primary.
COLUMNS = ["product_id", "name", "category", "gender", "size", "price", "color", "colors",
           "weather", "stock", "supplier_id"]

# The original 20, kept exactly (ids, names, prices, sizes) with the new columns filled in.
EXISTING = [
    ("P001", "Aster Flow Legging", "leggings", "women", "S", 98.0, "Black", "Black|Storm Blue|Deep Plum", "all-season", 42, "SUP01"),
    ("P002", "Aster Flow Legging", "leggings", "women", "M", 98.0, "Black", "Black|Storm Blue|Deep Plum", "all-season", 55, "SUP01"),
    ("P003", "Aster Flow Legging", "leggings", "women", "L", 98.0, "Black", "Black|Storm Blue|Deep Plum", "all-season", 37, "SUP01"),
    ("P004", "Aster Vent Tech Tee", "tops", "women", "M", 78.0, "Slate", "Slate|White|Ember Red", "hot", 61, "SUP02"),
    ("P005", "Aster Vent Tech Tee", "tops", "women", "L", 78.0, "Slate", "Slate|White|Ember Red", "hot", 44, "SUP02"),
    ("P006", "Aster Daytrip Belt Bag", "bags", "unisex", "OS", 38.0, "Black", "Black|Sand|Olive", "all-season", 80, "SUP01"),
    ("P007", "Aster Cloud Hoodie", "hoodies", "women", "M", 118.0, "Heather Grey", "Heather Grey|Black|Oatmeal", "mild", 33, "SUP01"),
    ("P008", "Aster Cloud Hoodie", "hoodies", "women", "L", 118.0, "Heather Grey", "Heather Grey|Black|Oatmeal", "mild", 21, "SUP01"),
    ("P009", "Aster Storm Shell Jacket", "jackets", "women", "S", 178.0, "Storm Blue", "Storm Blue|Black|Forest", "rain", 18, "SUP02"),
    ("P010", "Aster Storm Shell Jacket", "jackets", "women", "M", 178.0, "Storm Blue", "Storm Blue|Black|Forest", "rain", 26, "SUP02"),
    ("P011", "Aster Storm Shell Jacket", "jackets", "women", "L", 178.0, "Storm Blue", "Storm Blue|Black|Forest", "rain", 14, "SUP02"),
    ("P012", "Aster Trailhead Puffer", "jackets", "women", "M", 228.0, "Black", "Black|Sand|Navy", "cold", 12, "SUP02"),
    ("P013", "Aster Trailhead Puffer", "jackets", "women", "L", 228.0, "Black", "Black|Sand|Navy", "cold", 9, "SUP02"),
    ("P014", "Aster Base Merino Long Sleeve", "tops", "women", "M", 88.0, "Oatmeal", "Oatmeal|Black|Forest", "cold", 30, "SUP02"),
    ("P015", "Aster Base Merino Long Sleeve", "tops", "women", "L", 88.0, "Oatmeal", "Oatmeal|Black|Forest", "cold", 24, "SUP02"),
    ("P016", "Aster Studio Sports Bra", "bras", "women", "M", 58.0, "Black", "Black|Slate|Deep Plum", "hot", 48, "SUP01"),
    ("P017", "Aster Momentum Short", "shorts", "women", "M", 68.0, "Black", "Black|Navy|Sand", "hot", 52, "SUP01"),
    ("P018", "Aster Everywhere Jogger", "bottoms", "women", "M", 108.0, "Charcoal", "Charcoal|Black|Oatmeal", "mild", 40, "SUP01"),
    ("P019", "Aster Commute Tote", "bags", "unisex", "OS", 128.0, "Black", "Black|Charcoal|Sand", "all-season", 29, "SUP02"),
    ("P020", "Aster Peak Beanie", "accessories", "unisex", "OS", 38.0, "Charcoal", "Charcoal|Black|Oatmeal", "cold", 70, "SUP01"),
]

# category -> (noun choices, size set, price band, default weather rotation, material)
CATS = {
    "leggings":    (["Legging", "Tight", "Crop Legging"], ["XS", "S", "M", "L", "XL"], (88, 128), ["all-season", "hot", "mild"], "four-way-stretch"),
    "bras":        (["Sports Bra", "Bra", "Longline Bra"], ["XS", "S", "M", "L", "XL"], (48, 72), ["hot", "all-season"], "moisture-wicking"),
    "tops":        (["Tee", "Tank", "Long Sleeve", "Training Top"], ["XS", "S", "M", "L", "XL"], (48, 98), ["hot", "mild", "cold"], "breathable knit"),
    "jackets":     (["Jacket", "Shell", "Puffer", "Parka", "Windbreaker"], ["S", "M", "L", "XL"], (148, 268), ["rain", "cold", "winter", "mild"], "technical"),
    "hoodies":     (["Hoodie", "Zip Hoodie", "Pullover"], ["S", "M", "L", "XL"], (98, 138), ["mild", "cold"], "brushed fleece"),
    "shorts":      (["Short", "Lined Short", "Bike Short"], ["S", "M", "L", "XL"], (58, 82), ["hot", "mild"], "quick-dry"),
    "bottoms":     (["Jogger", "Pant", "Track Pant"], ["S", "M", "L", "XL"], (98, 138), ["mild", "cold", "all-season"], "travel knit"),
    "bags":        (["Backpack", "Duffel", "Sling", "Tote", "Belt Bag"], ["OS"], (38, 158), ["all-season"], "water-resistant"),
    "accessories": (["Beanie", "Cap", "Socks", "Gloves", "Headband", "Scarf"], ["OS"], (18, 48), ["cold", "all-season"], "soft-knit"),
}

# distinct, on-brand style words; combined with a category noun to make unique product names
ADJ = ["Storm", "Trailhead", "Alpine", "Summit", "Ridgeline", "Tempo", "Cloud", "Frost", "Nimbus",
       "Ember", "Vent", "Flow", "Base", "Pace", "Drift", "Aspen", "Coastal", "Meridian", "Boreal",
       "Solstice", "Zephyr", "Cedar", "Glacier", "Horizon", "Rally", "Cadence", "Torrent", "Dune",
       "Vertex", "Aurora", "Kodiak", "Marin", "Slate", "Lumen", "Trace", "Crest", "Field", "Halo"]

PALETTE = ["Black", "White", "Storm Blue", "Slate", "Charcoal", "Heather Grey", "Oatmeal", "Sand",
           "Olive", "Navy", "Deep Plum", "Forest", "Ember Red", "Sky", "Clay"]

WEATHER_USE = {
    "rain": "wet commutes and rainy-city days",
    "cold": "cold, dry winter days and layering",
    "winter": "deep winter and snow",
    "hot": "hot workouts, summer runs, and studio heat",
    "mild": "shoulder-season days and everyday wear",
    "all-season": "year-round training and travel",
}
FIT = ["Runs true to size", "Relaxed, roomy fit, size down if you like it trim",
       "Compressive and supportive", "Slim athletic fit", "Soft and stretchy with a stay-put waistband"]


def gen():
    rows = list(EXISTING)
    docs = []
    pid = 21
    used_names = {r[1] for r in EXISTING}
    buckets = ([("women", c) for c in ["leggings", "bras", "tops", "jackets", "hoodies", "shorts", "bottoms"]]
               + [("men", c) for c in ["tops", "jackets", "hoodies", "shorts", "bottoms"]]
               + [("unisex", c) for c in ["bags", "accessories"]])
    adj_pool = ADJ[:]
    random.shuffle(adj_pool)
    ai = 0
    for gender, cat in buckets:
        nouns, sizes, (lo, hi), weathers, material = CATS[cat]
        for n in range(10):  # at least ten per category per gender
            noun = nouns[n % len(nouns)]
            # find a unique "Aster <Adj> <Noun>"
            name = None
            for _ in range(len(adj_pool)):
                cand = "Aster {} {}".format(adj_pool[ai % len(adj_pool)], noun)
                ai += 1
                if cand not in used_names:
                    name = cand
                    break
            if name is None:
                name = "Aster {} {} {}".format(adj_pool[ai % len(adj_pool)], noun, pid)
                ai += 1
            used_names.add(name)
            price = float(random.randint(lo // 2, hi // 2) * 2)  # even prices in band
            weather = weathers[n % len(weathers)]
            colors = random.sample(PALETTE, 3)
            primary = colors[0]
            stock_base = random.choice([0, 4, 9] + list(range(12, 90)))  # some low / out of stock
            supplier = random.choice(["SUP01", "SUP02"])
            first_pid = pid  # the description references the product's first variant
            for size in sizes:  # each size variant is its own row with a unique product_id
                rows.append((("P%03d" % pid), name, cat, gender, size, price, primary,
                             "|".join(colors), weather, max(0, stock_base + random.randint(-6, 6)),
                             supplier))
                pid += 1
            use = WEATHER_USE.get(weather, "everyday wear")
            g = {"women": "women's", "men": "men's", "unisex": "unisex"}[gender]
            text = ("The {name} is a {g} {mat} {cat} built for {use}. {fit}. {price:.0f} dollars in "
                    "{primary}, also in {alt}. Sizes {sizes}.").format(
                        name=name, g=g, mat=material, cat=cat.rstrip("s"), use=use,
                        fit=random.choice(FIT), price=price, primary=primary,
                        alt=", ".join(colors[1:]), sizes="/".join(sizes))
            docs.append({"id": "PC%03d" % first_pid, "lang": "en", "product_id": "P%03d" % first_pid,
                         "category": cat, "gender": gender, "text": text})
    return rows, docs


_SIZE_ORDER = ["XS", "S", "M", "L", "XL", "OS"]
_GWORD = {"women": "women's", "men": "men's", "unisex": "unisex"}


def summaries(rows):
    """Per gender+category and per category facts (counts, sizes, colors, price range, examples), so
    the assistant can answer 'what bags do you have for women', 'price range for hoodies', 'what
    sizes / colors', which are catalog aggregates that plain description retrieval misses."""
    from collections import defaultdict

    groups = defaultdict(lambda: {"names": {}, "sizes": set(), "colors": set(), "prices": []})
    for r in rows:
        g = groups[(r[3], r[2])]  # (gender, category)
        g["names"][r[1]] = r[5]   # name -> price
        g["sizes"].add(r[4])
        g["colors"].update(c for c in r[7].split("|") if c)
        g["prices"].append(r[5])

    facts, i = [], 1
    for (gender, cat), g in sorted(groups.items()):
        names = sorted(g["names"])
        sizes = [s for s in _SIZE_ORDER if s in g["sizes"]]
        colors = sorted(g["colors"])
        examples = ", ".join("{} (${:.0f})".format(n.replace("Aster ", ""), g["names"][n])
                             for n in names[:6])
        text = ("Aster {gw} {cat}: {n} styles, in sizes {sizes}, in colors including {colors}. "
                "Prices range from ${lo:.0f} to ${hi:.0f}. Styles include {ex}.").format(
                    gw=_GWORD[gender], cat=cat, n=len(names), sizes="/".join(sizes),
                    colors=", ".join(colors[:8]), lo=min(g["prices"]), hi=max(g["prices"]), ex=examples)
        facts.append({"id": "CAT%03d" % i, "lang": "en", "category": cat, "gender": gender, "text": text})
        i += 1

    bycat = defaultdict(list)
    for (gender, cat), g in groups.items():
        bycat[cat] += g["prices"]
    for cat, prices in sorted(bycat.items()):
        facts.append({"id": "CATR-%s" % cat, "lang": "en", "category": cat,
                      "text": "Aster {cat} range in price from ${lo:.0f} to ${hi:.0f} across "
                              "women's and men's styles.".format(cat=cat, lo=min(prices), hi=max(prices))})
    return facts


def main():
    rows, docs = gen()
    with open(os.path.join(STRUCT, "products.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    with open(os.path.join(UNSTRUCT, "products_catalog.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    facts = summaries(rows)
    with open(os.path.join(UNSTRUCT, "catalog_facts.jsonl"), "w") as f:
        for d in facts:
            f.write(json.dumps(d) + "\n")
    products = len({r[1] for r in rows})
    print("wrote {} rows ({} products) to products.csv".format(len(rows), products))
    print("wrote {} descriptions to products_catalog.jsonl".format(len(docs)))
    print("wrote {} category facts to catalog_facts.jsonl".format(len(facts)))


if __name__ == "__main__":
    main()
