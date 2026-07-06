"""Run the batch review-enrichment pipeline and write the governed feature table.

  PYTHONPATH=. uv run python scripts/run_enrichment.py

Loads the seed reviews, computes fit features by consensus (a value is kept only when enough
reviews agree), writes them to the DuckDB feature table, and writes a reproducible JSON artifact at
data/product_features.json. computed_at is taken from the data (the latest review date), not the
wall clock, so the artifact is byte-stable across runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from data.enrichment import consensus, keyword_annotator, write_features

_REVIEWS = ["domains/apparel_ecommerce/seed/unstructured/reviews.jsonl",
            "domains/apparel_ecommerce/seed/unstructured/reviews_generated.jsonl"]
_REPORT = "data/product_features.json"
_DB = "data/enrichment.duckdb"


def load_reviews(paths: list[str]) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        if os.path.exists(p):
            out += [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_DB)
    ap.add_argument("--min-support", type=int, default=2)
    args = ap.parse_args()

    reviews = load_reviews(_REVIEWS)
    computed_at = max((r.get("date", "") for r in reviews), default="")
    rows = consensus(reviews, keyword_annotator, min_support=args.min_support)
    n = write_features(rows, args.db, computed_at=computed_at, annotator="keyword")

    with open(_REPORT, "w", encoding="utf-8") as f:
        json.dump({"reviews": len(reviews), "computed_at": computed_at, "features": rows}, f,
                  indent=2)

    by_value = Counter(r["value"] for r in rows)
    print("enriched {} reviews -> {} governed product features".format(len(reviews), n))
    print("  by value:", dict(by_value))
    print("  sample runs_small features (product, confidence, support/total, sources):")
    for r in [x for x in rows if x["value"] == "runs_small"][:8]:
        print("    {}  conf={}  {}/{}  {}".format(
            r["product_id"], r["confidence"], r["support"], r["total"], r["sources"][:3]))
    print("wrote", _REPORT, "and", args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
