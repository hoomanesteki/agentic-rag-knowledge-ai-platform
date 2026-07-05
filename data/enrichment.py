"""Batch enrichment: turn high-churn, untrusted reviews into governed product features.

Product descriptions are authored, stable, and trusted, so they are enriched once at ingest (and
re-run only when the text changes, keyed on a content hash). Reviews are user generated and must
never be trusted one at a time, so their signal is computed here by CONSENSUS: an aspect value
(like fit "runs small") is promoted to a product feature only when enough independent reviews agree,
and it carries a confidence and its provenance (the review ids, the annotator, the date). This is
the voting idea, many weak, noisy annotations vote and only the agreed signal is served.

The result lands in the DuckDB lakehouse as product_features, so the app reads a precomputed,
auditable feature instead of re-reading raw reviews at query time, which is what makes this get
cheaper and faster at scale. Why batch and not live: one new review must not flip a product's fit,
consensus needs a window, and an idempotent batch job is easy to test and govern. A live
moderation, PII, and injection gate at submit time is a separate, cheap check (see the ADR), not
this job. The annotator is pluggable: a deterministic keyword pass by default (cheap, reproducible)
or an LLM annotator; the consensus logic is identical either way.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

# Deterministic fit rules over the phrasings reviewers actually use. The order matters: a review
# that says "sized up because it runs small" is a runs_small signal, checked before true_to_size.
_FIT_RULES = [
    ("runs_small", re.compile(
        r"\b(runs? small|sized? up|size up|too small|tight|snug|smaller than|"
        r"size(d)? up a)\b", re.I)),
    ("runs_large", re.compile(
        r"\b(runs? large|runs? big|sized? down|size down|too big|too large|"
        r"oversized|baggy|bigger than)\b", re.I)),
    ("true_to_size", re.compile(
        r"\b(true to size|tts|fits? (perfectly|great|spot on|as expected)|"
        r"fit is (spot on|perfect|great)|perfect fit|as expected)\b", re.I)),
]


# A negation shortly before a fit phrase flips its meaning ("does not run small"), so a negated
# match is skipped rather than annotated as the opposite.
_NEG = re.compile(r"\b(not|isn'?t|wasn'?t|doesn'?t|don'?t|didn'?t|never|no)\b", re.I)

# The only aspect/value pairs a feature may carry. An LLM annotator whose output is steered by
# injected review text cannot land an arbitrary string in the served features: consensus() drops
# anything not in this allowlist.
ALLOWED = {"fit": {"runs_small", "runs_large", "true_to_size"}}


def keyword_annotator(review_text: str) -> dict | None:
    """A cheap, deterministic first-pass annotator: the fit signal a review expresses, or None. It
    returns the first NON-negated match, so "does not run small, true to size" reads as
    true_to_size, not runs_small. In production an LLM annotator plugs into this slot unchanged."""
    text = review_text or ""
    for value, rx in _FIT_RULES:
        m = rx.search(text)
        if m and not _NEG.search(text[max(0, m.start() - 24):m.start()]):
            return {"aspect": "fit", "value": value}
    return None


def consensus(reviews, annotate=keyword_annotator, *, min_support: int = 2,
              min_ratio: float = 0.5) -> list[dict]:
    """Annotate each review, group votes by (product, aspect), and keep the winning value only when
    at least min_support reviews back it and they are at least min_ratio of the reviews that spoke
    to that aspect. Each kept feature carries its confidence, its support, and the source ids."""
    votes: dict = defaultdict(lambda: defaultdict(list))  # product -> aspect -> [(value, rev_id)]
    seen_ids: set = set()
    for r in reviews:
        rid = r.get("id")
        if rid is not None:  # a re-ingested or repeat-posted review must not double-vote
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
        a = annotate(r.get("text", ""))
        if not a:
            continue
        aspect, value = a.get("aspect"), a.get("value")
        if value not in ALLOWED.get(aspect, set()):
            continue  # only allowlisted pairs, so an injected annotation cannot reach a feature
        votes[r.get("product_id")][aspect].append((value, rid))
    rows = []
    for product, aspects in votes.items():
        if product is None:
            continue
        for aspect, vs in aspects.items():
            total = len(vs)
            value, support = Counter(v for v, _ in vs).most_common(1)[0]
            ratio = support / total if total else 0.0
            # strict majority: a tie (0.5) does not win, so an order-dependent Counter tie-break
            # can never promote a value that only half the reviews support
            if support >= min_support and ratio > min_ratio:
                rows.append({
                    "product_id": product, "aspect": aspect, "value": value,
                    "confidence": round(ratio, 3), "support": support, "total": total,
                    "sources": [rid for v, rid in vs if v == value],
                })
    return sorted(rows, key=lambda r: (-r["support"], r["product_id"]))


_SCHEMA = ("CREATE TABLE IF NOT EXISTS product_features ("
           "product_id VARCHAR, aspect VARCHAR, value VARCHAR, confidence DOUBLE, "
           "support INTEGER, total INTEGER, sources VARCHAR, annotator VARCHAR, "
           "computed_at VARCHAR)")


def write_features(rows: list[dict], db_path: str, *, computed_at: str,
                   annotator: str = "keyword") -> int:
    """Write the feature rows to the DuckDB lakehouse, idempotently: this annotator's previous rows
    are replaced, so re-running the batch converges instead of accumulating duplicates."""
    import duckdb
    con = duckdb.connect(db_path)
    try:
        con.execute(_SCHEMA)
        con.execute("DELETE FROM product_features WHERE annotator = ?", [annotator])
        con.executemany(
            "INSERT INTO product_features VALUES (?,?,?,?,?,?,?,?,?)",
            [[r["product_id"], r["aspect"], r["value"], r["confidence"], r["support"], r["total"],
              json.dumps(r["sources"]), annotator, computed_at] for r in rows])
    finally:
        con.close()
    return len(rows)
