"""Run the golden set through retrieval and the abstain gate, and produce a scorecard.

What is measured here (no LLM):
- Retrieval quality on questions the current index can actually answer. At M2 only the review
  vector index exists, so this is the qualitative subset; factual/relational/analytical
  questions are reported as deferred (they need the metric layer at M4 and the graph at M5).
- The gate: abstain recall on unanswerable/out-of-domain questions, and the false-abstain
  rate on measurable answerable questions (so an always-abstain gate cannot look perfect).

Each question is retrieved exactly once, then aggregated overall and by language, so a real
run makes the minimum number of paid embed/rerank calls. Answer quality (RAGAS) is M8.
"""
from __future__ import annotations

from adapters.base import Embedder, HybridStore, Reranker
from evaluation.metrics import hit_at_k, mean, reciprocal_rank
from pipeline.answer import DEFAULT_MIN_CONFIDENCE, retrieve, should_abstain

_ABSTAIN_TYPES = ("unanswerable", "out_of_domain")
_RETRIEVABLE_ROUTES = ("qualitative",)  # what the review vector index can answer at M2


def _partition(golden: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    measurable, deferred, abstain = [], [], []
    for g in golden:
        t = g.get("type")
        if t in _ABSTAIN_TYPES:
            abstain.append(g)
        elif t == "answerable":
            if g.get("route") in _RETRIEVABLE_ROUTES and g.get("expected_entities"):
                measurable.append(g)
            else:
                deferred.append(g)
    return measurable, deferred, abstain


def _relevance_flags(hits: list[dict], expected: list, entity_fields: list[str]) -> list[bool]:
    wanted = {str(e) for e in expected}
    flags = []
    for h in hits:
        payload = h.get("payload") or {}
        got = {str(payload.get(f)) for f in entity_fields if payload.get(f) is not None}
        flags.append(bool(got & wanted))
    return flags


def _entity_recall(hits: list[dict], expected: list, entity_fields: list[str]) -> float:
    if not expected:
        return 0.0
    wanted = {str(e) for e in expected}
    covered = set()
    for h in hits:
        payload = h.get("payload") or {}
        for f in entity_fields:
            v = payload.get(f)
            if v is not None and str(v) in wanted:
                covered.add(str(v))
    return len(covered) / len(wanted)


def _contexts(hits: list[dict]) -> list[dict]:
    return [{"text": (h.get("payload") or {}).get("text", "")} for h in hits]


def _score_retrieval(g, embedder, store, top_k, top_k_in, entity_fields, min_confidence,
                     reranker, dense_only) -> dict:
    hits = retrieve(g["question"], embedder, store, top_k, reranker=reranker,
                    top_k_in=top_k_in, dense_only=dense_only)
    flags = _relevance_flags(hits, g["expected_entities"], entity_fields)
    abstained, _ = should_abstain(g["question"], _contexts(hits), min_confidence)
    return {
        "lang": g.get("lang", "unknown"),
        "hit": hit_at_k(flags),
        "rr": reciprocal_rank(flags),
        "erecall": _entity_recall(hits, g["expected_entities"], entity_fields),
        "false_abstain": 1.0 if abstained else 0.0,
        "nhits": len(hits),
    }


def _score_abstain(g, embedder, store, top_k, top_k_in, min_confidence, reranker,
                   dense_only) -> dict:
    hits = retrieve(g["question"], embedder, store, top_k, reranker=reranker,
                    top_k_in=top_k_in, dense_only=dense_only)
    abstained, _ = should_abstain(g["question"], _contexts(hits), min_confidence)
    return {"lang": g.get("lang", "unknown"), "abstained": 1.0 if abstained else 0.0}


def _agg_retrieval(records: list[dict]) -> dict:
    return {
        "hit_rate_at_k": round(mean([r["hit"] for r in records]), 3),
        "entity_recall_at_k": round(mean([r["erecall"] for r in records]), 3),
        "mrr": round(mean([r["rr"] for r in records]), 3),
        "false_abstain_rate": round(mean([r["false_abstain"] for r in records]), 3),
        "n": len(records),
        "total_hits": sum(r["nhits"] for r in records),
    }


def _agg_gate(records: list[dict]) -> dict:
    return {"abstain_recall": round(mean([r["abstained"] for r in records]), 3), "n": len(records)}


def _route_counts(items: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for g in items:
        route = g.get("route") or "unspecified"
        counts[route] = counts.get(route, 0) + 1
    return counts


def evaluate(golden: list[dict], *, embedder: Embedder, store: HybridStore,
             entity_fields: list[str], reranker: Reranker | None = None, top_k: int = 8,
             top_k_in: int = 50, min_confidence: float = DEFAULT_MIN_CONFIDENCE,
             dense_only: bool = False) -> dict:
    measurable, deferred, abstain = _partition(golden)
    m_records = [_score_retrieval(g, embedder, store, top_k, top_k_in, entity_fields,
                                  min_confidence, reranker, dense_only) for g in measurable]
    a_records = [_score_abstain(g, embedder, store, top_k, top_k_in, min_confidence, reranker,
                                dense_only) for g in abstain]
    overall = _agg_retrieval(m_records)
    scorecard = {
        "top_k": top_k,
        "top_k_in": top_k_in,
        "dense_only": dense_only,
        "reranked": reranker is not None,
        "coverage": {"measured": len(measurable), "deferred": len(deferred),
                     "deferred_by_route": _route_counts(deferred), "abstain_set": len(abstain)},
        "degenerate": bool(m_records) and overall["total_hits"] == 0,
        "overall": {"retrieval": overall, "gate": _agg_gate(a_records)},
        "by_language": {},
    }
    for lang in sorted({r["lang"] for r in m_records} | {r["lang"] for r in a_records}):
        scorecard["by_language"][lang] = {
            "retrieval": _agg_retrieval([r for r in m_records if r["lang"] == lang]),
            "gate": _agg_gate([r for r in a_records if r["lang"] == lang]),
        }
    return scorecard


def _fmt(value, n: int) -> str:
    return "n/a" if n == 0 else "{:.3f}".format(value)


def format_scorecard(scorecard: dict, label: str = "current") -> str:
    cov = scorecard["coverage"]
    lines = [
        "Eval scorecard ({}), top_k={}, top_k_in={}, reranked={}".format(
            label, scorecard["top_k"], scorecard["top_k_in"], scorecard.get("reranked", False)),
        "coverage: measured {}, deferred {} {}, abstain-set {}".format(
            cov["measured"], cov["deferred"], cov["deferred_by_route"], cov["abstain_set"]),
    ]
    if scorecard["degenerate"]:
        lines.append("WARNING: no query returned any hit. Is the index empty (run make ingest)?")
    lines.append("")

    def row(name, r, g):
        return ("  {:<9} hit@k={:<6} entity_recall@k={:<6} mrr={:<6} false_abstain={:<6} "
                "(n={})  abstain_recall={:<6} (n={})".format(
                    name, _fmt(r["hit_rate_at_k"], r["n"]), _fmt(r["entity_recall_at_k"], r["n"]),
                    _fmt(r["mrr"], r["n"]), _fmt(r["false_abstain_rate"], r["n"]), r["n"],
                    _fmt(g["abstain_recall"], g["n"]), g["n"]))

    o = scorecard["overall"]
    lines.append(row("overall", o["retrieval"], o["gate"]))
    for lang, s in scorecard["by_language"].items():
        lines.append(row(lang, s["retrieval"], s["gate"]))
    return "\n".join(lines)
