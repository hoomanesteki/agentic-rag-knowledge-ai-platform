"""Run the golden set through retrieval and the abstain gate, and produce a scorecard.

What is measured here (no LLM):
- Retrieval quality on questions the current index can actually answer. At M2 only the review
  vector index exists, so this is the qualitative subset; factual/relational/analytical
  questions are reported as deferred (they need the metric layer at M4 and the graph at M5).
- The gate: abstain recall on unanswerable/out-of-domain questions, and the false-abstain
  rate on measurable answerable questions (so an always-abstain gate cannot look perfect).

Relevance is decided only against the entity fields the domain manifest declares (entity_ref),
not against every payload value. Answer quality (RAGAS) comes at M8.
"""
from __future__ import annotations

from adapters.base import Embedder, HybridStore
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


def _retrieval_block(items, embedder, store, top_k, entity_fields, min_confidence):
    hit_rates, entity_recalls, rrs, false_abstains = [], [], [], []
    total_hits = 0
    for g in items:
        hits = retrieve(g["question"], embedder, store, top_k)
        total_hits += len(hits)
        flags = _relevance_flags(hits, g["expected_entities"], entity_fields)
        hit_rates.append(hit_at_k(flags))
        rrs.append(reciprocal_rank(flags))
        entity_recalls.append(_entity_recall(hits, g["expected_entities"], entity_fields))
        abstained, _ = should_abstain(g["question"], _contexts(hits), min_confidence)
        false_abstains.append(1.0 if abstained else 0.0)
    return {
        "hit_rate_at_k": round(mean(hit_rates), 3),
        "entity_recall_at_k": round(mean(entity_recalls), 3),
        "mrr": round(mean(rrs), 3),
        "false_abstain_rate": round(mean(false_abstains), 3),
        "n": len(items),
        "total_hits": total_hits,
    }


def _gate_block(items, embedder, store, top_k, min_confidence):
    abstained_flags = []
    for g in items:
        hits = retrieve(g["question"], embedder, store, top_k)
        abstained, _ = should_abstain(g["question"], _contexts(hits), min_confidence)
        abstained_flags.append(1.0 if abstained else 0.0)
    return {"abstain_recall": round(mean(abstained_flags), 3), "n": len(items)}


def _route_counts(items: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for g in items:
        route = g.get("route") or "unspecified"
        counts[route] = counts.get(route, 0) + 1
    return counts


def evaluate(golden: list[dict], *, embedder: Embedder, store: HybridStore,
             entity_fields: list[str], top_k: int = 8,
             min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
    measurable, deferred, abstain = _partition(golden)
    overall_retrieval = _retrieval_block(
        measurable, embedder, store, top_k, entity_fields, min_confidence)
    scorecard = {
        "top_k": top_k,
        "coverage": {"measured": len(measurable), "deferred": len(deferred),
                     "deferred_by_route": _route_counts(deferred), "abstain_set": len(abstain)},
        "degenerate": bool(measurable) and overall_retrieval["total_hits"] == 0,
        "overall": {
            "retrieval": overall_retrieval,
            "gate": _gate_block(abstain, embedder, store, top_k, min_confidence),
        },
        "by_language": {},
    }
    for lang in sorted({g.get("lang", "unknown") for g in golden}):
        m = [g for g in measurable if g.get("lang", "unknown") == lang]
        a = [g for g in abstain if g.get("lang", "unknown") == lang]
        scorecard["by_language"][lang] = {
            "retrieval": _retrieval_block(m, embedder, store, top_k, entity_fields, min_confidence),
            "gate": _gate_block(a, embedder, store, top_k, min_confidence),
        }
    return scorecard


def _fmt(value, n: int) -> str:
    return "n/a" if n == 0 else "{:.3f}".format(value)


def format_scorecard(scorecard: dict, label: str = "current") -> str:
    cov = scorecard["coverage"]
    lines = [
        "Eval scorecard ({}), top_k={}".format(label, scorecard["top_k"]),
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
