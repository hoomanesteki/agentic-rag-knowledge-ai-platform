"""M8.3 drift monitors: compare a reference window of traffic to a current one and flag drift.

Four named monitors:
- query-embedding drift: cosine distance between the reference and current query-embedding
  centroids (needs an embedder; the same one the app uses).
- retrieval-score distribution: PSI of the top vector-hit score per request (governed metric and
  graph blocks are excluded, since their score is a constant 1.0 that would mask a real drop).
- confidence distribution: PSI of the lexical-overlap gate confidence, read from the same field
  across answer paths so a change in brain or metric-traffic mix is not misread as drift.
- feedback rate: change in the thumbs-down rate.

The retrieval-score and confidence monitors are also stratified by language.

Everything is derived from the traces and feedback the app already writes, so drift is measured on
real traffic with no extra store. Trace fields are generic, so this stays domain agnostic.
"""
from __future__ import annotations

import math
from collections import defaultdict

_EPS = 1e-6
_PSI_THRESHOLD = 0.2       # >0.2 is a meaningful population shift
_EMBED_THRESHOLD = 0.1     # cosine distance between query centroids
_FEEDBACK_THRESHOLD = 0.1  # rise in thumbs-down rate


def _numeric(values: list) -> list:
    return [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def psi(reference: list, current: list, bins: int = 10) -> float | None:
    """Population stability index over quantile bins of the reference. None when there is too
    little data to bin."""
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < bins or not cur:
        return None
    ordered = sorted(ref)
    # a (near) constant reference gives collapsed quantile edges, so binning can never see a
    # downward shift; fall back to the fraction of current values that moved off the constant
    if len(set(ordered)) < 2:
        constant = ordered[0]
        tolerance = max(1e-6, 0.05 * abs(constant))
        return round(sum(1 for v in cur if abs(v - constant) > tolerance) / len(cur), 4)
    edges = [ordered[int(i * len(ordered) / bins)] for i in range(1, bins)]

    def distribution(values: list) -> list:
        counts = [0] * bins
        for value in values:
            b = 0
            while b < len(edges) and value > edges[b]:
                b += 1
            counts[b] += 1
        total = len(values) or 1
        return [max(c / total, _EPS) for c in counts]

    ref_dist, cur_dist = distribution(ref), distribution(cur)
    return round(sum((cur_dist[i] - ref_dist[i]) * math.log(cur_dist[i] / ref_dist[i])
                     for i in range(bins)), 4)


def _l2(vec: list) -> list:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _centroid(vectors: list) -> list:
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


_EMBED_DRIFT_CAP = 200  # embed at most this many queries per window


def _stride_sample(items: list, cap: int) -> list:
    """A deterministic, evenly-spread subsample so a large window doesn't embed thousands of queries
    (slow, metered, and near statistically inert as both centroids approach the global mean)."""
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def embedding_drift(ref_queries: list, cur_queries: list, embedder) -> float | None:
    """Cosine distance between the reference and current query-embedding centroids (0 identical).
    Returns None if there is nothing to compare, or if the embedder is unavailable (429/outage),
    so a monitoring caller sees 'unavailable' rather than a crash that looks like detected drift."""
    ref_queries = _stride_sample([q for q in ref_queries if q], _EMBED_DRIFT_CAP)
    cur_queries = _stride_sample([q for q in cur_queries if q], _EMBED_DRIFT_CAP)
    if not ref_queries or not cur_queries:
        return None
    try:
        ref = _l2(_centroid(embedder.embed(ref_queries, input_type="query")))
        cur = _l2(_centroid(embedder.embed(cur_queries, input_type="query")))
    except Exception:  # embedder rate-limited or down: this monitor degrades, it does not crash
        return None
    return round(1 - sum(a * b for a, b in zip(ref, cur)), 4)


def _top_score(trace: dict) -> float | None:
    # top vector-hit score, excluding injected metric/graph blocks whose score is a constant 1.0
    scores = [r.get("score") for r in (trace.get("retrieved") or [])
              if isinstance(r.get("score"), (int, float))
              and not str(r.get("id", "")).startswith(("metric:", "graph:"))]
    return max(scores) if scores else None


def _gate_confidence(trace: dict):
    # the lexical-overlap confidence, consistent across paths (the agent path also traces it as
    # overlap_confidence; a metric/graph specialist's pinned 1.0 is not this signal)
    return trace.get("overlap_confidence", trace.get("confidence"))


def _down_rate(feedback: list) -> float | None:
    verdicts = [f.get("verdict") for f in feedback if f.get("verdict") in ("up", "down")]
    return round(verdicts.count("down") / len(verdicts), 3) if verdicts else None


def _psi_monitor(reference: list, current: list, threshold: float) -> dict:
    value = psi(reference, current)
    return {"psi": value, "drift": value is not None and value > threshold}


def _numeric_monitors(reference: list, current: list) -> dict:
    return {
        "retrieval_score": _psi_monitor(
            [_top_score(t) for t in reference], [_top_score(t) for t in current], _PSI_THRESHOLD),
        "confidence": _psi_monitor(
            [_gate_confidence(t) for t in reference], [_gate_confidence(t) for t in current],
            _PSI_THRESHOLD),
    }


def drift_report(reference: list, current: list, *, feedback_ref: list | None = None,
                 feedback_cur: list | None = None, embedder=None) -> dict:
    """Overall and per-language drift across the four monitors, with a single `drifted` flag."""
    monitors = _numeric_monitors(reference, current)
    if embedder is not None:
        distance = embedding_drift([t.get("query", "") for t in reference],
                                   [t.get("query", "") for t in current], embedder)
        monitors["query_embedding"] = {
            "distance": distance,
            "drift": distance is not None and distance > _EMBED_THRESHOLD}
    ref_rate, cur_rate = _down_rate(feedback_ref or []), _down_rate(feedback_cur or [])
    rose = (ref_rate is not None and cur_rate is not None
            and cur_rate - ref_rate > _FEEDBACK_THRESHOLD)
    monitors["feedback_rate"] = {"reference": ref_rate, "current": cur_rate, "drift": rose}

    by_language = {}
    ref_by_lang: dict = defaultdict(list)
    cur_by_lang: dict = defaultdict(list)
    for trace in reference:
        ref_by_lang[trace.get("lang") or "unknown"].append(trace)
    for trace in current:
        cur_by_lang[trace.get("lang") or "unknown"].append(trace)
    for lang in sorted(set(ref_by_lang) | set(cur_by_lang)):
        by_language[lang] = _numeric_monitors(ref_by_lang.get(lang, []), cur_by_lang.get(lang, []))

    # drift confined to one language must still raise the flag, not average out overall
    drifted = (any(m.get("drift") for m in monitors.values())
               or any(m.get("drift") for lang in by_language.values() for m in lang.values()))
    return {"monitors": monitors, "by_language": by_language, "drifted": drifted}
