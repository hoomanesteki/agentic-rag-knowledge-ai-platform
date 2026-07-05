"""Route number questions to the governed metric layer via LLM slot-filling.

A cheap lexical pre-gate decides whether a query even looks like a metric question, so the
extra LLM slot-fill call is paid only when it might pay off (set METRIC_ROUTER_ALWAYS=1 to
force it, e.g. for recall-sensitive evals). The model then picks at most one metric and its
params (structured JSON); the resolver runs it read-only; the result becomes a labeled
evidence block for the answer prompt, never fused with vector hits.
"""
from __future__ import annotations

import json
import logging
import os
import re

from data.metrics import MetricResolver, MetricResult

_log = logging.getLogger("skein.metric_router")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"by", "of", "the", "per", "a", "an"}
# Bare "how" is deliberately excluded: it fires on every "how do I..." support question and
# wastes a slot-fill call. "how many"/"how much" still trip the gate via "many"/"much".
_NUMERIC_CUES = {"many", "much", "rate", "count", "average", "avg", "total",
                 "number", "percent", "percentage",
                 # price superlatives, so "cheapest <category>" reaches the price metric
                 "cheapest", "cheap", "priciest", "expensive", "price", "budget", "affordable"}
_SLOT_SYSTEM = (
    "You map a user question to at most one governed metric. Reply with ONLY a JSON object: "
    '{"metric": <name or null>, "params": {<name>: <value>}}. Pick a metric only if the '
    "question asks for exactly that number; otherwise use null. Use only the listed params."
)


def _vocab(resolver: MetricResolver) -> set[str]:
    vocab: set[str] = set()
    for name, spec in resolver.metrics.items():
        vocab.update(_WORD.findall(name.lower()))
        for dim in spec.get("dimensions", []) or []:
            vocab.update(_WORD.findall(str(dim).lower()))
        for param in (spec.get("params", {}) or {}):
            vocab.update(_WORD.findall(str(param).lower()))
    return (vocab | _NUMERIC_CUES) - _STOP


def _looks_like_metric(query: str, resolver: MetricResolver) -> bool:
    if os.getenv("METRIC_ROUTER_ALWAYS"):
        return True
    return bool(set(_WORD.findall(query.lower())) & _vocab(resolver))


def _extract_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])  # first complete object, ignore prose
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _slot_prompt(query: str, resolver: MetricResolver) -> str:
    lines = ["Available metrics:"]
    for name, spec in resolver.metrics.items():
        lines.append("- {} params={} dimensions={}".format(
            name, dict(spec.get("params", {}) or {}), spec.get("dimensions", []) or []))
    lines += ["", "Question: " + query, "JSON:"]
    return "\n".join(lines)


def route_metric(query: str, llm, resolver: MetricResolver) -> MetricResult | None:
    if not resolver.metrics or not os.path.exists(resolver.db_path):
        return None
    if not _looks_like_metric(query, resolver):
        return None
    # Slot-filling a metric name and params from a fixed list is constrained extraction the cheap
    # model handles well, and the result is validated below (the metric must exist, params are
    # filtered, resolve() can still reject), so classification does not pay for the large model.
    # getattr picks the ResilientLLM's small fallback when present; a bare llm is used as-is.
    clf = getattr(llm, "fallback", None) or llm
    raw = clf.generate(_slot_prompt(query, resolver), system=_SLOT_SYSTEM, max_tokens=200).text
    parsed = _extract_json(raw)
    if not parsed:
        return None
    name = parsed.get("metric")
    if not isinstance(name, str) or name not in resolver.metrics:
        return None
    declared = resolver.metrics[name].get("params", {}) or {}
    raw_params = parsed.get("params")
    params = {k: v for k, v in raw_params.items() if k in declared} \
        if isinstance(raw_params, dict) else {}
    try:
        return resolver.resolve(name, params)
    except ValueError as exc:
        _log.debug("metric %s slot-fill rejected: %s", name, exc)
    except Exception as exc:  # a real DB/schema error, not a bad slot-fill
        _log.warning("metric %s resolve failed: %s", name, exc)
    return None


_SMALL_SAMPLE = 24  # a rate over fewer sales than this is not statistically reliable


def _small_sample_note(result: MetricResult) -> str:
    """A deterministic caveat baked into the evidence block when a rate is computed over a small
    number of sales, so the warning does not depend on the model choosing to add it. Fires only when
    the result carries both a rate and an n_sales column."""
    cols = [str(c).lower() for c in getattr(result, "columns", []) or []]
    if not any("rate" in c for c in cols) or "n_sales" not in cols:
        return ""
    n_idx = cols.index("n_sales")
    try:
        small = [row[n_idx] for row in (result.rows or [])
                 if isinstance(row[n_idx], (int, float)) and row[n_idx] <= _SMALL_SAMPLE]
    except (IndexError, TypeError):
        return ""
    if not small:
        return ""
    return (" (small sample: some sizes have only a handful of sales, so these rates are not "
            "statistically reliable and should be described that way, not stated as settled fact)")


def metric_context(result: MetricResult) -> dict:
    """The metric result as a context block (kept separate from vector hits, never fused)."""
    return {"id": "metric:" + result.name, "text": "Metric " + result.summary() +
            _small_sample_note(result),
            "source": "metric", "doc_type": "metric", "score": 1.0}
