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
    raw = llm.generate(_slot_prompt(query, resolver), system=_SLOT_SYSTEM, max_tokens=200).text
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


def metric_context(result: MetricResult) -> dict:
    """The metric result as a context block (kept separate from vector hits, never fused)."""
    return {"id": "metric:" + result.name, "text": "Metric " + result.summary(),
            "source": "metric", "doc_type": "metric", "score": 1.0}
