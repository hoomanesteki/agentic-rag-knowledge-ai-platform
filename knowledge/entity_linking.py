"""M5.2 entity linking: connect mentions in unstructured docs to canonical graph entities.

A build-time pass. For each unstructured source that declares a `mentions` block, we shortlist
candidate entities by cheap lexical overlap, ask the LLM which are actually referred to and how
confident it is, then add a typed edge from the doc node to each entity at or above a threshold.
Lower-confidence matches go to a review list for a human, never silently dropped. Domain
agnostic: the doc label, edge type, target label, and name property all come from the manifest.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from adapters.base import GraphStore
from data.lakehouse import load_manifest

_log = logging.getLogger("skein.entity_linking")
_WORD = re.compile(r"[a-z0-9]+")
DEFAULT_THRESHOLD = 0.6
_MAX_CANDIDATES = 5000   # find_nodes cap; warn if a catalog is larger
_MAX_SHORTLIST = 25      # cap the per-doc prompt so cost and truncation stay bounded
_MAX_TOKENS = 384

_SYSTEM = (
    "You match entity mentions in a short document to a catalog. Given candidates (id and name) "
    "and the text, reply with ONLY a JSON array of the entities the text actually refers to, "
    'each {"id": <id>, "confidence": <0..1>}. Prefer precision: include an id only if the text '
    "is genuinely about it. Reply [] if none apply."
)


@dataclass
class LinkReport:
    docs: int = 0
    linked: int = 0
    review_list: list = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _extract_list(raw: str | None) -> list | None:
    """The parsed JSON array, or None if the output could not be parsed (truncated or not JSON).
    An empty list is a real answer (no mentions), so it is kept distinct from None."""
    if not raw:
        return None
    start = raw.find("[")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, list) else None


def _coerce_confidence(value) -> float | None:
    if isinstance(value, bool):  # True/False is not a confidence, even though it is an int
        return None
    try:
        return float(value)  # accept 0.9 and "0.9"; LLMs emit both
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _name_tokens(name: str) -> set[str]:
    # keep short alphanumerics (len >= 2) so entities named "M6" or "Go" can still be matched
    return {t for t in _tokens(name) if len(t) >= 2}


def _common_tokens(candidates: list[dict]) -> set[str]:
    """Tokens shared by many candidate names (a common brand word). Excluding them keeps
    the shortlist from matching every entity on a token they all have."""
    df = Counter()
    for cand in candidates:
        for token in _name_tokens(cand["name"]):
            df[token] += 1
    cutoff = max(2, len(candidates) // 2)
    return {token for token, n in df.items() if n >= cutoff}


def _shortlist(text_tokens: set[str], candidates: list[dict], common: set[str]) -> list[dict]:
    """Candidates whose distinctive name tokens overlap the text. Keeps the LLM call small and
    precise instead of sending the whole catalog every time."""
    out = []
    for cand in candidates:
        name_tokens = _name_tokens(cand["name"])
        distinctive = (name_tokens - common) or name_tokens
        if distinctive & text_tokens:
            out.append(cand)
    return out


def _score(llm, text: str, shortlist: list[dict]) -> list | None:
    lines = ["Candidates:"]
    for cand in shortlist:
        lines.append("- id={} name={}".format(cand["id"], cand["name"]))
    lines += ["", "Text: " + text, "JSON:"]
    raw = llm.generate("\n".join(lines), system=_SYSTEM, max_tokens=_MAX_TOKENS).text
    return _extract_list(raw)  # None means unparseable, [] means the model found no mention


def link_mentions(domain: str, graph: GraphStore, llm, domains_dir: str = "domains",
                  threshold: float = DEFAULT_THRESHOLD) -> LinkReport:
    """Link mentions for every unstructured source that declares a `mentions` block. The graph
    must already hold the target nodes (run the loader first)."""
    pack = os.path.join(domains_dir, domain)
    manifest = load_manifest(pack)
    report = LinkReport()
    node_keys = {n["label"]: n["key"]
                 for n in (manifest.get("graph", {}) or {}).get("nodes", []) or []}

    for src in (manifest.get("sources", {}) or {}).get("unstructured", []) or []:
        mention = src.get("mentions")
        if not mention:
            continue
        doc_label = mention["node_label"]
        edge_type = mention["edge_type"]
        target_label = mention["target_label"]
        name_prop = mention.get("target_name_property", "name")
        target_key = node_keys.get(target_label)
        if not target_key:
            _log.warning("mentions target %s is not a graph node; skipping", target_label)
            continue

        nodes = graph.find_nodes(target_label, limit=_MAX_CANDIDATES + 1)
        if len(nodes) > _MAX_CANDIDATES:
            _log.warning("%s has more than %d entities; linking only the first %d",
                         target_label, _MAX_CANDIDATES, _MAX_CANDIDATES)
            nodes = nodes[:_MAX_CANDIDATES]
        candidates = [{"id": n.id, "name": str(n.properties.get(name_prop, ""))} for n in nodes]
        candidate_ids = {c["id"] for c in candidates}
        common = _common_tokens(candidates)
        id_field, text_field = src["id_field"], src["text_field"]
        cache: dict[str, list | None] = {}  # identical texts share one LLM call

        for rec in _load_jsonl(os.path.join(pack, src["file"])):
            report.docs += 1
            if id_field not in rec or text_field not in rec:
                _log.warning("record missing '%s' or '%s'; skipping", id_field, text_field)
                continue
            doc_id = str(rec[id_field])
            text = (rec.get(text_field) or "").strip()
            shortlist = _shortlist(_tokens(text), candidates, common) if text else []
            if not shortlist:
                continue
            text_tokens = _tokens(text)
            shortlist = sorted(
                shortlist, key=lambda c: len(_name_tokens(c["name"]) & text_tokens),
                reverse=True)[:_MAX_SHORTLIST]

            if text not in cache:
                cache[text] = _score(llm, text, shortlist)
            scored = cache[text]
            if scored is None:  # could not parse: a human must resolve it, never a silent drop
                _log.warning("unparseable link output for %s; queuing for review", doc_id)
                report.review_list.append(
                    {"domain": domain, "doc": doc_id, "candidate": None,
                     "confidence": None, "reason": "unparsed_llm_output"})
                continue
            if not scored:
                continue  # the model found no mention (a real answer, not a drop)

            high, seen = [], set()
            for item in scored:
                if not isinstance(item, dict):
                    _log.debug("skipping non-dict link item: %r", item)
                    continue
                cid = str(item.get("id"))
                conf = _coerce_confidence(item.get("confidence"))
                if cid not in candidate_ids:
                    _log.debug("skipping unknown candidate id: %r", cid)
                    continue
                if conf is None:
                    _log.debug("skipping unparseable confidence for %s", cid)
                    continue
                if conf >= threshold and cid not in seen:
                    seen.add(cid)
                    high.append((doc_id, cid))
                elif 0 < conf < threshold:
                    report.review_list.append(
                        {"domain": domain, "doc": doc_id, "candidate": cid, "confidence": conf})
            if high:
                # create the doc node only when it has a confident link, so low-confidence-only
                # docs do not leave orphan nodes in the graph
                graph.upsert_nodes(doc_label, id_field, [{id_field: doc_id}])
                report.linked += graph.upsert_edges(
                    edge_type, doc_label, id_field, target_label, target_key, high)
    return report
