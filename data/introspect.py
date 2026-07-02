"""Read-only views of a domain's declared structure (M7.4): ontology, governed metrics, and the
medallion lineage. Everything is read from the pack manifest and metrics file, so these render
any domain and never expose a metric's SQL template or raw PII.
"""
from __future__ import annotations

import os

from data.lakehouse import load_manifest
from data.metrics import load_metrics


def _pack(domain: str, domains_dir: str) -> str:
    return os.path.join(domains_dir, domain)


def ontology_view(domain: str, domains_dir: str = "domains") -> dict:
    """Entity labels and the typed relationships between them, from the pack's graph section."""
    manifest = load_manifest(_pack(domain, domains_dir))
    graph = manifest.get("graph", {}) or {}
    return {
        "entity_types": manifest.get("entity_types", []) or [],
        "nodes": [{"label": n.get("label"), "key": n.get("key"), "source": n.get("source")}
                  for n in (graph.get("nodes", []) or []) if n.get("label")],
        "edges": [{"type": e.get("type"), "from": e.get("from"), "to": e.get("to")}
                  for e in (graph.get("edges", []) or [])
                  if e.get("type") and e.get("from") and e.get("to")],
    }


def metrics_view(domain: str, domains_dir: str = "domains") -> list:
    """The governed metrics as metadata only (name, grain, source, dimensions, params). The SQL
    template is never exposed."""
    metrics = load_metrics(_pack(domain, domains_dir))
    return [{"name": spec.get("name"), "grain": spec.get("grain"), "source": spec.get("source"),
             "dimensions": spec.get("dimensions", []) or [],
             "params": list((spec.get("params") or {}).keys())}
            for spec in metrics.values() if spec.get("name")]


def lineage_view(domain: str, domains_dir: str = "domains") -> dict:
    """The medallion lineage per structured source: the seed file flows bronze -> silver -> gold,
    which PII is masked, and which governed metrics read each gold table."""
    pack = _pack(domain, domains_dir)
    sources = (load_manifest(pack).get("sources", {}) or {}).get("structured", []) or []
    metric_of_role: dict[str, list] = {}
    for spec in load_metrics(pack).values():
        if spec.get("name"):
            metric_of_role.setdefault(spec.get("source"), []).append(spec["name"])
    rows = []
    roles = set()
    for src in sources:
        role = src.get("role")
        if not role:
            continue
        roles.add(role)
        rows.append({
            "role": role,
            "file": src.get("file"),
            "layers": ["bronze_" + role, "silver_" + role, role],
            "pii_columns": src.get("pii_columns", []) or [],
            "metrics": metric_of_role.get(role, []),
        })
    # metrics whose source is not a raw role (a typo, or a derived gold model) are surfaced, not
    # silently dropped
    unmapped = [{"source": source, "metrics": names}
                for source, names in metric_of_role.items() if source not in roles]
    return {"medallion": rows, "unmapped_metrics": unmapped}
