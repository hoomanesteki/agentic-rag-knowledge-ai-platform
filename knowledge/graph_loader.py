"""Load the knowledge graph from the gold lakehouse, driven entirely by the pack manifest.

The `graph` section of domain.yaml declares nodes (label, gold source table, key, properties)
and edges (type, from/to labels, and the source columns that carry each id). This reads the
gold DuckDB tables read-only and writes nodes and typed edges into a GraphStore. Nothing here
names a label or relationship; the same code builds any domain's graph. Values are coerced to
JSON-safe scalars so the same rows work for the in-memory fake and the Neo4j HTTP path.
"""
from __future__ import annotations

import datetime
import decimal
import math
import os
import re

import duckdb

from adapters.base import GraphStore
from data.lakehouse import load_manifest, quote_ident

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str, kind: str) -> str:
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError("unsafe {} identifier in manifest: {!r}".format(kind, name))
    return name


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value  # NaN is not valid JSON
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)  # keep numeric so M5.3 property filters compare as numbers
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return str(value)  # times, anything else -> stable string


def _fetch_rows(con, table: str, columns: list[str], key: str) -> list[dict]:
    _ident(table, "source")
    for col in columns:
        _ident(col, "column")
    projection = ", ".join(quote_ident(c) for c in columns)
    # Skip null keys (Neo4j MERGE rejects a null key property) and coerce the key to a string, so
    # node ids match the string-coerced edge endpoints and query values on every backend. Drop
    # null/NaN properties so `SET n += row` never deletes a property on Neo4j (the fake matches).
    stmt = "SELECT {} FROM {} WHERE {} IS NOT NULL".format(
        projection, quote_ident(table), quote_ident(key))
    out = []
    for row in con.execute(stmt).fetchall():
        record = {}
        for c, v in zip(columns, row):
            jv = _jsonable(v)
            if jv is not None:
                record[c] = jv
        record[key] = str(record[key])  # key is non-null by the WHERE, so present
        out.append(record)
    return out


def _fetch_pairs(con, table: str, from_col: str, to_col: str) -> list[tuple[str, str]]:
    _ident(table, "source")
    _ident(from_col, "column")
    _ident(to_col, "column")
    stmt = ("SELECT DISTINCT {f}, {t} FROM {tbl} "
            "WHERE {f} IS NOT NULL AND {t} IS NOT NULL").format(
                f=quote_ident(from_col), t=quote_ident(to_col), tbl=quote_ident(table))
    return [(str(f), str(t)) for f, t in con.execute(stmt).fetchall()]


def _ontology_statements(pack: str) -> list[str]:
    # Split on ';' after dropping // comment lines. Constraint DDL has no string literals, so
    # this is safe for the ontology files we ship; a statement with a ';' inside a string would
    # need a real parser.
    path = os.path.join(pack, "ontology.cypher")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if not ln.strip().startswith("//")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def load_graph(domain: str, db_path: str, store: GraphStore, domains_dir: str = "domains",
               apply_ontology: bool = True) -> dict:
    """Build the graph for one domain into `store`. Returns per-label and per-edge counts."""
    pack = os.path.join(domains_dir, domain)
    graph = (load_manifest(pack).get("graph", {}) or {})
    nodes_spec = graph.get("nodes", []) or []
    edges_spec = graph.get("edges", []) or []
    counts: dict = {"nodes": {}, "edges": {}}
    if not nodes_spec:
        store.reset()  # clear any stale graph even when this domain declares none
        return counts  # a domain may ship no graph; retrieval still works vector-only
    if not os.path.exists(db_path):
        raise RuntimeError("lakehouse not found at {}; run make lakehouse first".format(db_path))

    con = duckdb.connect(db_path, read_only=True)
    try:
        store.reset()
        if apply_ontology:
            store.apply_schema(_ontology_statements(pack))

        node_keys: dict[str, str] = {}
        for spec in nodes_spec:
            label = _ident(spec["label"], "label")
            key = _ident(spec["key"], "key")
            props = [p for p in (spec.get("properties", []) or []) if p != key]
            rows = _fetch_rows(con, spec["source"], [key, *props], key)
            counts["nodes"][label] = store.upsert_nodes(label, key, rows)
            node_keys[label] = key

        for spec in edges_spec:
            etype = _ident(spec["type"], "edge_type")
            from_label = _ident(spec["from"], "label")
            to_label = _ident(spec["to"], "label")
            if from_label not in node_keys or to_label not in node_keys:
                raise ValueError(
                    "edge {} references a node label not declared in graph.nodes".format(etype))
            pairs = _fetch_pairs(con, spec["source"], spec["from_key"], spec["to_key"])
            counts["edges"][etype] = store.upsert_edges(
                etype, from_label, node_keys[from_label], to_label, node_keys[to_label], pairs)
        return counts
    finally:
        con.close()
