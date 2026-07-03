"""Generate the dbt medallion models from a domain manifest.

A dbt model is static SQL, but this platform is domain-swappable: one engine serves any domain from
its `domain.yaml`. To keep dbt manifest-driven rather than hand-copied per domain, this reads the
active domain's manifest and writes the bronze -> silver -> gold models plus their schema tests.
The silver transform reuses the exact same column expressions as the Python lakehouse builder
(`data/lakehouse._column_expr`), so there is one definition of typing and PII masking, not two.

Run it before `dbt build` (the `make dbt-build` target does both):
    DOMAIN=apparel_ecommerce uv run python scripts/dbt_codegen.py
"""
from __future__ import annotations

import os
import sys

from data.lakehouse import _column_expr, load_manifest, validate_source

_DBT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dbt")
_MODELS = os.path.join(_DBT_DIR, "models")
_GEN_DIRS = ("staging", "silver", "marts")
_HEADER = "-- GENERATED from the {} manifest by scripts/dbt_codegen.py. Do not edit by hand.\n"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _clean_generated() -> None:
    # Remove last run's models so switching domains never leaves an orphan model behind.
    for sub in _GEN_DIRS:
        d = os.path.join(_MODELS, sub)
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.endswith(".sql"):
                    os.remove(os.path.join(d, name))
    for name in ("sources.yml", "schema.yml"):
        p = os.path.join(_MODELS, name)
        if os.path.exists(p):
            os.remove(p)


def _node_index(manifest: dict) -> dict:
    """label -> (source table, key column), so foreign keys from graph edges become dbt
    relationships tests."""
    index = {}
    for node in ((manifest.get("graph", {}) or {}).get("nodes", []) or []):
        if node.get("label") and node.get("source") and node.get("key"):
            index[node["label"]] = (node["source"], node["key"])
    return index


def _relationships(manifest: dict) -> dict:
    """(table, column) -> (target_table, target_column) derived from the manifest's graph edges,
    which encode the real foreign keys (e.g. tickets.plan_id -> plans.plan_id)."""
    nodes = _node_index(manifest)
    rels: dict = {}
    for edge in ((manifest.get("graph", {}) or {}).get("edges", []) or []):
        table = edge.get("source")
        # An edge encodes two foreign keys: the child's own key (from_key) and the key it points at
        # (to_key). Derive both, so referential integrity covers every FK the graph relies on.
        for key_field, node_label in (("to_key", edge.get("to")), ("from_key", edge.get("from"))):
            fk = edge.get(key_field)
            target = nodes.get(node_label)
            if table and fk and target:
                target_table, target_key = target
                if fk != target_key or table != target_table:  # skip a self/degenerate reference
                    rels[(table, fk)] = (target_table, target_key)
    return rels


def _yaml_list(items: list[str], indent: str) -> str:
    return "".join("{}- {}\n".format(indent, i) for i in items)


def generate(domain: str, domains_dir: str = "domains") -> list[str]:
    pack = os.path.join(domains_dir, domain)
    manifest = load_manifest(pack)
    sources = (manifest.get("sources", {}) or {}).get("structured", []) or []
    rels = _relationships(manifest)
    _clean_generated()

    source_tables, schema_models, built = [], [], []
    for src in sources:
        validate_source(src)  # same guards as the Python builder (safe role/path, PII declared)
        role = src["role"]
        # escape the single quotes so a path with a quote cannot break out of the SQL string literal
        csv_path = os.path.abspath(os.path.join(pack, src["file"])).replace("'", "''")
        columns = src.get("columns", {}) or {}
        pii = set(src.get("pii_columns", []) or [])
        pk = src.get("primary_key")
        grain = src.get("grain", "")

        # bronze: raw CSV as text (lineage). silver: typed + PII-masked. gold mart: the curated
        # table the metric layer and the graph read.
        _write(os.path.join(_MODELS, "staging", "bronze_{}.sql".format(role)),
               _HEADER.format(domain) + "select * from read_csv_auto('{}', header=true, "
               "all_varchar=true)\n".format(csv_path))
        projection = ", ".join(_column_expr(n, t, pii) for n, t in columns.items()) or "*"
        _write(os.path.join(_MODELS, "silver", "silver_{}.sql".format(role)),
               _HEADER.format(domain) + "select {} from {{{{ ref('bronze_{}') }}}}\n".format(
                   projection, role))
        _write(os.path.join(_MODELS, "marts", "{}.sql".format(role)),
               _HEADER.format(domain) + "select * from {{{{ ref('silver_{}') }}}}\n".format(role))
        built.append(role)

        source_tables.append("      - name: {}\n        description: \"{}\"\n".format(role, grain))
        schema_models.append(_model_schema(role, columns, pii, pk, grain, rels))

    _write(os.path.join(_MODELS, "sources.yml"),
           "version: 2\n\nsources:\n  - name: raw\n"
           "    description: \"Raw {} CSVs (bronze reads these unchanged).\"\n"
           "    tables:\n{}".format(domain, "".join(source_tables)))
    _write(os.path.join(_MODELS, "schema.yml"),
           "version: 2\n\nmodels:\n{}".format("".join(schema_models)))
    return built


def _model_schema(role, columns, pii, pk, grain, rels) -> str:
    lines = ["  - name: {}\n".format(role),
             "    description: \"Gold: {}\"\n".format(grain or role),
             "    columns:\n"]
    for name in columns:
        col_tests = []
        if name == pk:
            col_tests += ["          - not_null\n", "          - unique\n"]
        if name in pii:
            col_tests.append("          - is_masked\n")
        if (role, name) in rels:
            target_table, target_key = rels[(role, name)]
            # dbt 1.11 wants generic-test arguments nested under `arguments:`
            col_tests.append("          - relationships:\n"
                             "              arguments:\n"
                             "                to: ref('{}')\n"
                             "                field: {}\n".format(target_table, target_key))
        lines.append("      - name: {}\n".format(name))
        if col_tests:
            lines.append("        tests:\n")
            lines += col_tests
    return "".join(lines)


def main() -> int:
    domain = os.getenv("DOMAIN", "apparel_ecommerce")
    built = generate(domain)
    print("dbt models generated for {}: {}".format(domain, ", ".join(built)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
