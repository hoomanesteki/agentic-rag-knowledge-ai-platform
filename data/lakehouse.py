"""Manifest-driven medallion lakehouse on DuckDB (embedded, no server).

- bronze_<role>: the raw CSV loaded as text, unchanged (lineage; may hold raw PII).
- silver_<role>: cast to the manifest's declared types, text trimmed, PII columns masked.
- <role>: the curated gold table the metric layer queries (products, sales, ...).

The engine reads only the domain manifest, so the same code builds any domain's lakehouse.
Masking and the schema come from the pack; the engine fails loud (never open) on a bad
manifest. This is the light path; PySpark/Delta is a later enterprise-parity swap.
"""
from __future__ import annotations

import os
import re

import duckdb
import yaml

_DUCK_TYPES = {
    "string": "VARCHAR", "text": "VARCHAR",
    "int": "BIGINT", "integer": "BIGINT",
    "float": "DOUBLE", "double": "DOUBLE",
    "bool": "BOOLEAN", "boolean": "BOOLEAN",
    "date": "DATE",
}
_ROLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_manifest(pack: str) -> dict:
    with open(os.path.join(pack, "domain.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _duck_type(declared: str | None) -> str:
    if declared is None:
        return "VARCHAR"
    key = declared.lower()
    if key not in _DUCK_TYPES:
        raise ValueError("unknown column type '{}' (supported: {})".format(
            declared, ", ".join(sorted(_DUCK_TYPES))))
    return _DUCK_TYPES[key]


def _column_expr(name: str, declared: str, pii: set[str]) -> str:
    col = quote_ident(name)
    if name in pii:
        # stable pseudonym over the trimmed value; raw PII never lands in silver/gold.
        # md5 is not cryptographic; adequate for synthetic demo data only.
        return ("CASE WHEN {c} IS NULL THEN NULL "
                "ELSE 'masked:' || substr(md5(trim(CAST({c} AS VARCHAR))), 1, 16) END AS {c}"
                ).format(c=col)
    duck = _duck_type(declared)
    if duck == "VARCHAR":
        return "trim(CAST({c} AS VARCHAR)) AS {c}".format(c=col)
    return "CAST({c} AS {t}) AS {c}".format(c=col, t=duck)


def validate_source(src: dict) -> None:
    """Fail loud on an unsafe or fail-open source, so every build path (the Python builder and the
    dbt codegen) enforces the same guards: a role name that is a safe identifier, a relative path
    that cannot escape the pack, and no PII column left out of `columns` (which would otherwise
    fall through to an unmasked `select *`)."""
    role = src.get("role", "")
    if not _ROLE_RE.match(role):
        raise ValueError("invalid role name: {}".format(role))
    rel = src.get("file", "")
    if os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
        raise ValueError("unsafe source path: {}".format(rel))
    missing = set(src.get("pii_columns", []) or []) - set(src.get("columns", {}) or {})
    if missing:
        raise ValueError("{}: pii_columns {} not declared in columns".format(role, sorted(missing)))


def _build_source(con: duckdb.DuckDBPyConnection, pack: str, src: dict) -> str:
    validate_source(src)
    role = src["role"]
    rel = src["file"]
    columns = src.get("columns", {}) or {}
    pii = set(src.get("pii_columns", []) or [])

    bronze, silver = "bronze_" + role, "silver_" + role
    try:
        con.execute("CREATE OR REPLACE TABLE {} AS SELECT * FROM "
                    "read_csv_auto(?, header=true, all_varchar=true)".format(quote_ident(bronze)),
                    [os.path.join(pack, rel)])
        header = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [bronze]).fetchall()}
        for name in columns:
            if name not in header:
                raise ValueError("declared column '{}' not in CSV header {}".format(
                    name, sorted(header)))
        projection = ", ".join(_column_expr(n, t, pii) for n, t in columns.items())
        if not projection:  # no declared columns and (checked above) no PII
            projection = "*"
        con.execute("CREATE OR REPLACE TABLE {} AS SELECT {} FROM {}".format(
            quote_ident(silver), projection, quote_ident(bronze)))
        con.execute("CREATE OR REPLACE TABLE {} AS SELECT * FROM {}".format(
            quote_ident(role), quote_ident(silver)))
    except (duckdb.Error, ValueError) as exc:
        raise RuntimeError("{} ({}): {}".format(role, rel, exc)) from exc
    return role


def build_lakehouse(domain: str, db_path: str, domains_dir: str = "domains") -> list[str]:
    pack = os.path.join(domains_dir, domain)
    sources = (load_manifest(pack).get("sources", {}) or {}).get("structured", []) or []
    # Build into a temp file and swap on success, so a failed build never leaves a half-updated
    # or stale-orphan-carrying database behind.
    tmp_path = db_path + ".building"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    con = duckdb.connect(tmp_path)
    try:
        built = [_build_source(con, pack, src) for src in sources]
    except Exception:
        con.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    con.close()
    os.replace(tmp_path, db_path)
    return built
