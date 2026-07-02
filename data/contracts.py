"""Manifest-driven data contracts: the checks a dbt test suite would give, read from the pack.

For each structured source: the gold table exists, every declared column is present, and the
primary key is present, non-null, and unique. Runs on a read-only DuckDB connection.
"""
from __future__ import annotations

import os

import duckdb

from data.lakehouse import load_manifest, quote_ident


def check_contracts(domain: str, db_path: str, domains_dir: str = "domains") -> list[str]:
    pack = os.path.join(domains_dir, domain)
    sources = (load_manifest(pack).get("sources", {}) or {}).get("structured", []) or []
    try:
        con = duckdb.connect(db_path, read_only=True)
    except duckdb.Error:
        return ["database missing or unreadable at {} — run the build first".format(db_path)]

    violations: list[str] = []
    try:
        for src in sources:
            role = src["role"]
            declared = list((src.get("columns", {}) or {}).keys())
            pk = src.get("primary_key")

            present = [row[0] for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? AND table_schema = 'main'", [role]).fetchall()]
            if not present:
                violations.append("{}: gold table is missing".format(role))
                continue
            for col in declared:
                if col not in present:
                    violations.append("{}: declared column '{}' is missing".format(role, col))

            if pk and pk in present:
                nulls = con.execute("SELECT count(*) FROM {} WHERE {} IS NULL".format(
                    quote_ident(role), quote_ident(pk))).fetchone()[0]
                if nulls:
                    violations.append("{}: primary key '{}' has {} null(s)".format(role, pk, nulls))
                dupes = con.execute(
                    "SELECT count(*) FROM (SELECT {k} FROM {t} GROUP BY {k} HAVING count(*) > 1)"
                    .format(k=quote_ident(pk), t=quote_ident(role))).fetchone()[0]
                if dupes:
                    violations.append(
                        "{}: primary key '{}' has {} duplicate value(s)".format(role, pk, dupes))
            elif pk:
                violations.append(
                    "{}: primary key '{}' column is missing from gold".format(role, pk))
    finally:
        con.close()
    return violations
