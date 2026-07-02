"""The governed metric layer.

Metrics are defined in the domain pack (metrics.yaml) using DuckDB named parameters ($name).
The engine fills slots through a validated call and runs the template on a read-only DuckDB
connection with external access disabled, and rejects anything that is not a single SELECT or
that reads the raw bronze/silver layers. So the model never writes free SQL, can never mutate
data, read the filesystem, or reach raw PII: it gets trustworthy governed numbers only.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import duckdb
import yaml

_PARAM_RE = re.compile(r"\$(\w+)")
_RAW_LAYER_RE = re.compile(r"\b(bronze|silver)_\w+", re.IGNORECASE)
_MAX_SUMMARY_ROWS = 20
_SCALAR = (str, int, float, bool)


def load_metrics(pack: str) -> dict:
    with open(os.path.join(pack, "metrics.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {m["name"]: m for m in (data.get("metrics", []) or [])}


def _strip_sql(sql: str) -> str:
    """Remove block/line comments and string-literal contents, so structural checks below do
    not trip on ';' or keywords that live inside strings or comments."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"'(?:''|[^'])*'", "''", sql)
    return sql


def _is_read_only(sql: str) -> bool:
    stripped = _strip_sql(sql).strip().rstrip(";")
    if ";" in stripped:  # a second statement
        return False
    low = stripped.lstrip().lower()
    return low.startswith("select") or low.startswith("with")


@dataclass
class MetricResult:
    name: str
    params: dict
    columns: list
    rows: list

    def summary(self) -> str:
        args = ", ".join("{}={}".format(k, v) for k, v in self.params.items() if v is not None)
        header = "{}({})".format(self.name, args)
        if not self.rows:
            return "{}: no data".format(header)

        def cell(v):
            return "null" if v is None else str(v)

        shown = self.rows[:_MAX_SUMMARY_ROWS]
        body = " | ".join(
            ", ".join("{}={}".format(c, cell(r[i])) for i, c in enumerate(self.columns))
            for r in shown)
        if len(self.rows) > _MAX_SUMMARY_ROWS:
            body += " | (+{} more rows)".format(len(self.rows) - _MAX_SUMMARY_ROWS)
        return "{}: {}".format(header, body)


class MetricResolver:
    def __init__(self, domain: str, db_path: str, domains_dir: str = "domains") -> None:
        self.db_path = db_path
        self.metrics = load_metrics(os.path.join(domains_dir, domain))

    def names(self) -> list[str]:
        return list(self.metrics)

    def resolve(self, name: str, params: dict | None = None) -> MetricResult:
        params = params or {}
        spec = self.metrics.get(name)
        if not spec:
            raise ValueError("unknown metric: {}".format(name))
        template = spec.get("sql_template")
        if not template:
            raise ValueError("metric {} has no sql_template".format(name))
        if not _is_read_only(template):
            raise ValueError("metric {} must be a single read-only SELECT".format(name))
        if _RAW_LAYER_RE.search(_strip_sql(template)):
            raise ValueError("metric {} may not read the raw bronze/silver layers".format(name))

        declared = spec.get("params", {}) or {}
        unknown = set(params) - set(declared)
        if unknown:
            raise ValueError("unknown params for {}: {}".format(name, sorted(unknown)))
        for key, value in params.items():
            if value is not None and not isinstance(value, _SCALAR):
                raise ValueError("param {} must be a scalar value".format(key))
        referenced = set(_PARAM_RE.findall(template))
        undeclared = referenced - set(declared)
        if undeclared:
            raise ValueError("metric {} references undeclared params: {}".format(
                name, sorted(undeclared)))
        bind = {key: params.get(key) for key in declared if key in referenced}

        con = duckdb.connect(self.db_path, read_only=True,
                             config={"enable_external_access": False})
        try:
            cur = con.execute(template, bind)
            columns = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchall()]
        finally:
            con.close()
        return MetricResult(name=name, params=params, columns=columns, rows=rows)
