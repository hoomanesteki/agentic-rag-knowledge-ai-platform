"""T5 semantic layer: every governed metric in every domain is well-formed and actually resolves
against gold, so the single source of truth is validated, not just trusted. Offline, no keys."""
import glob
import os
import re

import pytest

from data.lakehouse import build_lakehouse
from data.metrics import _PARAM_RE, MetricResolver, _is_read_only, load_metrics

_DOMAINS = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob("domains/*/metrics.yaml"))


@pytest.mark.parametrize("domain", _DOMAINS)
def test_metrics_are_well_formed(domain):
    metrics = load_metrics(os.path.join("domains", domain))
    assert metrics, "{} declares no governed metrics".format(domain)
    for name, spec in metrics.items():
        template = spec.get("sql_template")
        assert spec.get("source"), "{}.{} has no source".format(domain, name)
        assert template, "{}.{} has no sql_template".format(domain, name)
        assert _is_read_only(template), "{}.{} is not read-only".format(domain, name)
        assert not re.search(r"\b(bronze|silver)_", template), \
            "{}.{} reads a raw layer".format(domain, name)
        # declared params and the $placeholders in the SQL must match exactly: no undeclared
        # placeholder (would be a runtime error) and no dead declared param.
        declared = set(spec.get("params", {}) or {})
        referenced = set(_PARAM_RE.findall(template))
        assert declared == referenced, \
            "{}.{} params {} != SQL placeholders {}".format(domain, name, declared, referenced)


@pytest.mark.parametrize("domain", _DOMAINS)
def test_metrics_resolve_against_gold(domain, tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse(domain, db)
    resolver = MetricResolver(domain, db)
    for name in resolver.names():
        result = resolver.resolve(name, {})  # no params binds to NULL -> "all values"
        assert result.columns, "{}.{} returned no columns".format(domain, name)
