"""T5 reproducibility: the medallion build is deterministic, so anyone who clones and builds gets
the same gold. Building a domain twice must produce byte-identical gold (the PII pseudonyms are a
stable hash, and the CSV read is ordered), so a rerun never silently changes a governed number."""
import glob
import os

import duckdb
import pytest

from data.lakehouse import build_lakehouse

_DOMAINS = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob("domains/*/domain.yaml"))


def _gold(db_path: str, role: str):
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = 'main' ORDER BY ordinal_position",
            [role]).fetchall()]
        rows = con.execute("SELECT * FROM main.{} ORDER BY ALL".format(role)).fetchall()
    finally:
        con.close()
    return cols, rows


@pytest.mark.parametrize("domain", _DOMAINS)
def test_build_is_deterministic(domain, tmp_path):
    db1, db2 = str(tmp_path / "a.duckdb"), str(tmp_path / "b.duckdb")
    roles1 = build_lakehouse(domain, db1)
    roles2 = build_lakehouse(domain, db2)
    assert roles1 == roles2 and roles1, "{}: unstable role set".format(domain)
    for role in roles1:
        assert _gold(db1, role) == _gold(db2, role), \
            "{}: gold table '{}' differs between two builds".format(domain, role)
