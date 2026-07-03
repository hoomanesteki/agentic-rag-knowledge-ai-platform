"""T1.2 parity: the Python lakehouse builder and the dbt models produce identical gold, so the two
build paths can never silently drift. Skips when dbt is not installed, so the offline engine and
the base test suite do not need it."""
import os
import shutil
import subprocess

import duckdb
import pytest

from data.lakehouse import build_lakehouse
from scripts.dbt_codegen import generate

_DBT = shutil.which("dbt")
pytestmark = pytest.mark.skipif(_DBT is None, reason="dbt not installed (optional extra)")


def _build_with_dbt(db_path: str, domain: str) -> None:
    generate(domain)  # regenerate the models for this domain (apparel is the committed default)
    env = {**os.environ, "LAKEHOUSE_DB": db_path, "DOMAIN": domain, "DBT_PROFILES_DIR": "dbt"}
    # a subprocess so dbt's DuckDB connection is fully closed before we read the file back
    result = subprocess.run([_DBT, "build", "--project-dir", "dbt"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, "dbt build failed:\n{}".format(result.stdout[-2000:])


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


def test_dbt_gold_matches_python_builder(tmp_path):
    domain = "apparel_ecommerce"
    py_db = str(tmp_path / "py.duckdb")
    roles = build_lakehouse(domain, py_db)

    dbt_db = str(tmp_path / "dbt.duckdb")
    _build_with_dbt(dbt_db, domain)

    for role in roles:
        assert _gold(py_db, role) == _gold(dbt_db, role), \
            "{}: dbt gold differs from the Python builder".format(role)
