"""M4.2 metric layer: validated slot-fill, correct numbers, read-only enforcement."""
import duckdb
import pytest

from data.lakehouse import build_lakehouse
from data.metrics import MetricResolver, _is_read_only


def _resolver(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    return MetricResolver("apparel_ecommerce", db), db


def test_return_rate_by_size(tmp_path):
    resolver, _ = _resolver(tmp_path)
    result = resolver.resolve("return_rate_by_size", {"size": "M"})
    row = {result.columns[i]: v for i, v in enumerate(result.rows[0])}
    assert row["size"] == "M"
    assert row["return_rate"] == 0.5


def test_units_sold(tmp_path):
    resolver, _ = _resolver(tmp_path)
    result = resolver.resolve("units_sold", {"product_id": "P006"})
    assert result.rows[0][result.columns.index("units_sold")] == 3
    assert "units_sold" in result.summary()


def test_unknown_metric_and_param_rejected(tmp_path):
    resolver, _ = _resolver(tmp_path)
    with pytest.raises(ValueError):
        resolver.resolve("nope")
    with pytest.raises(ValueError):
        resolver.resolve("units_sold", {"colour": "blue"})


def test_non_select_template_rejected(tmp_path):
    resolver, _ = _resolver(tmp_path)
    resolver.metrics["evil"] = {"name": "evil", "params": {}, "sql_template": "DELETE FROM sales"}
    with pytest.raises(ValueError):
        resolver.resolve("evil")


def test_is_read_only():
    assert _is_read_only("SELECT 1")
    assert _is_read_only("-- a comment\nWITH t AS (SELECT 1) SELECT * FROM t")
    assert _is_read_only("/* header */ SELECT 1")          # block comment tolerated
    assert _is_read_only("SELECT ';' AS x")                # semicolon inside a string is fine
    assert not _is_read_only("DELETE FROM sales")
    assert not _is_read_only("SELECT 1; DROP TABLE sales")


def test_resolve_refuses_raw_layer_read(tmp_path):
    resolver, _ = _resolver(tmp_path)
    resolver.metrics["peek"] = {"name": "peek", "params": {},
                                "sql_template": "SELECT * FROM bronze_sales"}
    with pytest.raises(ValueError):
        resolver.resolve("peek")


def test_resolve_refuses_filesystem_read(tmp_path):
    resolver, _ = _resolver(tmp_path)
    resolver.metrics["leak"] = {"name": "leak", "params": {},
                                "sql_template": "SELECT * FROM read_csv('/etc/hosts')"}
    with pytest.raises(duckdb.Error):  # external access is disabled on the metric connection
        resolver.resolve("leak")


def test_unused_declared_param_is_ok(tmp_path):
    resolver, _ = _resolver(tmp_path)
    resolver.metrics["u"] = {"name": "u", "params": {"unused": "string"},
                             "sql_template": "SELECT 1 AS x"}
    assert resolver.resolve("u").rows == [[1]]


def test_param_value_is_inert(tmp_path):
    resolver, db = _resolver(tmp_path)
    resolver.resolve("return_rate_by_size", {"size": "M'; DROP TABLE sales; --"})
    con = duckdb.connect(db, read_only=True)
    try:  # the injection value bound as a literal; sales is untouched
        assert con.execute("SELECT count(*) FROM sales").fetchone()[0] > 0
    finally:
        con.close()


def test_read_only_connection_refuses_writes(tmp_path):
    _, db = _resolver(tmp_path)
    con = duckdb.connect(db, read_only=True)
    try:
        with pytest.raises(duckdb.Error):
            con.execute("INSERT INTO sales (sale_id) VALUES ('X')")
    finally:
        con.close()
