"""M4.1 lakehouse: manifest-driven build, type casting, PII masking, and data contracts."""
import duckdb
import pytest

from data.contracts import check_contracts
from data.lakehouse import build_lakehouse, validate_source

_MANIFEST = """\
name: testdom
languages: [en]
entity_types: [Account]
sources:
  structured:
    - file: seed/structured/accounts.csv
      role: accounts
      primary_key: account_id
      columns:
        account_id: string
        email: string
        seats: int
        active: bool
      pii_columns: [email]
"""

_ACCOUNTS = """\
account_id,email,seats,active
A1,alice@example.com,5,true
A2,bob@example.com,12,false
"""


def _make_domain(tmp_path, manifest=_MANIFEST, accounts=_ACCOUNTS):
    pack = tmp_path / "domains" / "testdom"
    (pack / "seed" / "structured").mkdir(parents=True)
    (pack / "domain.yaml").write_text(manifest)
    (pack / "seed" / "structured" / "accounts.csv").write_text(accounts)
    return str(tmp_path / "domains")


def test_build_types_and_pii_mask(tmp_path):
    domains = _make_domain(tmp_path)
    db = str(tmp_path / "lh.duckdb")
    built = build_lakehouse("testdom", db, domains_dir=domains)
    assert built == ["accounts"]

    con = duckdb.connect(db, read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM accounts").fetchone()[0] == 2
        # email is PII -> masked in silver/gold, raw value gone
        emails = [r[0] for r in con.execute("SELECT email FROM accounts").fetchall()]
        assert all(e.startswith("masked:") for e in emails)
        assert not any("example.com" in e for e in emails)
        # types cast per manifest
        types = {r[0]: r[1] for r in con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'accounts'").fetchall()}
        assert types["seats"] in ("BIGINT", "HUGEINT")
        assert types["active"] == "BOOLEAN"
        # bronze keeps the raw value for lineage
        raw = [r[0] for r in con.execute("SELECT email FROM bronze_accounts").fetchall()]
        assert any("example.com" in e for e in raw)
    finally:
        con.close()


def test_validate_source_rejects_undeclared_pii():
    # fail loud, never open: a PII column not in `columns` would fall through to an unmasked
    # select *, so both the Python builder and the dbt codegen must refuse it.
    with pytest.raises(ValueError, match="pii_columns"):
        validate_source({"role": "accounts", "file": "a.csv", "columns": {},
                         "pii_columns": ["email"]})


def test_validate_source_rejects_unsafe_role_and_path():
    with pytest.raises(ValueError, match="invalid role"):
        validate_source({"role": "../escape", "file": "a.csv"})
    with pytest.raises(ValueError, match="unsafe source path"):
        validate_source({"role": "accounts", "file": "../../etc/passwd"})
    with pytest.raises(ValueError, match="unsafe source path"):
        validate_source({"role": "accounts", "file": "/abs/path.csv"})


def test_validate_source_accepts_a_clean_source():
    validate_source({"role": "accounts", "file": "seed/structured/a.csv",
                     "columns": {"account_id": "string", "email": "string"},
                     "pii_columns": ["email"]})  # does not raise


def test_contracts_pass_on_clean_data(tmp_path):
    domains = _make_domain(tmp_path)
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("testdom", db, domains_dir=domains)
    assert check_contracts("testdom", db, domains_dir=domains) == []


def test_contracts_flag_duplicate_primary_key(tmp_path):
    dupes = _ACCOUNTS + "A1,carol@example.com,3,true\n"
    domains = _make_domain(tmp_path, accounts=dupes)
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("testdom", db, domains_dir=domains)
    violations = check_contracts("testdom", db, domains_dir=domains)
    assert any("duplicate" in v for v in violations)


def test_pii_column_not_declared_fails_closed(tmp_path):
    manifest = _MANIFEST.replace("        email: string\n", "")  # email dropped from columns
    domains = _make_domain(tmp_path, manifest=manifest)
    with pytest.raises((ValueError, RuntimeError)):
        build_lakehouse("testdom", str(tmp_path / "lh.duckdb"), domains_dir=domains)


def test_unknown_column_type_rejected(tmp_path):
    manifest = _MANIFEST.replace("seats: int", "seats: decimal")
    domains = _make_domain(tmp_path, manifest=manifest)
    with pytest.raises((ValueError, RuntimeError)):
        build_lakehouse("testdom", str(tmp_path / "lh.duckdb"), domains_dir=domains)


def test_primary_key_missing_from_gold_flagged(tmp_path):
    # declare a PK that is not among the columns, so it never reaches gold
    manifest = _MANIFEST.replace("      primary_key: account_id", "      primary_key: missing_id")
    domains = _make_domain(tmp_path, manifest=manifest)
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("testdom", db, domains_dir=domains)
    violations = check_contracts("testdom", db, domains_dir=domains)
    assert any("missing from gold" in v for v in violations)


def test_contracts_on_missing_db_returns_violation(tmp_path):
    domains = _make_domain(tmp_path)
    missing = str(tmp_path / "never-built.duckdb")
    violations = check_contracts("testdom", missing, domains_dir=domains)
    assert violations and "database missing" in violations[0]


def test_rebuild_swaps_and_drops_orphans(tmp_path):
    domains = _make_domain(tmp_path)
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("testdom", db, domains_dir=domains)
    # rebuild a different domain into the same path; the old table must be gone
    other = _make_domain(tmp_path / "other",
                         manifest=_MANIFEST.replace("role: accounts", "role: widgets")
                         .replace("accounts.csv", "widgets.csv"))
    (tmp_path / "other" / "domains" / "testdom" / "seed" / "structured" / "widgets.csv").write_text(
        _ACCOUNTS)
    build_lakehouse("testdom", db, domains_dir=other)
    con = duckdb.connect(db, read_only=True)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
    finally:
        con.close()
    assert "widgets" in tables and "accounts" not in tables


def test_apparel_lakehouse_supports_return_rate(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    built = build_lakehouse("apparel_ecommerce", db)
    assert "sales" in built and "products" in built
    assert check_contracts("apparel_ecommerce", db) == []
    con = duckdb.connect(db, read_only=True)
    try:
        rate = con.execute(
            "SELECT sum(CASE WHEN s.returned THEN 1 ELSE 0 END)::DOUBLE / count(*) "
            "FROM sales s JOIN products p ON p.product_id = s.product_id "
            "WHERE p.size = 'M'").fetchone()[0]
        assert rate == 0.5  # 2 of 4 size-M sale lines were returned
    finally:
        con.close()
