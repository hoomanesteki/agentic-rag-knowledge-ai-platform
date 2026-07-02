"""M5.1 knowledge graph: load nodes and typed edges from the gold lakehouse, driven by the
manifest, into the in-memory fake (the Neo4j path is the same contract, verified on a real
database). Proves the loader is domain agnostic and that a traversal returns the expected
relations, and that only allowlisted identifiers ever reach a query.
"""
import pytest

from adapters.factory import make_graph
from adapters.neo4j_store import _ident as neo4j_ident
from data.lakehouse import build_lakehouse
from knowledge.graph_loader import _ident as loader_ident
from knowledge.graph_loader import load_graph


def _loaded(domain, tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse(domain, db)
    store = make_graph("memory")
    counts = load_graph(domain, db, store)
    return store, counts


def test_apparel_graph_nodes_edges_and_traversal(tmp_path):
    store, counts = _loaded("apparel_ecommerce", tmp_path)
    assert counts["nodes"]["Product"] > 0
    assert counts["edges"]["SUPPLIES"] > 0 and counts["edges"]["SOLD_AT"] > 0

    # P002 is supplied by SUP01 (products.csv) and was sold at ST01 (sales.csv).
    supplied = store.neighbors("Supplier", "supplier_id", "SUP01",
                               edge_type="SUPPLIES", direction="out")
    assert any(n.node.id == "P002" and n.node.label == "Product" for n in supplied)

    sold_at = store.neighbors("Product", "product_id", "P002",
                              edge_type="SOLD_AT", direction="out", to_label="Store")
    assert any(n.node.id == "ST01" for n in sold_at)
    product = store.get_node("Product", "product_id", "P002")
    assert product.properties["name"] == "Aster Flow Legging"


def test_saas_graph_nodes_edges_and_traversal(tmp_path):
    store, counts = _loaded("saas_support", tmp_path)
    assert counts["nodes"]["Ticket"] > 0 and counts["nodes"]["Plan"] > 0
    assert counts["edges"]["ON_PLAN"] > 0

    # T0001 is on plan PL02 (tickets.csv); the reverse hop lists a plan's tickets.
    on_plan = store.neighbors("Ticket", "ticket_id", "T0001", edge_type="ON_PLAN", direction="out")
    assert any(n.node.id == "PL02" and n.node.label == "Plan" for n in on_plan)

    tickets = store.neighbors("Plan", "plan_id", "PL02", edge_type="ON_PLAN", direction="in")
    assert any(n.node.id == "T0001" for n in tickets)


@pytest.mark.parametrize("domain", ["apparel_ecommerce", "saas_support"])
def test_same_loader_builds_any_domain(domain, tmp_path):
    _store, counts = _loaded(domain, tmp_path)
    assert sum(counts["nodes"].values()) > 0
    assert sum(counts["edges"].values()) > 0


def test_node_ids_are_strings_and_get_node_checks_key(tmp_path):
    store, _ = _loaded("apparel_ecommerce", tmp_path)
    product = store.get_node("Product", "product_id", "P002")
    # ids and the key property are strings on every backend, so string-coerced edges match them
    assert isinstance(product.id, str)
    assert isinstance(product.properties["product_id"], str)
    # asking with a key this label was not loaded with must miss, mirroring Neo4j
    assert store.get_node("Product", "sku", "P002") is None


def test_edges_are_idempotent_across_calls():
    store = make_graph("memory")
    store.upsert_nodes("A", "id", [{"id": "1"}])
    store.upsert_nodes("B", "id", [{"id": "9"}])
    first = store.upsert_edges("R", "A", "id", "B", "id", [("1", "9")])
    again = store.upsert_edges("R", "A", "id", "B", "id", [("1", "9")])
    assert first == 1 and again == 0  # MERGE semantics: the repeat adds nothing
    assert len(store.neighbors("A", "id", "1", edge_type="R", direction="out")) == 1


def test_load_graph_missing_lakehouse_raises(tmp_path):
    with pytest.raises(RuntimeError):
        load_graph("apparel_ecommerce", str(tmp_path / "nope.duckdb"), make_graph("memory"))


def test_identifier_allowlist_blocks_injection():
    for bad in ["", "Product; DROP", "1abc", "a b", "a-b", "`x`", "a)-[:X]-("]:
        with pytest.raises(ValueError):
            neo4j_ident(bad, "label")
        with pytest.raises(ValueError):
            loader_ident(bad, "label")
    assert neo4j_ident("Product", "label") == "Product"
    assert loader_ident("supplier_id", "key") == "supplier_id"
