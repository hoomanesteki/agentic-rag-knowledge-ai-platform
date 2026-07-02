"""M7.4 read-only domain views: ontology, governed metrics, and medallion lineage, plus the
knowledge-gap worklist. Domain agnostic and free of any SQL template or raw PII."""
from data.introspect import lineage_view, metrics_view, ontology_view
from evaluation.monitoring import aggregate_gaps


def test_ontology_view_lists_entities_and_edges():
    view = ontology_view("apparel_ecommerce")
    assert "Product" in view["entity_types"]
    labels = {n["label"] for n in view["nodes"]}
    assert {"Product", "Supplier"} <= labels
    edge_types = {e["type"] for e in view["edges"]}
    assert "SUPPLIES" in edge_types


def test_metrics_view_is_metadata_only():
    view = metrics_view("apparel_ecommerce")
    names = {m["name"] for m in view}
    assert "return_rate_by_size" in names
    for m in view:
        assert "sql_template" not in m  # the template is never exposed
        assert isinstance(m["params"], list)


def test_lineage_view_shows_medallion_and_metric_use():
    view = lineage_view("apparel_ecommerce")
    roles = {r["role"]: r for r in view["medallion"]}
    assert "sales" in roles
    assert roles["sales"]["layers"] == ["bronze_sales", "silver_sales", "sales"]
    # the return-rate metric reads the sales gold table
    assert "return_rate_by_size" in roles["sales"]["metrics"]


def test_lineage_view_flags_pii_columns():
    view = lineage_view("saas_support")
    tickets = next(r for r in view["medallion"] if r["role"] == "tickets")
    assert "customer_email" in tickets["pii_columns"]


def test_partial_manifest_does_not_crash(tmp_path):
    # a graph node/edge missing required keys is skipped, not a 500
    pack = tmp_path / "domains" / "broken"
    pack.mkdir(parents=True)
    (pack / "domain.yaml").write_text(
        "name: broken\nlanguages: [en]\nentity_types: [Thing]\nsources: {}\n"
        "graph:\n  nodes:\n    - {label: Thing, key: id}\n    - {source: x}\n"
        "  edges:\n    - {type: REL, from: Thing, to: Thing}\n    - {type: BAD}\n")
    view = ontology_view("broken", domains_dir=str(tmp_path / "domains"))
    assert [n["label"] for n in view["nodes"]] == ["Thing"]  # the keyless node dropped
    assert [e["type"] for e in view["edges"]] == ["REL"]      # the incomplete edge dropped


def test_aggregate_gaps_counts_and_masks():
    traces = [
        {"tier": "abstain", "query": "What is the SLA", "lang": "en"},
        {"tier": "escalate", "query": "what is the SLA", "lang": "en"},  # same, different case
        {"tier": "abstain", "query": "email me at a@b.com"},
        {"tier": "auto", "query": "how much does it cost"},
    ]
    gaps = aggregate_gaps(traces)
    top = gaps[0]
    assert top["count"] == 2 and top["lang"] == "en"  # casefolded together
    masked = next(g for g in gaps if "email me" in g["question"])
    assert "<email>" in masked["question"] and "a@b.com" not in masked["question"]
    assert all(g["question"] != "how much does it cost" for g in gaps)
