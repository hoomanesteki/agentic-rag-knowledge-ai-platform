"""The enrichment pipeline turns untrusted reviews into governed features by consensus. These tests
pin the annotator, the voting rule (agreement and a support floor), the provenance, and that the
write is idempotent, so re-running the batch converges instead of accumulating duplicates."""
from data.enrichment import consensus, keyword_annotator, write_features


def test_keyword_annotator_reads_the_fit_signals():
    assert keyword_annotator("runs small, I sized up to a large")["value"] == "runs_small"
    assert keyword_annotator("true to size, fits perfectly")["value"] == "true_to_size"
    assert keyword_annotator("runs large and too baggy")["value"] == "runs_large"
    assert keyword_annotator("love the deep plum colour") is None


def test_consensus_needs_agreement_and_a_support_floor():
    reviews = [
        {"id": "1", "product_id": "P1", "text": "runs small, sized up"},
        {"id": "2", "product_id": "P1", "text": "too tight, size up"},
        {"id": "3", "product_id": "P2", "text": "runs small"},   # lone signal, no consensus
        {"id": "4", "product_id": "P3", "text": "lovely and warm"},
    ]
    rows = consensus(reviews, min_support=2)
    p1 = [r for r in rows if r["product_id"] == "P1"]
    assert p1 and p1[0]["value"] == "runs_small" and p1[0]["support"] == 2
    assert p1[0]["confidence"] == 1.0 and p1[0]["sources"] == ["1", "2"]  # provenance kept
    assert not [r for r in rows if r["product_id"] == "P2"]  # one review is not consensus


def test_consensus_withholds_when_reviews_disagree():
    reviews = [
        {"id": "1", "product_id": "P1", "text": "runs small"},
        {"id": "2", "product_id": "P1", "text": "runs large"},
    ]
    assert consensus(reviews, min_support=2) == []  # 1 vs 1, no winner clears the support floor


def test_write_features_is_idempotent(tmp_path):
    db = str(tmp_path / "e.duckdb")
    rows = [{"product_id": "P1", "aspect": "fit", "value": "runs_small", "confidence": 1.0,
             "support": 2, "total": 2, "sources": ["1", "2"]}]
    write_features(rows, db, computed_at="2026-01-01")
    write_features(rows, db, computed_at="2026-01-01")  # re-run the batch
    import duckdb
    con = duckdb.connect(db)
    count = con.execute("SELECT count(*) FROM product_features").fetchone()[0]
    con.close()
    assert count == 1  # replaced, not duplicated
