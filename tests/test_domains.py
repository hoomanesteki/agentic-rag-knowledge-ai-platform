"""M4.4 dual-domain proof. The same engine must serve both domain packs with no engine code
change: ingest each pack's text, answer an in-domain question with citations, abstain on an
out-of-domain one, build each lakehouse to passing contracts, and mask every declared PII
column. This runs inside make check, so CI re-proves reproducibility on every commit.

It runs fully offline on the fakes (hash embedder, BM25 sparse, echo LLM). The hosted path is
exercised on the user's machine; here we prove the wiring is domain agnostic.
"""
import json
import os

import duckdb
import pytest
import yaml

from adapters.factory import make_embedder, make_llm, make_store
from data.contracts import check_contracts
from data.lakehouse import build_lakehouse
from data.metrics import MetricResolver
from ingest.chunk import chunk_records
from pipeline.answer import answer_question
from retrieval.sparse import SparseEncoder

# (domain, in-domain question, out-of-domain question, a lowercased substring the answer's
# evidence must contain so the in-domain assertion checks real retrieval, not a fallback).
DOMAINS = [
    ("apparel_ecommerce", "does the flow legging run small", "what is the capital of france",
     "small"),
    ("saas_support", "how do I reset my Northwind Cloud password", "what is the capital of france",
     "reset"),
]


def _manifest(domain: str) -> dict:
    with open(os.path.join("domains", domain, "domain.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pack_chunks(domain: str):
    pack = os.path.join("domains", domain)
    chunks = []
    for src in (_manifest(domain).get("sources", {}) or {}).get("unstructured", []) or []:
        rows = []
        with open(os.path.join(pack, src["file"]), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        chunks.extend(chunk_records(
            rows, id_field=src["id_field"], text_field=src["text_field"],
            lang_field=src.get("lang_field"), meta_fields=src.get("meta_fields", []),
            doc_type=src.get("doc_type", "doc"), context_fields=src.get("context_fields", [])))
    return chunks


def _seed_store(chunks):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    texts = [c.text for c in chunks]
    dense = embedder.embed(texts)
    store.upsert([
        {"id": c.id, "text": c.text, "payload": {**c.metadata, "chunk_id": c.id},
         "dense": d, "sparse": {"indices": s.indices, "values": s.values}}
        for c, d, s in zip(chunks, dense, [encoder.encode(t) for t in texts])
    ])
    return embedder, store


@pytest.mark.parametrize("domain, in_domain_q, out_of_domain_q, evidence", DOMAINS)
def test_same_engine_answers_and_abstains_per_domain(
        domain, in_domain_q, out_of_domain_q, evidence, tmp_path):
    chunks = _pack_chunks(domain)
    assert chunks, "domain {} has no unstructured chunks to ingest".format(domain)
    embedder, store = _seed_store(chunks)

    answered = answer_question(in_domain_q, embedder=embedder, store=store, llm=make_llm("fake"),
                               trace_path=str(tmp_path / "t.jsonl"))
    assert answered.tier == "auto", "{}: expected an in-domain answer".format(domain)
    # Non-vacuous: the offline fake LLM cites nothing, so asserting on citations alone would pass
    # on the all-contexts fallback. Assert retrieval actually surfaced the relevant chunk.
    assert any(evidence in (c["text"] or "").lower() for c in answered.contexts), \
        "{}: retrieval did not surface evidence '{}'".format(domain, evidence)

    abstained = answer_question(out_of_domain_q, embedder=embedder, store=store,
                                llm=make_llm("fake"), trace_path=str(tmp_path / "t.jsonl"))
    assert abstained.abstained, "{}: an out-of-domain question must abstain".format(domain)


@pytest.mark.parametrize("domain, _q, _ood, _ev", DOMAINS)
def test_lakehouse_builds_with_passing_contracts_per_domain(domain, _q, _ood, _ev, tmp_path):
    db = str(tmp_path / "lh.duckdb")
    built = build_lakehouse(domain, db)
    assert built, "{}: no gold tables built".format(domain)
    assert check_contracts(domain, db) == [], "{}: data contracts failed".format(domain)


@pytest.mark.parametrize("domain, _q, _ood, _ev", DOMAINS)
def test_governed_metric_layer_resolves_per_domain(domain, _q, _ood, _ev, tmp_path):
    # The governed metric layer must serve any pack: build the lakehouse, then resolve the
    # pack's first declared metric with no params (the "$x is null or ..." guard returns all
    # groups). Proves metrics are domain-agnostic, not just the first domain's.
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse(domain, db)
    resolver = MetricResolver(domain, db)
    names = resolver.names()
    assert names, "{}: pack declares no metrics".format(domain)
    result = resolver.resolve(names[0])
    assert result.rows, "{}: metric {} returned no rows".format(domain, names[0])
    assert any(cell is not None for row in result.rows for cell in row), \
        "{}: metric {} produced only nulls".format(domain, names[0])


@pytest.mark.parametrize("domain, _q, _ood, _ev", DOMAINS)
def test_declared_pii_is_masked_in_gold_per_domain(domain, _q, _ood, _ev, tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse(domain, db)
    con = duckdb.connect(db, read_only=True)
    try:
        for src in (_manifest(domain).get("sources", {}) or {}).get("structured", []) or []:
            for col in src.get("pii_columns", []) or []:
                rows = con.execute("select {} from {}".format(col, src["role"])).fetchall()
                leaked = [v for (v,) in rows if v is not None and not str(v).startswith("masked:")]
                assert not leaked, "{}.{} leaked raw PII into gold".format(src["role"], col)
    finally:
        con.close()
