"""M8.3 CI eval gate: run the pipeline on recorded fixtures and block a regression.

The fixtures are neutral synthetic Q&A (no domain vocabulary), so the gate runs fully offline on
the fakes and is deterministic: each fixture either retrieves the expected fact or correctly
abstains. A cheap lexical judge (does a retrieved context contain the expected keyword, or did it
abstain) needs no LLM, so this runs in CI. If a change drops a fixture below the score threshold,
the gate fails and blocks the merge.
"""
from __future__ import annotations

import json

from adapters.factory import make_embedder, make_llm, make_store
from pipeline.answer import answer_question
from retrieval.sparse import SparseEncoder


def load_gate(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _seed_store(corpus: list, embedder, store) -> None:
    encoder = SparseEncoder()
    dense = embedder.embed(corpus)
    store.upsert([
        {"id": "F{}".format(i), "text": text, "payload": {"doc_type": "fixture"},
         "dense": dense[i],
         "sparse": {"indices": encoder.encode(text).indices,
                    "values": encoder.encode(text).values}}
        for i, text in enumerate(corpus)])


def _judge(fixture: dict, result) -> bool:
    if fixture["expect"] == "abstain":
        return result.abstained
    if fixture["expect"] == "retrieves":
        value = fixture["value"].lower()
        return (not result.abstained
                and any(value in (c.get("text") or "").lower() for c in result.contexts))
    return False


def run_gate(gate: dict, *, trace_path: str, min_score: float = 1.0) -> dict:
    """Run every fixture through the offline pipeline and score it. passed is score >= min_score;
    the default 1.0 blocks on any single failed fixture (a real regression). top_k is small so a
    retrieves-fixture tests ranking, not just the abstain gate. trace_path is required so the gate
    never writes to the real traces that feed MLflow and drift."""
    embedder, store, llm = make_embedder("fake"), make_store("memory"), make_llm("fake")
    _seed_store(gate["corpus"], embedder, store)
    results = []
    for fixture in gate["fixtures"]:
        result = answer_question(fixture["query"], embedder=embedder, store=store, llm=llm,
                                 top_k=2, trace_path=trace_path)
        results.append({"id": fixture["id"], "expect": fixture["expect"],
                        "passed": _judge(fixture, result)})
    passed = sum(1 for r in results if r["passed"])
    score = round(passed / len(results), 3) if results else 0.0
    return {"score": score, "passed": score >= min_score, "min_score": min_score,
            "results": results}
