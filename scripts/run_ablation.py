#!/usr/bin/env python3
"""Run the retrieval ablation (dense vs hybrid vs hybrid+rerank) and write an eval report.

Run: make ablation   (needs keys in .env, Qdrant up, and an ingest done for real numbers)
Offline the store is empty, so the report is written as a labelled placeholder.

The rerank variant is labelled with its actual provider so a fake-lexical run is never
mistaken for a Voyage run. dense+rerank is intentionally omitted (rerank gain is attributed
against hybrid).

The +chunking column is not automated: re-ingest with and without the manifest context_fields
(make ingest), then re-run with a different output/label so you do not clobber the baseline:
    ABLATION_OUT=docs/eval-report-ctx.md ABLATION_LABEL=@ctx make ablation
and compare the two reports.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import yaml

from adapters.config import get_settings
from adapters.factory import make_embedder, make_reranker, make_store
from evaluation.golden import load_golden
from evaluation.harness import evaluate
from evaluation.report import build_ablation_report
from ingest.naming import collection_name

_TOP_K_IN = 50


def _entity_fields(pack: str) -> list[str]:
    with open(os.path.join(pack, "domain.yaml"), encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    fields = set()
    for src in (manifest.get("sources", {}) or {}).get("unstructured", []) or []:
        fields.update(src.get("entity_ref", []) or [])
    return sorted(fields)


def main() -> int:
    settings = get_settings()
    pack = os.path.join("domains", settings.domain)
    if not os.path.isfile(os.path.join(pack, "eval", "golden.jsonl")):
        print("no golden set at {}/eval/golden.jsonl".format(pack))
        return 1

    out_path = os.getenv("ABLATION_OUT", "docs/eval-report.md")
    suffix = os.getenv("ABLATION_LABEL", "")
    golden = load_golden(pack)
    entity_fields = _entity_fields(pack)
    store = make_store(collection=collection_name(settings.domain, settings.embed_model))
    embedder = make_embedder()

    # Label the reranker by its real provider; never silently pass a fake off as real.
    reranker = make_reranker()
    rerank_label = settings.rerank_provider
    if reranker is None:
        reranker = make_reranker("fake")
        rerank_label = "fake-lexical"

    variants = [
        ("dense", {"dense_only": True, "reranker": None}),
        ("hybrid", {"dense_only": False, "reranker": None}),
        ("hybrid+rerank({})".format(rerank_label), {"dense_only": False, "reranker": reranker}),
    ]
    results = []
    try:
        for label, cfg in variants:
            scorecard = evaluate(golden, embedder=embedder, store=store,
                                 entity_fields=entity_fields, reranker=cfg["reranker"],
                                 dense_only=cfg["dense_only"])
            results.append((label + suffix, scorecard))
    except RuntimeError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        print("hint: is Qdrant up (make up) and ingested (make ingest), and are the hosted "
              "APIs reachable and not rate-limited?", file=sys.stderr)
        return 1

    meta = {"embed": settings.embed_provider, "rerank": rerank_label, "top_k_in": _TOP_K_IN,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    note = ("Retrieval quality across variants on the measurable (qualitative) golden "
            "questions. The +chunking column needs a re-ingest (toggle the manifest "
            "context_fields) written to a separate report; compare the two.")
    if any(sc.get("degenerate") for _, sc in results):
        note = ("OFFLINE PLACEHOLDER: the index is empty, so retrieval numbers are zero and "
                "the gate trivially abstains on everything (so the gate columns read 1.000, "
                "which is not a real score). Run `make up && make ingest && make ablation` on "
                "a machine with the API keys to fill real numbers.\n\n" + note)

    report = build_ablation_report(results, domain=settings.domain, note=note, meta=meta)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("wrote {}".format(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
