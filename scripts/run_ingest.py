#!/usr/bin/env python3
"""Ingest the active domain's unstructured text into Qdrant with dense + sparse vectors.

Reads DOMAIN from .env, loads the pack manifest, chunks each unstructured source, embeds
dense with the configured embedder (EMBED_PROVIDER) and sparse with BM25, and upserts into a
Qdrant collection named for the domain and embedding model. Idempotent: re-running overwrites
the same points.

Run: make ingest   (needs the embedder API key in .env and Qdrant up via make up)
"""
from __future__ import annotations

import json
import os
import sys

import yaml

from adapters.config import get_settings
from adapters.factory import make_embedder, make_store
from ingest.chunk import chunk_records
from ingest.naming import collection_name
from retrieval.sparse import SparseEncoder


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    settings = get_settings()
    pack = os.path.join("domains", settings.domain)
    manifest_path = os.path.join(pack, "domain.yaml")
    if not os.path.isfile(manifest_path):
        print("no domain pack at {}".format(pack))
        return 1
    if settings.embed_provider in ("fake", ""):
        print("set EMBED_PROVIDER (cohere or voyage) and its API key in .env to ingest for real")
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    sources = (manifest.get("sources", {}) or {}).get("unstructured", []) or []

    chunks = []
    for src in sources:
        rows = _load_jsonl(os.path.join(pack, src["file"]))
        chunks.extend(chunk_records(
            rows,
            id_field=src["id_field"],
            text_field=src["text_field"],
            lang_field=src.get("lang_field"),
            meta_fields=src.get("meta_fields", []),
            doc_type=src.get("doc_type", "doc"),
            context_fields=src.get("context_fields", []),
        ))
    if not chunks:
        print("no unstructured records to ingest")
        return 0

    # Embed the context prefix plus the text, but store and display the clean text only, so
    # the prefix helps retrieval without polluting citations or the confidence gate.
    embed_texts = [(c.metadata.get("context", "") + " " + c.text).strip() for c in chunks]
    embedder = make_embedder()
    dense = embedder.embed(embed_texts, input_type="document")
    sparse_encoder = SparseEncoder()
    sparse = [sparse_encoder.encode(t) for t in embed_texts]

    collection = collection_name(settings.domain, embedder.model)
    store = make_store("qdrant", collection=collection)
    store.ensure_collection(embedder.dim)
    store.upsert([
        {"id": c.id, "text": c.text, "payload": c.metadata,
         "dense": d, "sparse": {"indices": s.indices, "values": s.values}}
        for c, d, s in zip(chunks, dense, sparse)
    ])
    print("ingested {} chunks into {}".format(len(chunks), collection))
    return 0


if __name__ == "__main__":
    sys.exit(main())
