#!/usr/bin/env python3
"""Ingest the active domain's unstructured text into Qdrant with dense + sparse vectors.

Reads DOMAIN from .env, loads the pack manifest, chunks each unstructured source, embeds
dense with Voyage and sparse with BM25, and upserts into a Qdrant collection named for the
domain and embedding model. Idempotent: re-running overwrites the same points.

Run: make ingest   (needs VOYAGE_API_KEY in .env and Qdrant up via make up)
"""
from __future__ import annotations

import json
import os
import re
import sys

import yaml

from adapters.config import get_settings
from adapters.qdrant_store import QdrantStore
from adapters.voyage import VoyageEmbedder
from ingest.chunk import chunk_records
from retrieval.sparse import Bm25SparseEncoder


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collection_name(domain: str, model: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", "{}__{}".format(domain, model).lower()).strip("_")
    return "{}__v1".format(safe)


def main() -> int:
    settings = get_settings()
    pack = os.path.join("domains", settings.domain)
    manifest_path = os.path.join(pack, "domain.yaml")
    if not os.path.isfile(manifest_path):
        print("no domain pack at {}".format(pack))
        return 1
    if settings.embed_provider != "voyage" or not settings.voyage_api_key:
        print("set EMBED_PROVIDER=voyage and VOYAGE_API_KEY in .env to ingest for real")
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
        ))
    if not chunks:
        print("no unstructured records to ingest")
        return 0

    texts = [c.text for c in chunks]
    embedder = VoyageEmbedder()
    dense = embedder.embed(texts, input_type="document")
    sparse_encoder = Bm25SparseEncoder().fit(texts)
    sparse = [sparse_encoder.encode(t) for t in texts]

    collection = _collection_name(settings.domain, embedder.model)
    store = QdrantStore(collection=collection)
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
