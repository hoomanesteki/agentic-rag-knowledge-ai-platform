"""Turn raw records into chunks with metadata.

Records are dicts from a domain pack's unstructured seed (for example reviews). The field
names are read from the pack manifest, never hardcoded, so this stays domain agnostic.
Each chunk carries a content checksum so re-ingest can be idempotent.
"""
from __future__ import annotations

import hashlib

from adapters.base import Chunk


def split_text(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    """Word-window split with overlap. Short text stays a single chunk."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    pieces = []
    start = 0
    step = max(max_words - overlap, 1)
    while start < len(words):
        pieces.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
        start += step
    return pieces


def chunk_records(
    records: list[dict],
    *,
    id_field: str,
    text_field: str,
    lang_field: str | None = None,
    meta_fields: list[str] | None = None,
    doc_type: str = "doc",
    source: str = "unstructured",
    max_words: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    meta_fields = meta_fields or []
    chunks: list[Chunk] = []
    for rec in records:
        record_id = str(rec[id_field])
        pieces = split_text(rec[text_field], max_words, overlap)
        for i, piece in enumerate(pieces):
            chunk_id = record_id if len(pieces) == 1 else "{}#{}".format(record_id, i)
            metadata = {
                "source": source,
                "doc_type": doc_type,
                "record_id": record_id,
                "checksum": hashlib.sha256(piece.encode()).hexdigest(),
            }
            if lang_field and lang_field in rec:
                metadata["lang"] = rec[lang_field]
            for field_name in meta_fields:
                if field_name in rec:
                    metadata[field_name] = rec[field_name]
            chunks.append(Chunk(id=chunk_id, text=piece, metadata=metadata))
    return chunks
