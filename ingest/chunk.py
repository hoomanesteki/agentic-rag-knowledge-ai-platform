"""Turn raw records into chunks with metadata.

Records are dicts from a domain pack's unstructured seed (for example reviews). The field
names are read from the pack manifest, never hardcoded, so this stays domain agnostic.

Chunking is sentence and token aware: text is split on sentence boundaries and packed to a
token budget with a sentence of overlap, so a chunk is a coherent unit rather than an
arbitrary word window. (Paragraph and heading structure is not preserved yet; whitespace is
collapsed first.) An optional contextual prefix (built from manifest-declared fields) is
stored in metadata["context"] so the ingest step can embed context-plus-text while the stored
and displayed text stays clean. Each chunk gets a content checksum.
"""
from __future__ import annotations

import hashlib
import re

from adapters.base import Chunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    # Rough word-to-token factor; good enough for budgeting without a tokenizer dependency.
    return max(1, round(len(text.split()) * 1.3))


def split_text(text: str, max_tokens: int = 180, overlap_sentences: int = 1) -> list[str]:
    """Split into sentence-packed chunks under a token budget, overlapping by whole sentences.
    Short text stays a single chunk."""
    text = " ".join(text.split())
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        tokens = _estimate_tokens(sentence)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(_estimate_tokens(s) for s in current)
        current.append(sentence)
        current_tokens += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _context_prefix(rec: dict, context_fields: list[str]) -> str:
    parts = ["{}={}".format(f, rec[f]) for f in context_fields if rec.get(f) not in (None, "")]
    return "[{}]".format("; ".join(parts)) if parts else ""


def chunk_records(
    records: list[dict],
    *,
    id_field: str,
    text_field: str,
    lang_field: str | None = None,
    meta_fields: list[str] | None = None,
    doc_type: str = "doc",
    source: str = "unstructured",
    context_fields: list[str] | None = None,
    max_tokens: int = 180,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    meta_fields = meta_fields or []
    context_fields = context_fields or []
    chunks: list[Chunk] = []
    for rec in records:
        record_id = str(rec[id_field])
        text = (rec.get(text_field) or "").strip()
        if not text:
            continue  # skip empty or whitespace-only records
        prefix = _context_prefix(rec, context_fields)
        pieces = split_text(text, max_tokens, overlap_sentences)
        for i, piece in enumerate(pieces):
            chunk_id = record_id if len(pieces) == 1 else "{}#{}".format(record_id, i)
            metadata = {
                "source": source,
                "doc_type": doc_type,
                "record_id": record_id,
                "checksum": hashlib.sha256(piece.encode()).hexdigest(),
            }
            if prefix:
                metadata["context"] = prefix  # embedded with the text; not part of stored text
            if lang_field and lang_field in rec:
                metadata["lang"] = rec[lang_field]
            for field_name in meta_fields:
                if field_name in rec:
                    metadata[field_name] = rec[field_name]
            chunks.append(Chunk(id=chunk_id, text=piece, metadata=metadata))
    return chunks
