"""M7.3 the flywheel: turn resolved human answers into knowledge and tune the gate.

A resolved review-queue item (a question a person answered) becomes (1) a verified chunk indexed
into the same vector store the retriever uses, so the next similar question retrieves the human
answer with provenance, and (2) a new answerable entry in a verified eval set that grows the
golden coverage. Thumbs feedback suggests a gate threshold. Nothing here names a domain; items
carry their own domain.
"""
from __future__ import annotations

import json
import os

from retrieval.sparse import SparseEncoder

_MIN_THRESHOLD, _MAX_THRESHOLD = 0.2, 0.6


def _verified_id(item: dict) -> str:
    return "verified:" + item["id"]


def reindex_verified(items: list[dict], embedder, store,
                     encoder: SparseEncoder | None = None) -> int:
    """Index each resolved answer as a retrievable verified chunk (idempotent by id). The question
    and answer are embedded together so the question matches; the clean answer is what is stored
    and shown."""
    items = [it for it in items if (it.get("answer") or "").strip()]
    if not items:
        return 0
    encoder = encoder or SparseEncoder()
    embed_texts = ["{} {}".format(it["question"], it["answer"]).strip() for it in items]
    dense = embedder.embed(embed_texts, input_type="document")
    points = []
    for item, emb_text, vector in zip(items, embed_texts, dense):
        sparse = encoder.encode(emb_text)
        vid = _verified_id(item)
        points.append({
            "id": vid, "text": item["answer"],
            "payload": {"doc_type": "verified", "source": "hitl", "chunk_id": vid,
                        "question": item["question"], "answered_by": item.get("answered_by"),
                        "domain": item.get("domain")},
            "dense": vector, "sparse": {"indices": sparse.indices, "values": sparse.values}})
    store.upsert(points)
    return len(points)


def grow_verified_eval(items: list[dict], path: str) -> int:
    """Append resolved Q&A as answerable entries to a verified eval set, skipping ones already
    written so a re-run does not duplicate. Kept separate from the curated golden.jsonl."""
    existing: set = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue
    written = 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            if not (item.get("answer") or "").strip():
                continue  # a blank answer is not a usable eval row
            gid = "V-" + item["id"]
            if gid in existing:
                continue
            f.write(json.dumps({
                "id": gid, "lang": item.get("lang") or "unknown", "question": item["question"],
                "answer": item["answer"], "type": "answerable", "route": "factual",
                "source": "hitl"}, ensure_ascii=False) + "\n")
            written += 1
    return written


def suggest_threshold(quality: dict, current: float) -> dict:
    """Suggest a gate threshold from thumbs: a high thumbs-down rate means the gate is answering
    when it should not, so raise it; a clean up-rate means it can be a touch more permissive.
    Advisory only, clamped; a human applies it."""
    overall = quality.get("overall", {})
    up, down = overall.get("thumbs_up", 0), overall.get("thumbs_down", 0)
    rated = up + down
    if rated < 5:
        return {"suggested": current, "reason": "not enough thumbs yet ({})".format(rated),
                "down_rate": None}
    down_rate = round(down / rated, 3)
    if down_rate > 0.3:
        suggested, reason = current + 0.05, "high thumbs-down rate; answer more cautiously"
    elif down_rate < 0.1:
        suggested, reason = current - 0.02, "answers rate well; can be slightly more permissive"
    else:
        suggested, reason = current, "thumbs are healthy; hold the threshold"
    suggested = round(min(_MAX_THRESHOLD, max(_MIN_THRESHOLD, suggested)), 3)
    return {"suggested": suggested, "reason": reason, "down_rate": down_rate}
