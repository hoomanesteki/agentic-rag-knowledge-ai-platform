"""Human-triggered knowledge consolidation: distill recent traffic into a PROPOSED knowledge pack a
person approves, so the assistant gets smarter from real conversations without ever silently
mutating its own memory.

The only new knowledge is human-verified: candidate chunks come from answers an operator already
closed in the review queue, never an LLM rewrite of existing verified content. Repeat questions,
thumbs-down clusters, and abstain patterns are surfaced as GAPS to fill, not auto-answered. The
output is a proposal file; a human edits and approves it, and only then does the existing flywheel
(reindex_verified, grow_verified_eval) index it. We deduplicate rather than roll-summarize, because
continuously letting an LLM rewrite stored memory corrupts it.
"""
from __future__ import annotations

from collections import Counter


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def propose_pack(traces: list[dict], feedback: list[dict], closed_reviews: list[dict], *,
                 min_repeats: int = 3, top_gaps: int = 20) -> dict:
    """Build a proposed knowledge pack from recent signals. Pure and deterministic, so it is
    unit-tested with no infrastructure. Returns candidate (human-verified) chunks and eval rows plus
    the gaps worth a human's attention, all deduplicated."""
    seen_q: set = set()
    candidate_chunks: list[dict] = []
    candidate_eval_rows: list[dict] = []
    for r in closed_reviews:
        question = r.get("question", "")
        answer = r.get("answer", "")
        key = _norm(question)
        if not key or not answer or key in seen_q:
            continue
        seen_q.add(key)
        candidate_chunks.append({"text": answer, "source": "hitl-consolidated",
                                 "question": question, "lang": r.get("lang", "")})
        candidate_eval_rows.append({"question": question, "type": "answerable",
                                    "lang": r.get("lang", ""), "source": "consolidation"})

    down_ids = {f.get("message_id") for f in feedback if f.get("verdict") == "down"}
    gaps: Counter = Counter()
    for t in traces:
        q = _norm(t.get("query", ""))
        if q and (t.get("tier") == "abstain" or t.get("message_id") in down_ids):
            gaps[q] += 1

    repeats: Counter = Counter(_norm(t.get("query", "")) for t in traces if t.get("query"))
    frequent = sorted(
        ({"query": q, "count": c} for q, c in repeats.items() if q and c >= min_repeats),
        key=lambda x: -x["count"])

    return {
        # candidate_chunks are human-verified answers only, never an LLM rewrite
        "candidate_chunks": candidate_chunks,
        "candidate_eval_rows": candidate_eval_rows,
        "knowledge_gaps": [{"query": q, "count": c} for q, c in gaps.most_common(top_gaps)],
        "frequent_queries": frequent,
        "counts": {"chunks": len(candidate_chunks), "eval_rows": len(candidate_eval_rows),
                   "gaps": len(gaps), "frequent": len(frequent)},
        "notes": ["PROPOSED only: a human reviews and approves before the flywheel indexes it.",
                  "Candidate chunks come from human-verified review answers, never an LLM rewrite "
                  "of existing knowledge, so self-updates cannot corrupt stored memory."],
    }
