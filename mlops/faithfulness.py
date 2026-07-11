"""The online faithfulness ladder: a cheap-first hallucination detector on live traffic.

The lexical grounding_score is citation SUPPORT, not faithfulness, so it cannot see a fabricated but
cited claim. This samples answered turns and runs the repo's own RAGAS faithfulness judge on them
(decompose the answer into atomic claims, verify each against the retrieved context, score =
supported / total) with a DIFFERENT-family judge, so it does not grade its own style up.

The ladder keeps it cheap and catches the right turns: EVERY low-grounding turn is checked (the ones
most likely to be wrong) plus a small deterministic fraction of the rest, so a regression that lifts
groundedness while lowering faithfulness still gets caught. It never blocks streaming: the request
path only appends a candidate to a queue file AFTER the answer is sent, and the scorer drains that
queue offline. This is the shape of AWS Bedrock's contextual grounding check, done lean on Groq.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

DEFAULT_LOW_GROUNDING = 0.7
DEFAULT_SAMPLE_RATE = 0.08  # ~8% of otherwise-healthy turns get a spot check
QUEUE_PATH = os.getenv("FAITHFULNESS_QUEUE", "traces/faithfulness_queue.jsonl")
SCORES_PATH = os.getenv("FAITHFULNESS_SCORES", "traces/faithfulness.jsonl")


def _enabled() -> bool:
    return os.getenv("FAITHFULNESS_SAMPLING", "off").strip().lower() in ("1", "true", "on", "yes")


def _sampled(message_id: str | None, sample_rate: float) -> bool:
    """Deterministic sampling from the message id: reproducible (no RNG), uniform, and stable so a
    turn is either always or never sampled regardless of when the ladder runs."""
    if not message_id or sample_rate <= 0:
        return False
    h = int(hashlib.sha256(message_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < sample_rate


def check_reason(grounding: float | None, message_id: str | None, *,
                 low_grounding: float = DEFAULT_LOW_GROUNDING,
                 sample_rate: float = DEFAULT_SAMPLE_RATE) -> str | None:
    """Why this answered turn should get a faithfulness check, or None. Low-grounding turns are
    always checked; the rest are sampled at a small rate."""
    if grounding is not None and grounding < low_grounding:
        return "low_grounding"
    if _sampled(message_id, sample_rate):
        return "sampled"
    return None


def enqueue_candidate(message_id, question, answer, contexts, grounding, *,
                      queue_path: str | None = None) -> str | None:
    """Append a faithfulness candidate to the queue if sampling is enabled and the ladder selects
    this turn. A cheap file append after the answer is already sent, so it never blocks streaming.
    Returns the reason it was enqueued, or None."""
    if not _enabled():
        return None
    reason = check_reason(grounding, message_id)
    if reason is None:
        return None
    rec = {"message_id": message_id, "question": question, "answer": answer,
           "contexts": [(c.get("text", "") if isinstance(c, dict) else str(c))
                        for c in (contexts or [])],
           "grounding": grounding, "reason": reason, "ts": time.time()}
    path = queue_path or QUEUE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return reason


def score_candidate(judge, candidate: dict) -> float | None:
    """Faithfulness of one queued candidate via the repo's RAGAS faithfulness (claim-level
    verification against the retrieved context). Lazy import so the request path never pulls it."""
    from evaluation.ragas_eval import faithfulness
    context = "\n".join(candidate.get("contexts") or [])
    return faithfulness(judge, candidate.get("answer", ""), context)


def drain_queue(judge, *, queue_path: str | None = None, scores_path: str | None = None) -> dict:
    """Score every queued candidate and append the results to the scores log. Returns a summary.
    Offline: run from a scheduled job or `make faithfulness`, never in the request path."""
    qpath = queue_path or QUEUE_PATH
    spath = scores_path or SCORES_PATH
    if not os.path.exists(qpath):
        return {"scored": 0, "flagged": 0, "mean": None}
    scored, faiths = [], []
    with open(qpath, encoding="utf-8") as f:
        candidates = [json.loads(ln) for ln in f if ln.strip()]
    for cand in candidates:
        f_score = score_candidate(judge, cand)
        rec = {"message_id": cand.get("message_id"), "grounding": cand.get("grounding"),
               "faithfulness": f_score, "reason": cand.get("reason"), "ts": time.time()}
        scored.append(rec)
        if f_score is not None:
            faiths.append(f_score)
    os.makedirs(os.path.dirname(spath) or ".", exist_ok=True)
    with open(spath, "a", encoding="utf-8") as f:
        for rec in scored:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    open(qpath, "w").close()  # clear the drained queue
    flagged = sum(1 for v in faiths if v < 0.8)
    return {"scored": len(scored), "flagged": flagged,
            "mean": round(sum(faiths) / len(faiths), 4) if faiths else None}
