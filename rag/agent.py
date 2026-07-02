"""M6.4 gate and bounded agent loop.

The gate reads a reconciled turn and decides its tier:
- auto: a confident, grounded answer with no unresolved conflict.
- agent: nothing confident yet, but the query looked answerable (some overlap), so try again.
- escalate: unanswerable (no relevant evidence), or a conflict the reconciler could not resolve.

For the agent tier the loop runs a bounded ReAct over the same specialists: reformulate the query,
re-dispatch, accumulate evidence, re-reconcile, up to a step cap. If a pass turns confident the
tier becomes auto; if the cap is hit while still weak, it escalates. Every extra pass is one small
reformulation call plus one synthesis, and the cap bounds the total, so a hard question cannot spin.
"""
from __future__ import annotations

import time
import uuid

from pipeline.answer import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_TRACE_PATH,
    AnswerResult,
    _content_tokens,
    _estimate_cost,
    write_trace,
)
from rag.supervisor import dispatch, reconcile
from rag.understand import heuristic_route, rewrite_followup

_ESCALATED = ("I do not have a confident answer for that from the available sources, so I have "
              "flagged it for a person to follow up.")
_MIN_AGENT_CONFIDENCE = 0.05  # below this the query matched nothing relevant; do not loop
_MAX_STEPS = 2

_REFORMULATE_SYSTEM = (
    "The previous search did not find a confident answer. Rewrite the question with different "
    "wording or more specific terms so a search finds better results. Reply with ONLY the query."
)


def decide_tier(reconcile_tier: str, *, conflict_resolved: bool, confidence: float,
                grounding: float, step: int, max_steps: int,
                min_agent_confidence: float = _MIN_AGENT_CONFIDENCE,
                min_grounding: float = 0.0) -> str:
    if reconcile_tier == "auto":
        if not conflict_resolved:
            return "escalate"  # a conflict the reconciler could not resolve needs a human
        if grounding < min_grounding and step < max_steps:
            return "agent"     # answered but weakly grounded; try once more for better support
        return "auto"
    # reconcile abstained
    if confidence <= min_agent_confidence or step >= max_steps:
        return "escalate"
    return "agent"


def reformulate(query: str, llm, prior: tuple = ()) -> str:
    """Propose a better search query, told what was already tried so it diversifies, accepted only
    if it shares a content word with the original (so an offline or off-the-rails model falls back
    to the original instead of drifting)."""
    if llm is None:
        return query
    prompt = "Question: " + query
    if prior:
        prompt += "\nAlready tried, avoid repeating: " + "; ".join(prior)
    prompt += "\nBetter search query:"
    try:
        text = llm.generate(prompt, system=_REFORMULATE_SYSTEM, max_tokens=60).text.strip()
    except Exception:
        return query
    if not text or not (set(_content_tokens(text)) & set(_content_tokens(query))):
        return query
    return text


def _confidence(findings: list, floor: float = 0.0) -> float:
    return round(max((f.confidence for f in findings), default=floor), 3)


def _context_ids(findings: list) -> set:
    return {c["id"] for f in findings for c in f.contexts}


def answer_with_agent(query: str, *, components: dict, history: list | None = None,
                      message_id: str | None = None, max_steps: int = _MAX_STEPS,
                      min_confidence: float = DEFAULT_MIN_CONFIDENCE, min_grounding: float = 0.0,
                      review_queue=None, domain: str | None = None, lang: str | None = None,
                      trace_path: str = DEFAULT_TRACE_PATH) -> AnswerResult:
    """Run the supervisor with the gate and bounded agent loop, returning an AnswerResult whose
    tier is auto or escalate. min_grounding (calibrated on real infra, off by default) makes a
    weakly-grounded auto answer take another pass. On escalate, if a review_queue is given, the
    question is enqueued for a human and the queue id is in the trace."""
    started = time.perf_counter()
    llm = components["llm"]
    message_id = message_id or uuid.uuid4().hex
    rewritten = rewrite_followup(query, history or [], llm)
    route = heuristic_route(rewritten)

    findings = dispatch(rewritten, components, min_confidence=min_confidence)
    result = reconcile(rewritten, findings, llm)
    confidence = _confidence(findings)
    seen_ids = _context_ids(findings)
    prior_queries: list[str] = []
    prompt_tokens, completion_tokens = result["prompt_tokens"], result["completion_tokens"]
    step = 0
    tier = decide_tier(result["tier"], conflict_resolved=result["conflict_resolved"],
                       confidence=confidence, grounding=result["grounding"], step=step,
                       max_steps=max_steps, min_grounding=min_grounding)

    while tier == "agent":
        step += 1
        reformulated = reformulate(rewritten, llm, prior=tuple(prior_queries))
        prior_queries.append(reformulated)
        fresh = dispatch(reformulated, components, min_confidence=min_confidence)
        if _context_ids(fresh) <= seen_ids:
            tier = "escalate"  # the pass surfaced no new evidence; more passes cannot help
            break
        seen_ids |= _context_ids(fresh)
        findings = findings + fresh
        result = reconcile(rewritten, findings, llm)
        prompt_tokens += result["prompt_tokens"]
        completion_tokens += result["completion_tokens"]
        confidence = _confidence(findings, confidence)
        tier = decide_tier(result["tier"], conflict_resolved=result["conflict_resolved"],
                           confidence=confidence, grounding=result["grounding"], step=step,
                           max_steps=max_steps, min_grounding=min_grounding)

    answer = result["answer"]
    citations = result["citations"]
    if tier == "escalate" and result["tier"] == "abstain":
        answer, citations = _ESCALATED, []  # honest hand-off, not a guessed answer

    escalation_id = None
    if tier == "escalate" and review_queue is not None:
        escalation_id = review_queue.enqueue(rewritten, domain=domain, message_id=message_id,
                                             route=route, lang=lang)

    trace = {
        "ts": time.time(), "message_id": message_id, "raw_query": query, "query": rewritten,
        "lang": lang, "route": route, "specialists": sorted({f.specialist for f in findings}),
        "tier": tier, "agent_steps": step, "escalation_id": escalation_id,
        "conflict": result["conflict"],
        "conflict_resolved": result["conflict_resolved"],
        "reranked": components.get("reranker") is not None,
        "retrieved": [{"id": c["id"], "score": c["score"]} for c in result["contexts"]],
        "metric": any(f.kind == "metric" for f in findings),
        "graph": any(f.kind == "graph" for f in findings),
        "confidence": confidence, "grounding": round(result["grounding"], 3),
        "model": result["model"], "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": _estimate_cost(result["model"], prompt_tokens, completion_tokens),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    write_trace(trace, trace_path)
    return AnswerResult(answer=answer, tier=tier, confidence=confidence,
                        grounding=result["grounding"], citations=citations,
                        contexts=result["contexts"], trace=trace)
