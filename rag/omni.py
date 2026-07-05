"""The omni brain's single-turn fast path.

Every turn is routed by the cheap-first router, then answered by the one gated streaming pipeline
with the chosen lane's focus applied. Because the answer still flows through stream_answer, the
order-PII gate, safety intercepts, smalltalk handling, retrieval, and grounding are exactly the
linear path's. Routing adds specialization without standing up a second, weaker safety surface,
which is the PII-parity guarantee.

A genuinely ambiguous turn is not answered: the router returns a two-option clarify and the brain
asks the one question that splits them, instead of guessing. Multi-task turns are handled by the
heavy graph (rag/omni_graph.py); on this fast path the primary lane answers.
"""
from __future__ import annotations

import time
import uuid

from pipeline.answer import DEFAULT_TRACE_PATH, stream_answer, write_trace
from rag.roles import lane_persona, role_fragment
from rag.router import route


def _clarify_text(clarify: dict) -> str:
    a = clarify.get("a", "one thing")
    b = clarify.get("b", "something else")
    return ("Happy to help, and I want to point you the right way. Are you after {a}, or {b}?"
            .format(a=a, b=b))


def stream_omni(query, *, embedder, store, llm, reranker=None, metric_resolver=None,
                graph_retriever=None, lang=None, persona=None, history=None, concise=False,
                auth_identity=None, notes=None, message_id=None, small_llm=None, trace_path=None):
    """Yield the same event dicts as stream_answer (token chunks then one final), after routing."""
    signed_in = bool(auth_identity and auth_identity[0])
    decision = route(query, history=history, signed_in=signed_in, small_llm=small_llm)

    # ask, do not guess: a genuinely ambiguous turn gets the one question that splits it
    if decision.clarify is not None:
        mid = message_id or uuid.uuid4().hex
        text = _clarify_text(decision.clarify)
        write_trace({"ts": time.time(), "message_id": mid, "query": query, "lang": lang,
                     "tier": "clarify", "lane": decision.lane, "streamed": True,
                     "clarify_axis": decision.clarify.get("axis")},
                    trace_path or DEFAULT_TRACE_PATH)
        yield {"type": "token", "text": text}
        yield {"type": "final", "message_id": mid, "answer": text, "tier": "auto",
               "confidence": round(decision.confidence, 3), "grounding": 1.0, "citations": []}
        return

    # escalation speaks as the specialist persona; every other lane keeps the caller's persona
    effective_persona = "agent" if lane_persona(decision.lane) == "agent" else persona

    kwargs = dict(embedder=embedder, store=store, llm=llm, reranker=reranker,
                  metric_resolver=metric_resolver, graph_retriever=graph_retriever, lang=lang,
                  persona=effective_persona, history=history, concise=concise,
                  auth_identity=auth_identity, notes=notes, message_id=message_id,
                  role_fragment=role_fragment(decision.lane), lane=decision.lane)
    if trace_path is not None:
        kwargs["trace_path"] = trace_path
    yield from stream_answer(query, **kwargs)
