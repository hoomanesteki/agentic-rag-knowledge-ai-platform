"""The omni brain: route a turn, then answer it through the one gated pipeline.

Single-task turns take the fast path: route, then stream the chosen lane straight through
stream_answer, so the order-PII gate, safety intercepts, smalltalk, retrieval, and grounding are
exactly the linear path's. Routing adds specialization, not a second weaker safety surface, which
is the PII-parity guarantee.

Multi-task turns ("suggest a gift AND check my order") take the heavy path: split into clauses,
route and answer each through the same gated pipeline, then stitch the parts into one reply with a
complaint clause leading. A reroute budget lets one clause that abstained try the answers lane
once, and an output guard scans the stitched reply for a leaked email as a last-line tripwire.
This is deterministic control flow rather than a LangGraph Send fan-out; for a two or three clause
plan that is simpler to read and test, and the map-reduce is the documented scale-up.

A genuinely ambiguous turn is not answered at all: the router returns a two-option clarify and the
brain asks the one question that splits it.
"""
from __future__ import annotations

import re
import time
import uuid

from pipeline.answer import DEFAULT_TRACE_PATH, stream_answer, write_trace
from rag.roles import role_fragment
from rag.router import route

# Lanes that carry a real task. A clause that only routes to "answers" is not enough on its own to
# call a turn multi-task, so "a red and blue jacket" stays a single shopping turn.
_ACTIONABLE = ("complaint", "care", "stylist", "escalation")
_MAX_CLAUSES = 3  # cap the fan-out so a run-on sentence cannot spawn unbounded work

# split on a coordinating conjunction between clauses. Kept conservative, and a split only counts
# as multi-task when the clauses route to two or more different actionable lanes (checked below).
_CONJ = re.compile(r"\s*(?:\band then\b|\band also\b|\balso\b|\bthen\b|;|\band\b)\s*", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_SAFE_LEAK = ("I want to be careful with personal details here. Could you confirm the order number "
              "and the email on the order, and I'll pull it up securely?")


def _clarify_text(clarify: dict) -> str:
    a = clarify.get("a", "one thing")
    b = clarify.get("b", "something else")
    return ("Happy to help, and I want to point you the right way. Are you after {a}, or {b}?"
            .format(a=a, b=b))


def _split_clauses(query: str) -> list[str]:
    parts = [p.strip(" ,.;:") for p in _CONJ.split(query)]
    parts = [p for p in parts if len(p.split()) >= 2]  # drop fragments like "a red"
    return parts[:_MAX_CLAUSES] if len(parts) > 1 else [query]


def _output_leaks(text: str, auth_identity) -> bool:
    """A stitched reply is the one new surface the per-clause gate did not see as a whole, so scan
    it for an email that is not the authorized shopper's. Each clause is already gated, so this is
    defense in depth, not the primary control."""
    authorized = ""
    if auth_identity and len(auth_identity) > 1 and auth_identity[1]:
        authorized = auth_identity[1].lower()
    return any(m.lower() != authorized for m in _EMAIL.findall(text))


def _answer_once(clause: str, lane: str, deps: dict):
    """Answer one clause through the gated pipeline, buffered. Returns (text, final_event)."""
    parts, final = [], {}
    for ev in stream_answer(clause, role_fragment=role_fragment(lane), lane=lane, **deps):
        if ev.get("type") == "token":
            parts.append(ev.get("text", ""))
        elif ev.get("type") == "final":
            final = ev
    return ("".join(parts) or final.get("answer", "")), final


def _stream_multitask(routed, *, message_id, auth_identity, deps):
    # a complaint leads (empathy before anything else); the rest keep the order the shopper typed
    ordered = ([x for x in routed if x[1] == "complaint"]
               + [x for x in routed if x[1] != "complaint"])
    reroute_budget = 1
    answers = []
    for clause, lane in ordered:
        text, final = _answer_once(clause, lane, deps)
        if final.get("tier") == "abstain" and lane != "answers" and reroute_budget > 0:
            reroute_budget -= 1  # one clause may retry as the answers catch-all
            text, final = _answer_once(clause, "answers", deps)
        if text:
            answers.append(text)
    stitched = "\n\n".join(answers)
    if _output_leaks(stitched, auth_identity):
        stitched = _SAFE_LEAK
    mid = message_id or uuid.uuid4().hex
    yield {"type": "token", "text": stitched}
    yield {"type": "final", "message_id": mid, "answer": stitched, "tier": "auto",
           "confidence": 0.8, "grounding": 1.0, "citations": [], "lane": "multi"}


_ORDER_ID = re.compile(r"\b[A-Z]{2}\d{5,}\b")


def _find_order_ref(query: str, history) -> str:
    text = query + " " + " ".join(
        t.get("content", "") for t in (history or []) if t.get("role") == "user")
    m = _ORDER_ID.search(text)
    return m.group(0) if m else ""


def _escalate_handoff(query, *, message_id, auth_identity, history, lang, review_queue, domain,
                      trace_path):
    """The handoff to a person. The escalation specialist confirms what a human will need as a
    numbered list (only the shopper's OWN proven identity, so this echoes nothing they did not
    supply), files a case brief to the durable review queue, and asks if there is anything else.
    The human then picks up a ready case instead of starting cold, which is the AI helper doing the
    repetitive prep so the person spends their time on the resolution."""
    name = auth_identity[0] if (auth_identity and auth_identity[0]) else ""
    email = (auth_identity[1] if (auth_identity and len(auth_identity) > 1 and auth_identity[1])
             else "")
    order_ref = _find_order_ref(query, history)
    concern = query.strip()
    case_id = ""
    if review_queue is not None:
        brief = " | ".join(p for p in (
            "Escalation: " + concern,
            "shopper: " + name if name else "",
            "email: " + email if email else "",
            "order: " + order_ref if order_ref else "") if p)
        try:
            case_id = review_queue.enqueue(brief, domain=domain, message_id=message_id,
                                           route="escalation", lang=lang)
        except Exception:
            case_id = ""
    rows = [("Your concern", concern), ("Name on the account", name),
            ("Email on the order", email), ("Order reference", order_ref)]
    lines = ["I've got you, and I'll make this quick. Here is what I'm noting for our care "
             "specialist:"]
    for i, (label, val) in enumerate(rows, start=1):
        lines.append("{}. {}: {}".format(i, label, val or "please confirm"))
    ref = " (ref {})".format(case_id[:8]) if case_id else ""
    lines.append("I've logged this{} and a specialist will follow up by email. Is there anything "
                 "else you'd like me to add before I pass it over?".format(ref))
    text = "\n".join(lines)
    mid = message_id or uuid.uuid4().hex
    write_trace({"ts": time.time(), "message_id": mid, "query": query, "lang": lang,
                 "tier": "escalate", "lane": "escalation", "escalation_id": case_id,
                 "streamed": True}, trace_path or DEFAULT_TRACE_PATH)
    yield {"type": "token", "text": text}
    yield {"type": "final", "message_id": mid, "answer": text, "tier": "escalate",
           "confidence": 1.0, "grounding": 1.0, "citations": [], "escalation_id": case_id}


def stream_omni(query, *, embedder, store, llm, reranker=None, metric_resolver=None,
                graph_retriever=None, lang=None, persona=None, history=None, concise=False,
                auth_identity=None, notes=None, message_id=None, small_llm=None, trace_path=None,
                review_queue=None, domain=None):
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

    # A request to reach a person: the escalation specialist files a ready case brief and confirms.
    # This takes the whole turn, whatever the persona; a human request never fans out.
    if decision.lane == "escalation":
        yield from _escalate_handoff(query, message_id=message_id, auth_identity=auth_identity,
                                     history=history, lang=lang, review_queue=review_queue,
                                     domain=domain, trace_path=trace_path)
        return

    deps = dict(embedder=embedder, store=store, llm=llm, reranker=reranker,
                metric_resolver=metric_resolver, graph_retriever=graph_retriever, lang=lang,
                persona=persona, history=history, concise=concise, auth_identity=auth_identity,
                notes=notes)
    if trace_path is not None:
        deps["trace_path"] = trace_path

    # Already with the specialist after an earlier handoff: keep the specialist persona and answer
    # the routed lane's focus, rather than restarting the shopper with the assistant.
    if persona == "agent":
        yield from stream_answer(query, message_id=message_id,
                                 role_fragment=role_fragment(decision.lane), lane=decision.lane,
                                 **deps)
        return

    # Multi-task: split, and only fan out when the clauses hit two or more distinct actionable
    # lanes (so "a red and blue jacket" stays one shopping turn).
    clauses = _split_clauses(query)
    if len(clauses) > 1:
        routed = [(c, route(c).lane) for c in clauses]
        actionable = [(c, ln) for c, ln in routed if ln in _ACTIONABLE]
        if len({ln for _, ln in actionable}) >= 2:
            yield from _stream_multitask(actionable, message_id=message_id,
                                         auth_identity=auth_identity, deps=deps)
            return

    # Single-task fast path.
    yield from stream_answer(query, message_id=message_id,
                             role_fragment=role_fragment(decision.lane), lane=decision.lane,
                             **deps)
