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

from adapters.budget import BudgetedLLM, BudgetExceeded, TurnBudget
from pipeline.answer import DEFAULT_TRACE_PATH, stream_answer, write_trace
from rag.roles import role_fragment
from rag.router import route

# Lanes that carry a real task. A clause that only routes to "answers" is not enough on its own to
# call a turn multi-task, so "a red and blue jacket" stays a single shopping turn.
_ACTIONABLE = ("complaint", "care", "stylist", "escalation")
_MAX_CLAUSES = 3  # cap the fan-out so a run-on sentence cannot spawn unbounded work

# split on a coordinating conjunction between clauses. Kept conservative, and a split only counts
# as multi-task when the clauses route to two or more different actionable lanes (checked below).
_CONJ = re.compile(r"\s*(?:\band then\b|\band also\b|\balso\b|\bthen\b|;|,|\band\b)\s*", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_SAFE_LEAK = ("I want to be careful with personal details here. Could you confirm the order number "
              "and the email on the order, and I'll pull it up securely?")


def _clarify_text(clarify: dict) -> str:
    a = clarify.get("a", "one thing")
    b = clarify.get("b", "something else")
    return ("Happy to help, and I want to point you the right way. Are you after {a}, or {b}?"
            .format(a=a, b=b))


def _split_clauses(query: str) -> list[str]:
    parts = [p.strip(" ,.;:") for p in _CONJ.split(query or "")]
    parts = [p for p in parts if len(p.split()) >= 2]  # drop fragments like "a red"
    return parts if len(parts) > 1 else [query]  # the cap is applied after complaint-first ordering


def _output_leaks(text: str, auth_identity) -> bool:
    """A stitched reply is the one new surface the per-clause gate did not see as a whole. Each
    clause is already gated to the signed-in shopper's own data, so for a signed-in shopper this
    tripwire stands down: their own reply may legitimately contain their own or a gift recipient's
    email, and replacing it would be a false positive. For an ANONYMOUS turn, gated retrieval must
    never surface an order email, so any email in the stitched reply is unexpected and trips it."""
    if auth_identity and auth_identity[0]:
        return False
    return bool(_EMAIL.search(text))


def _answer_once(clause: str, lane: str, deps: dict):
    """Answer one clause through the gated pipeline, buffered. Returns (text, final_event)."""
    parts, final = [], {}
    for ev in stream_answer(clause, role_fragment=role_fragment(lane), lane=lane, **deps):
        if ev.get("type") == "token":
            parts.append(ev.get("text", ""))
        elif ev.get("type") == "final":
            final = ev
    return ("".join(parts) or final.get("answer", "")), final


def _merge_citations(finals) -> list:
    """Union the contributing clauses' citations for the stitched reply, de-duplicated in
    first-seen order (by citation id when present, else by value)."""
    seen: set = set()
    merged: list = []
    for f in finals:
        for c in (f.get("citations") or []):
            key = c.get("id") if isinstance(c, dict) else c
            if key is None:
                key = repr(c)
            if key not in seen:
                seen.add(key)
                merged.append(c)
    return merged


def _emit_with_budget(events, budget):
    """Pass answer events through, stamping the turn's budget snapshot onto the final so the spend
    is queryable. If a budget breach ends the stream before a final, emit a safe final instead of
    letting the turn die mid-flight."""
    saw_final = False
    try:
        for ev in events:
            if ev.get("type") == "final":
                saw_final = True
                ev = {**ev, "budget": budget.snapshot()}
            yield ev
    except BudgetExceeded as exc:
        if not saw_final:
            text = ("Let me keep this quick and accurate and bring in a specialist to help you "
                    "further.")
            yield {"type": "token", "text": text}
            yield {"type": "final", "message_id": uuid.uuid4().hex, "answer": text,
                   "tier": "abstain", "confidence": 0.0, "grounding": 0.0, "citations": [],
                   "lane": "budget", "budget": {**budget.snapshot(), "stopped": exc.reason}}


def _answers_with_cache(query, *, lane, deps, budget, answer_cache, message_id):
    """The answers lane with a semantic cache in front. A near-identical earlier FAQ is served
    without retrieval or generation; a miss streams normally and, if the answer is grounded and not
    an abstain, is stored for next time. Only reached for anonymous answers-lane turns, so nothing
    personalized or order-specific is ever cached."""
    hit = answer_cache.get(query)
    if hit is not None:
        mid = message_id or uuid.uuid4().hex
        yield {"type": "token", "text": hit["answer"]}
        yield {"type": "final", "message_id": mid, "answer": hit["answer"], "tier": "auto",
               "confidence": hit.get("confidence") or 0.0, "grounding": hit.get("grounding") or 0.0,
               "citations": hit.get("citations") or [], "lane": lane, "cache_hit": True,
               "cache_similarity": hit.get("similarity"), "budget": budget.snapshot()}
        return
    final_ev = None
    for ev in _emit_with_budget(
            stream_answer(query, message_id=message_id, role_fragment=role_fragment(lane),
                          lane=lane, **deps), budget):
        if ev.get("type") == "final":
            final_ev = ev
        yield ev
    # store only a grounded, non-abstained answer, so the cache never serves a hedge or a refusal
    if final_ev and final_ev.get("tier") == "auto" and (final_ev.get("grounding") or 0.0) > 0.0:
        answer_cache.put(query, final_ev.get("answer", ""), final_ev.get("citations", []),
                         grounding=final_ev.get("grounding"), confidence=final_ev.get("confidence"))


def _stream_multitask(routed, *, message_id, auth_identity, deps, budget):
    # answer each distinct LANE once (so a single complaint split across two comma clauses does not
    # double-apologize), with a complaint lane leading, then the rest in typed order, capped after.
    by_lane: dict = {}
    for clause, lane in routed:
        by_lane.setdefault(lane, clause)  # keep the first clause seen for each lane
    lanes_ordered = ([ln for ln in by_lane if ln == "complaint"]
                     + [ln for ln in by_lane if ln != "complaint"])[:_MAX_CLAUSES]
    ordered = [(by_lane[ln], ln) for ln in lanes_ordered]
    reroute_budget = 1
    answers = []
    finals = []  # the real per-clause finals, so the stitched reply reports measured telemetry
    for clause, lane in ordered:
        try:
            text, final = _answer_once(clause, lane, deps)
            if final.get("tier") == "abstain" and lane != "answers" and reroute_budget > 0:
                reroute_budget -= 1  # one clause may retry as the answers catch-all
                text, final = _answer_once(clause, "answers", deps)
        except BudgetExceeded:
            break  # out of budget: finish with the clauses already answered, never loop on
        if text:
            answers.append(text)
            finals.append(final)
    stitched = "\n\n".join(answers)
    leaked = _output_leaks(stitched, auth_identity)
    if leaked:
        stitched = _SAFE_LEAK
    mid = message_id or uuid.uuid4().hex
    # A stitched reply is only as grounded and as confident as its weakest contributing clause, so
    # report the min of each and the union of their citations, never an invented grounding=1.0. If
    # the leak tripwire replaced the reply, no clause backs it: report zero grounding, no citations.
    if leaked or not finals:
        grounding, confidence, citations = 0.0, 0.0, []
    else:
        grounding = min(f.get("grounding", 0.0) for f in finals)
        confidence = min(f.get("confidence", 0.0) for f in finals)
        citations = _merge_citations(finals)
    yield {"type": "token", "text": stitched}
    yield {"type": "final", "message_id": mid, "answer": stitched, "tier": "auto",
           "confidence": confidence, "grounding": grounding, "citations": citations,
           "lane": "multi", "budget": budget.snapshot()}


_ORDER_ID = re.compile(r"\b[A-Z]{2}\d{5,}\b")


def _find_order_ref(query: str, history) -> str:
    text = query + " " + " ".join(
        t.get("content", "") for t in (history or []) if t.get("role") == "user")
    m = _ORDER_ID.search(text)
    return m.group(0) if m else ""


_HANDOFF_MARK = "noting for our care specialist"  # the opening line of _escalate_handoff's brief


def _already_escalated(history) -> bool:
    """True when this session already filed an escalation handoff, so a repeated 'human please'
    (often the shopper just confirming the prior 'anything else?') should reassure, not open a
    second duplicate case."""
    for t in (history or []):
        if t.get("role") in ("assistant", "bot") and _HANDOFF_MARK in (t.get("content") or ""):
            return True
    return False


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
    # only claim it was logged and will be followed up when a case was actually filed; otherwise
    # take the details without a false promise
    if case_id:
        lines.append("I've logged this (ref {}) and a specialist will follow up by email. Is "
                     "there anything else you'd like me to add before I pass it over?".format(
                         case_id[:8]))
    else:
        lines.append("Let me take these details down for a specialist to pick up. Is there "
                     "anything else you'd like me to add before I pass it over?")
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
                review_queue=None, domain=None, budget=None, answer_cache=None):
    """Yield the same event dicts as stream_answer (token chunks then one final), after routing.

    Every model call on this turn goes through one shared TurnBudget: routing tie-break, generation,
    and each multi-task clause. The budget is checked before each call and the turn stops with the
    answers already in hand on breach, so it can never loop or run away on tokens or time."""
    budget = budget or TurnBudget()
    b_llm = BudgetedLLM(llm, budget)
    b_small = BudgetedLLM(small_llm, budget) if small_llm is not None else None
    signed_in = bool(auth_identity and auth_identity[0])
    decision = route(query, history=history, signed_in=signed_in, small_llm=b_small)

    # Lane continuity: a short follow-up ("when will it get here", "the cheaper one") often carries
    # no intent of its own and falls to the answers catch-all, breaking the thread it continues.
    # Re-route it with the prior shopper turns prepended so it inherits that lane (an order-status
    # follow-up stays in care, a styling one in stylist), instead of dropping to generic answers.
    if (decision.layer != 0 and decision.lane == "answers"
            and len((query or "").split()) <= 6 and history):
        prior = [t.get("content", "") for t in history
                 if t.get("role") == "user" and (t.get("content") or "").strip()]
        if prior:
            cont = route((" ".join(prior[-2:]) + " " + query).strip(), history=history,
                         signed_in=signed_in, small_llm=b_small)
            if cont.lane != "answers" and cont.clarify is None and not cont.tasks:
                decision = cont

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

    deps = dict(embedder=embedder, store=store, llm=b_llm, reranker=reranker,
                metric_resolver=metric_resolver, graph_retriever=graph_retriever, lang=lang,
                persona=persona, history=history, concise=concise, auth_identity=auth_identity,
                notes=notes)
    if trace_path is not None:
        deps["trace_path"] = trace_path

    # Already with the specialist after an earlier handoff: keep the specialist persona and answer
    # the routed lane's focus. Even a repeated "get me a human" here just reassures them and does
    # NOT file a second case, rather than restarting the shopper with the assistant.
    if persona == "agent":
        yield from _emit_with_budget(
            stream_answer(query, message_id=message_id,
                          role_fragment=role_fragment(decision.lane), lane=decision.lane, **deps),
            budget)
        return

    # A first request to reach a person: file exactly one ready case brief and confirm. This takes
    # the whole turn; a human request never fans out. Reached only when not already handed off.
    if decision.lane == "escalation":
        if _already_escalated(history):
            # a case was already filed this session, so a repeated ask (often just confirming the
            # prior "anything else?") reassures instead of opening a second, duplicate ticket
            text = ("You're all set, I've already passed your details to our specialist and "
                    "they'll follow up by email. Is there anything else I can help with meanwhile?")
            mid = message_id or uuid.uuid4().hex
            yield {"type": "token", "text": text}
            yield {"type": "final", "message_id": mid, "answer": text, "tier": "escalate",
                   "confidence": 1.0, "grounding": 1.0, "citations": []}
            return
        yield from _escalate_handoff(query, message_id=message_id, auth_identity=auth_identity,
                                     history=history, lang=lang, review_queue=review_queue,
                                     domain=domain, trace_path=trace_path)
        return

    # Multi-task: split (on conjunctions or commas) and fan out only when the clauses hit two or
    # more distinct actionable lanes, so "a red and blue jacket" stays one shopping turn. A turn
    # carrying an email is a verification turn: keep it whole so the name-plus-email gate still sees
    # both halves in one auth_text instead of losing one to a clause boundary.
    if not _EMAIL.search(query or ""):
        clauses = _split_clauses(query)
        if len(clauses) > 1:
            routed = [(c, route(c).lane) for c in clauses]
            # a human request anywhere in a multi-task turn escalates the WHOLE turn, so a real case
            # is filed instead of a plain escalation-lane answer with no handoff
            if any(ln == "escalation" for _, ln in routed):
                yield from _escalate_handoff(query, message_id=message_id,
                                             auth_identity=auth_identity, history=history,
                                             lang=lang, review_queue=review_queue, domain=domain,
                                             trace_path=trace_path)
                return
            actionable_lanes = {ln for _, ln in routed if ln in _ACTIONABLE}
            if len(actionable_lanes) >= 2:
                # fan out over ALL clauses, so an "answers" sub-question in the turn is still
                # answered and stitched in, not silently dropped by an actionable-only filter
                yield from _stream_multitask(routed, message_id=message_id,
                                             auth_identity=auth_identity, deps=deps, budget=budget)
                return

    # Single-task fast path. An anonymous answers-lane turn (generic FAQ, no order or profile data)
    # may be served from the semantic cache; every other lane bypasses it for safety.
    if answer_cache is not None and decision.lane == "answers" and not signed_in:
        yield from _answers_with_cache(query, lane=decision.lane, deps=deps, budget=budget,
                                       answer_cache=answer_cache, message_id=message_id)
        return
    yield from _emit_with_budget(
        stream_answer(query, message_id=message_id,
                      role_fragment=role_fragment(decision.lane), lane=decision.lane, **deps),
        budget)
