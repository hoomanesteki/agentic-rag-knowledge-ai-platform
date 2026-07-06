"""The master orchestrator's routing decision.

Every turn is routed by a cheap-first cascade, so the common case costs nothing and only genuine
ambiguity pays for a model call:

  Layer 0  deterministic intercept: an explicit request to reach a person. $0.
  Layer 1  the deterministic intent guards (complaint / account / shopping). $0, the majority.
  Layer 2  a small-model tie-break, only when the layers above do not decide. One cheap 8B call.

`route()` returns a RouteDecision naming the lane, a confidence, and which layer decided. When two
intent guards fire it returns a multi-task plan (complaint first) for the heavy path to fan out.
When even the small model cannot separate two lanes it returns a strict two-option clarify, so the
brain asks instead of guessing.

Safety and smalltalk are handled by the guards that run BEFORE routing (the input guard in the
graph, the smalltalk intercept in the fast path), so `route()` only chooses among service lanes and
assumes the query is already a standalone, non-malicious turn. Rewrite a follow-up before calling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.guards import account_intent, problem_intent, shopping_intent

# The service lanes the orchestrator routes among. Smalltalk and refusals never reach here.
LANES = ("stylist", "care", "complaint", "answers", "escalation")

# An explicit ask to reach a person. It needs an escalation verb near a human noun, an
# unambiguous phrase, or a bare human-noun message, so "a jacket for a tall person" or "in-person
# pickup" do not trip it while "human please" and a lone "representative" do.
# Human-support nouns. "someone" and "team" are deliberately excluded from the loose verb-to-noun
# match because they are core stylist vocabulary ("a gift for someone", "the team jacket"); they
# escalate only in the explicit phrases below.
_HUMAN = (r"(human|person|agent|representative|rep|reps|manager|supervisor|advisor|operator|"
          r"employee|staff)")
_ESCALATE = re.compile(
    # a verb landing directly on a human noun with only connective words between, so "get me an
    # agent" matches while "get me a gift for a travel agent" does not (product content breaks it)
    r"\b(speak|talk|chat|connect|transfer\w*|escalate|refer|put me|hand( me)? off|get( me)?|"
    r"give me)\b[ ,]*(to|with|me to|me with)?[ ,]*(a |an |the |your )?" + _HUMAN + r"\b"
    r"|\b(i (?:want|need)|need)\b[ ,]*(a |an |to (?:speak|talk) to (?:a |an )?)?" + _HUMAN + r"\b"
    r"|\b(real|actual|live)\s+" + _HUMAN + r"\b"
    r"|\bto (a |an )(real |actual |live )?" + _HUMAN + r"\b"
    r"|\b" + _HUMAN + r"[ ,]*(please|now|asap|pls)\b"
    r"|\bhuman (being|help|support)\b"
    r"|\bcustomer (service|care|support) (rep|agent|team|person)\b"
    r"|\bsomeone (real|from your team|on (?:this|the) (?:chat|line))\b"
    r"|\bto (?:speak|talk|chat) (?:to|with) someone\b"
    r"|\b(no more|stop with the|done with the) (bot|chatbot|robot)\b"
    r"|\byour (supervisor|manager)\b"
    r"|^\s*" + _HUMAN + r"\s*[.!?]*\s*$", re.I)

# Router-only lane cues that supplement the shared intent guards. The linear brain's guards are
# tuned narrowly for its recommend-versus-abstain logic; routing wants wider coverage so common
# phrasings land deterministically instead of paying for the small-model tie-break. Kept here, not
# in rag.guards, so the linear path is untouched. Domain-neutral (no product nouns or brand words).
_STYLIST_CUE = re.compile(
    r"\b(style|styling|outfit|wardrobe|capsule)\b"
    r"|\b(go|goes|pair|pairs|match)\b\s+(with|together)\b"
    r"|\bwhat\b[^.?!]{0,30}\bgo(es)?\s+with\b"
    r"|\bwhat (shoes|pants|top|tops|shirt|shirts|colou?rs?|accessor\w+|to wear)\b"
    r"|\bhow (do i|to|would you|should i)\b[^.?!]{0,24}\b(style|wear|dress|pair)\b"
    r"|\b(any ideas|ideas for|smart casual|black tie|dressed up|trending|which of your|"
    r"petite)\b", re.I)

_CARE_CUE = re.compile(
    r"\b[A-Z]{2}\d{4,}\b"  # an order id like OD100219
    r"|\bmy (order|orders|package|parcel|deliver\w*|shipment|tracking|return|returns|account|"
    r"subscription|purchase|purchases|stuff|refund|store credit|loyalty)\b"
    r"|\bwhen(?:'?s| is| will| does)?\b[^.?!]{0,30}\b(arrive|arriving|ship|shipped|shipping|"
    r"deliver\w*|get(ting)? here|come|here)\b"
    r"|\b(eta|tracking|loyalty points|store credit|order status)\b"
    r"|\bleft the warehouse\b|\b(did|has|have) my (return|order|package|parcel|refund)\b"
    r"|\bmy (last|recent|previous) (purchase|order)\b"
    r"|\b(what did i buy|what i bought|total on my|total of my)\b"
    r"|\bbought (in|last|from you|the)\b", re.I)

_COMPLAINT_CUE = re.compile(
    r"\b(hole|holes|torn|tear|ripped|rip|stain|stained|scuff\w*|peel\w*|unravel\w*|frayed|"
    r"defect\w*|damaged|broken|cracked)\b"
    r"|\b(wrong (size|colou?r|item|order|thing)|not what i ordered|someone else'?s|mislabel\w*)\b"
    r"|\b(charged|billed|charge|bill)\b[^.?!]{0,30}\b(twice|two times|again|double|duplicate|"
    r"extra)\b"
    r"|\b(two|double|extra|duplicate|second) (charge|charges|payment|billing)\b"
    r"|\bnever (authorized|authorised|signed up|ordered)\b"
    r"|\b(says|marked) (delivered|shipped)\b[^.?!]{0,30}\b(but|nothing|missing|empty|not here)\b"
    r"|\btracking\b[^.?!]{0,20}\b(hasn'?t|not|stopped)\b[^.?!]{0,12}\b(moved|updated|update)\b"
    r"|\b(still (no|waiting|processing|nothing)|no (package|jacket|update|sign|refund))\b"
    r"|\brefund\b[^.?!]{0,40}\b(nothing|still|hasn'?t|weeks|not (here|received))\b"
    r"|\bcancel\w*\b[^.?!]{0,30}\b(without|no (reason|explanation|warning))\b"
    r"|\b(ridiculous|unacceptable|frustrat\w*|ruined|useless|fed up|not ok(ay)?|cheap quality|"
    r"worst)\b"
    r"|\bfell (off|apart)\b|\bwrong address\b|\bcourier left it\b", re.I)

# Confidence per single-intent lane. Complaint is most certain (its cues are specific), a shopping
# request least (its cues are broad), so a low-confidence stylist route is re-checked first.
_INTENT_CONF = {"complaint": 0.85, "care": 0.8, "stylist": 0.75}


@dataclass
class RouteDecision:
    lane: str
    confidence: float
    layer: int  # 0 deterministic intercept, 1 intent guard, 2 small-model tie-break
    reason: str
    tasks: list[str] = field(default_factory=list)  # more than one when the turn needs a plan
    clarify: dict | None = None  # {"axis","a","b"} when asking beats guessing


def _intent_lanes(query: str) -> list[str]:
    """The service lanes whose deterministic intent fires, in resolution priority: a complaint
    leads (empathy before anything else), then an own-account lookup, then a shopping request."""
    lanes = []
    if problem_intent(query) or _COMPLAINT_CUE.search(query):
        lanes.append("complaint")
    if account_intent(query) or _CARE_CUE.search(query):
        lanes.append("care")
    if shopping_intent(query) or _STYLIST_CUE.search(query):
        lanes.append("stylist")
    return lanes


def route(query: str, *, history: list | None = None, signed_in: bool = False,
          small_llm=None, tiebreak_system: str | None = None) -> RouteDecision:
    q = (query or "").strip()
    if not q:
        return RouteDecision("answers", 0.3, 0, "empty query")

    # Layer 0: an explicit request to reach a person outranks every other signal.
    if _ESCALATE.search(q):
        return RouteDecision("escalation", 0.95, 0, "explicit request for a person")

    # Layer 1: the deterministic intent guards. One match decides. Two competing matches become a
    # multi-task plan (complaint first) that the heavy path fans out and stitches back together.
    lanes = _intent_lanes(q)
    if len(lanes) == 1:
        return RouteDecision(lanes[0], _INTENT_CONF[lanes[0]], 1, "intent guard: " + lanes[0])
    if len(lanes) > 1:
        return RouteDecision(lanes[0], 0.7, 1, "multiple intents, complaint-first plan",
                             tasks=lanes)

    # Layer 2: nothing fired. A cheap model tie-breaks into a lane or asks; with no model available
    # (offline, tests) fall back to the answers lane, which self-gates on facts, policy, and
    # catch-all and never fabricates.
    if small_llm is not None:
        decided = _model_tiebreak(q, small_llm, system=tiebreak_system)
        if decided is not None:
            return decided
    return RouteDecision("answers", 0.5, 1, "no intent fired, answers catch-all")


# Promoted from the prompt-optimization loop (docs/prompt-optimization.md): a conservative tie-break
# that prefers answers on general questions and reserves unclear for genuinely two-intent turns,
# rather than guessing a specialist lane. Human-reviewed and promoted from the candidate at
# mlops/prompt_registry/tiebreak_system.candidate.json (held-out routing test 73.9% -> 79.5%).
_TIEBREAK_SYSTEM = (
    "You are a strict router for a shopping assistant. Read the shopper's message and reply with "
    "ONLY a JSON object {\"lane\": L} where L is one of: stylist (specific product combination or "
    "gift ideas), care (direct reference to their own order or account), complaint (explicit "
    "problem, delay, or billing issue), answers (general or policy question, or seeking "
    "information), unclear (genuinely ambiguous between two specific intents). Prioritize answers "
    "for general inquiries and defer to unclear only when a message clearly conveys multiple "
    "distinct intents. No prose, only the JSON."
)


def _model_tiebreak(query: str, small_llm, system: str | None = None) -> RouteDecision | None:
    """One cheap classification call for the minority of turns the deterministic layers cannot
    place. On 'unclear' it returns a two-option clarify rather than a guess. Any parse or model
    failure returns None so the caller falls back to the answers catch-all. `system` overrides the
    tie-break prompt so a candidate from the prompt-optimization loop can be scored without an
    edit."""
    try:
        raw = small_llm.generate(query, system=system or _TIEBREAK_SYSTEM, max_tokens=40).text
    except Exception:
        return None
    match = re.search(r'"lane"\s*:\s*"([a-z]+)"', raw or "")
    lane = match.group(1) if match else ""
    if lane in ("stylist", "care", "complaint", "answers"):
        return RouteDecision(lane, 0.65, 2, "small-model tie-break")
    if lane == "unclear":
        # the most common genuine ambiguity: a gift idea versus a problem with an order. Ask the
        # one question that splits them instead of guessing and risking the wrong tone.
        return RouteDecision("answers", 0.4, 2, "ambiguous, clarify",
                             clarify={"axis": "intent",
                                      "a": "a product or gift suggestion",
                                      "b": "help with an existing order"})
    return None
