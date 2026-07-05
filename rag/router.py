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
# Human-support nouns for the escalation intercept. Kept narrow on purpose: "staff", "reps",
# "operator", "advisor", and "employee" are excluded because they collide with shopping vocabulary
# ("staff picks", "gym reps", "operator manual", "style advisor", "employee discount").
_HUMAN = r"(human|person|people|agent|representative|manager|supervisor)"

# A refusal to be handed to a person must NOT escalate ("I don't want to talk to a human").
_NO_HUMAN = re.compile(
    r"\b(don'?t|do not|no|not|never|without|rather not|no need)\b[^.?!]{0,20}\b" + _HUMAN + r"\b",
    re.I)

# An explicit request to reach a person. High precision on purpose: the frontend and the 8B
# tie-break catch the rest, so a false positive (which files a real case) is the worse error. The
# earlier "to a <human>" and "want/need a <human>" branches were removed: they hijacked gift
# recipients ("a scarf to give to a supervisor") and style requests ("I need an advisor for my
# outfit"). Escalation now needs a support verb landing on a human noun, or an unambiguous phrase.
_ESCALATE = re.compile(
    r"\b(speak|talk|connect|transfer\w*|chat)\b[ ,]*(to|with|me to|me with)[ ,]*"
    r"(a |an |the |your )?" + _HUMAN + r"\b"
    r"|\b(real|actual|live) (person|human|agent|representative|rep|employee|staff)\b"
    r"|\b(a |an )?" + _HUMAN + r" (please|now|right now|asap|pls)\b"
    r"|\bhuman (being|help|support|agent|one)\b"
    r"|\bcustomer (service|care|support) (rep|reps|representative|agent|team|person)\b"
    r"|\b(speak|talk|chat) (to|with) (someone|somebody|anybody|a real person)\b"
    r"|\bsomeone (real|from (your|the) team|on (this|the) (chat|line))\b"
    r"|\bescalate (this|it|me|my|to)\b"
    r"|\b((no more|stop with the|done with the) (bot|chatbot|robot)|get me off (this|the) "
    r"(bot|chat))\b"
    r"|\byour (supervisor|manager)\b"
    r"|^\s*(a )?" + _HUMAN + r"\s*[.!?]*\s*$", re.I)

# A looser second pass for "get me a human" / "I need a human" style asks. The human noun must sit
# at a clause boundary (end, punctuation, or a help/talk word), so "get me a human please" matches
# but "get me the manager cut blazer" does not (the noun is followed by a product word).
_ESCALATE_LOOSE = re.compile(
    r"\b(get|give) (me )?(a |an |the )?" + _HUMAN + r"\b\s*([.!?,]|please|now|asap|to help|"
    r"on (this|the)|$)"
    r"|\bi (want|need|wanna)\b[ ,]*(to (speak|talk) to )?(a |an )?" + _HUMAN
    + r"\b\s*([.!?,]|please|now|to help|to talk|not an ai|on (this|the)|about|$)"
    r"|\b(hand|refer|put|transfer|connect)\b[ ,]*(me |this |it )?(over |off |through )?"
    r"(to|with)[ ,]*(a |an |the |your )?" + _HUMAN + r"\b", re.I)

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

# Order ids are two capitals plus digits, matched case-sensitively so a lowercase product code
# ("ab1234") is not mistaken for an order. Checked separately in _intent_lanes because the rest of
# the care cue is case-insensitive.
_ORDER_ID = re.compile(r"\b[A-Z]{2}\d{4,}\b")
_CARE_CUE = re.compile(
    r"\bmy (order|orders|package|parcel|deliver\w*|shipment|tracking|return|returns|account|"
    r"subscription|refund|store credit|loyalty)\b"
    r"|\b(return|send back|sending back|take back) (my|this|it|these|them|the)\b"
    r"|\bwant (my|a) (refund|money back)\b|\b(refund|money back) (please|on my|for my)\b"
    r"|\bwhen (will|is|does) my (order|package|parcel|delivery|stuff|shipment)\b[^.?!]{0,20}"
    r"\b(arrive|arriving|ship|shipped|get here|here|coming|deliver\w*)\b"
    r"|\b(eta on my|tracking (number|info|for my)|loyalty points|store credit|order status)\b"
    r"|\bleft the warehouse\b|\b(did|has|have) my (return|order|package|parcel|refund)\b"
    r"|\bmy (last|recent|previous) (purchase|order)\b"
    r"|\b(what did i buy|what i bought|total on my)\b", re.I)

# A complaint. Garment-damage words ("torn", "stain", "ripped") are matched ONLY in a problem frame
# (arrived/came/is/has), so catalog styles like "ripped jeans" or "stain-resistant" do not read as
# complaints. (The shared problem_intent guard still fires on bare damage words for the linear
# path; that is its long-standing behavior and out of scope here.)
_COMPLAINT_CUE = re.compile(
    r"\b(arrived|came|showed up|shipped|is|was|has a|had a|got a|there'?s a|with a)\b[^.?!]{0,18}"
    r"\b(hole|torn|ripped|stain(ed)?|scuff\w*|peel\w*|unravel\w*|frayed|defect\w*|damaged|"
    r"broken|cracked|falling apart)\b"
    r"|\b(wrong (size|colou?r|item|order)|not what i ordered|someone else'?s (order|package))\b"
    r"|\b(charged|billed)\b[^.?!]{0,24}\b(twice|two times|again|double|duplicate|extra)\b"
    r"|\b(double|duplicate|extra) (charge|charges|payment|billing)\b"
    r"|\bnever (arrived|received|got|showed up|authorized|authorised|signed up)\b"
    r"|\b(says|marked) (delivered|shipped)\b[^.?!]{0,24}\b(but|nothing|missing|empty|not here)\b"
    r"|\btracking\b[^.?!]{0,18}\b(hasn'?t|not|stopped)\b[^.?!]{0,10}\b(moved|updated|update)\b"
    r"|\bstill (waiting|no (package|refund|update))\b"
    r"|\brefund\b[^.?!]{0,30}\b(still|hasn'?t|weeks|not (here|received))\b"
    r"|\bcancel\w*\b[^.?!]{0,24}\b(without|no (reason|explanation|warning))\b"
    r"|\b(this is (ridiculous|unacceptable)|fed up|so frustrated|ruined my)\b"
    r"|\bfell (off|apart)\b|\bwrong address\b", re.I)

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
    if account_intent(query) or _CARE_CUE.search(query) or _ORDER_ID.search(query):
        lanes.append("care")
    if shopping_intent(query) or _STYLIST_CUE.search(query):
        lanes.append("stylist")
    return lanes


def route(query: str, *, history: list | None = None, signed_in: bool = False,
          small_llm=None, tiebreak_system: str | None = None) -> RouteDecision:
    q = (query or "").strip()
    if not q:
        return RouteDecision("answers", 0.3, 0, "empty query")

    # Layer 0: an explicit request to reach a person outranks every other signal, unless it is a
    # refusal ("I don't want to talk to a human"), which must not escalate or file a case.
    if (_ESCALATE.search(q) or _ESCALATE_LOOSE.search(q)) and not _NO_HUMAN.search(q):
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
