"""The lanes the omni orchestrator routes among, as data rows, not a class hierarchy.

Each lane is a short system-prompt fragment appended to the shared assistant prompt, plus the
persona it speaks as. The answer still comes from the one gated pipeline (retrieval, the order-PII
gate, grounding, safety), so a lane sharpens TONE and FOCUS, it never changes the safety rules. New
lanes are a new row here, not new code, which is the point: specialization without a fork.

The fragments are deliberately domain-neutral (no product nouns or brand words) so the engine stays
swappable and the leak linter passes; the pack's persona and catalog supply the specifics.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lane:
    name: str
    persona: str | None  # None speaks as the assistant; "agent" speaks as the escalation specialist
    fragment: str        # appended to the system prompt to focus this lane


_STYLIST = Lane(
    "stylist", None,
    "This turn is about choosing well: a gift, an outfit, or what goes together. Ask at most one "
    "or two focused questions (the occasion, who it is for, a budget or size) only when you truly "
    "need them, then recommend with a short reason each pick fits. Use what you already know from "
    "the profile and the conversation, and never re-ask something they already told you.")

_CARE = Lane(
    "care", None,
    "This turn is about their own order or account. Answer their order and account questions "
    "directly from the records in the context, greet a known shopper by first name, and never ask "
    "a verified shopper to re-share what the records already show. If they are not verified, ask "
    "only for the name and email on the order, nothing more.")

_COMPLAINT = Lane(
    "complaint", None,
    "This turn is a problem. Lead with a brief, genuine acknowledgement before anything else, take "
    "ownership, and move straight to the fix. Do not upsell and do not ask them to pick a category "
    "or budget. If it needs a person, say you will connect them and gather what a specialist will "
    "need to pick it up quickly.")

_ANSWERS = Lane(
    "answers", None,
    "Answer their question directly and only from the context. If the answer is not there, say so "
    "plainly and offer a sensible next step instead of guessing.")

# Escalation speaks as the specialist persona; its detailed contract (gather, confirm, hand back)
# lives in the escalation prompt itself, so the lane fragment stays empty here. That specialist runs
# on the SAME Groq LLM as the rest of the app (no separate frontier vendor or second API); a more
# capable tier, if ever wanted, is a bigger Groq model set by config, never a second-vendor API.
_ESCALATION = Lane("escalation", "agent", "")

LANES: dict[str, Lane] = {
    lane.name: lane for lane in (_STYLIST, _CARE, _COMPLAINT, _ANSWERS, _ESCALATION)
}


def role_fragment(lane_name: str) -> str:
    return LANES.get(lane_name, _ANSWERS).fragment


def lane_persona(lane_name: str) -> str | None:
    return LANES.get(lane_name, _ANSWERS).persona
