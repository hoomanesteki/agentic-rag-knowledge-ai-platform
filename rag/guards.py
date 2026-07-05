"""The deterministic, promptless guard layer.

These are the intent classifiers and the routing heuristic that decide a turn's shape before any
model runs. Each is a pure function over the raw query (regex and token sets), so it costs nothing,
is trivially testable, and cannot be prompt-injected. This module is the single import surface the
omni brain (the router and the lane graph) depends on, so the new brain leans on a stable guard API
instead of reaching into pipeline internals.

The intent regexes grew up next to the linear answer path and still live in `pipeline.answer`, so
this module re-exports them rather than cloning them: one name for each guard, one implementation.
A later refactor can relocate the definitions here once the linear path is retired; until then this
keeps them DRY without a risky move of battle-tested code.

Order and account PII gating is deliberately NOT exposed here as a "safe to disclose" primitive.
The real disclosure gate stays in `pipeline.answer.retrieve()` and the name-plus-email check,
coupled to the retrieval pool it protects. `account_intent` only tells a router that a turn is
about the shopper's own account; it never authorizes disclosure on its own.
"""
from __future__ import annotations

from pipeline.answer import _account_intent as account_intent
from pipeline.answer import _problem_intent as problem_intent
from pipeline.answer import _shopping_intent as shopping_intent
from rag.understand import ROUTES, heuristic_route

__all__ = [
    "account_intent",   # first-person "my order/account" question (not a disclosure grant)
    "problem_intent",   # complaint / billing / delivery problem: empathy and handoff, not upsell
    "shopping_intent",  # a recommendation request: recommend the closest match, do not dead-end
    "heuristic_route",  # factual / relational / qualitative / metric, cheap and deterministic
    "ROUTES",
]
