"""Typed state for the chat graph. total=False so nodes fill in their own slice as it flows."""
from __future__ import annotations

from typing import TypedDict


class ChatState(TypedDict, total=False):
    # inputs
    query: str                 # the raw user turn
    history: list              # prior turns [{"role": "user"|"assistant", "content": str}]
    message_id: str
    started: float             # perf_counter set at entry, for latency

    # understand
    rewritten_query: str       # standalone question (follow-ups resolved)
    route: str                 # factual | relational | qualitative | metric | out_of_domain

    # retrieve + evidence
    vector_contexts: list      # hits before any evidence block is injected
    contexts: list             # final numbered contexts (graph/metric blocks prepended)
    has_metric: bool
    has_graph: bool
    graph_auth: bool           # a query-named graph entity (authoritative)

    # gate + answer
    abstained: bool
    confidence: float
    answer: str
    tier: str                  # abstain | auto
    grounding: float
    citations: list
    trace: dict
