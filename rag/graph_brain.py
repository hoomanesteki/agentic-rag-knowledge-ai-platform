"""LangGraph corrective / self-RAG brain: an OPT-IN lane (CHAT_BRAIN=graph) for the hard tail.

Determinism owns the default. The router -> omni -> one-gated-pipeline cascade handles every default
turn; it won the routing eval and gives one safety surface instead of two. This lane is deliberately
OFF that fast path: an opt-in corrective loop for genuinely hard, retrieval-heavy turns, shaped like
the classic corrective/self-RAG graph:

    retrieve -> grade documents -> (rewrite the query + re-retrieve, loop) -> generate
             -> grade the generation -> (loop) or abstain

The load-bearing safety property: every user-facing answer is produced by pipeline.answer
.stream_answer (buffered per attempt), so the deterministic gates (injection / enumeration / harm
intercepts, order-PII, gender redaction, abstain, per-citation grounding, citations, trace) run
BELOW the graph exactly once. The graph only chooses the retrieval and query STRATEGY; it never
re-implements a gate and never opens a second, weaker safety surface. The loop is bounded three
ways: a MAX_RETRIES counter in the graph state, the shared TurnBudget (checked before every model
call inside the budgeted LLM), and a recursion_limit backstop that should never be the thing that
fires. Because the loop must grade before committing, the graph runs buffered and emits the accepted
attempt's text plus its final event, preserving the SSE contract without double-streaming.

LangGraph is an optional dependency; this module is imported only when CHAT_BRAIN=graph is selected.
"""
from __future__ import annotations

import operator
import uuid
from typing import Annotated, TypedDict

try:
    from langgraph.errors import GraphRecursionError
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - only when the graph lane runs without the dep
    raise ImportError(
        "The graph brain (CHAT_BRAIN=graph) needs LangGraph. Install it: uv sync --extra graph"
    ) from exc

from adapters.budget import BudgetedLLM, BudgetExceeded, TurnBudget
from pipeline.answer import (
    DEFAULT_MIN_CONFIDENCE,
    _user_authored_text,
    build_contexts,
    retrieve,
    should_abstain,
    stream_answer,
)
from rag.roles import role_fragment
from rag.router import route
from rag.understand import rewrite_followup

MAX_RETRIES = 2  # corrective rewrites before we settle; the primary loop bound
_RECURSION_LIMIT = 12  # LangGraph superstep backstop; MAX_RETRIES + TurnBudget must fire first
_GROUNDING_FLOOR = 0.5  # below this a cited answer is weakly supported -> correct or abstain


class _State(TypedDict, total=False):
    question: str            # the current (possibly rewritten) retrieval + generation query
    original_question: str   # the shopper's words, preserved so a rewrite never drifts intent
    documents: list          # build_contexts()-shaped chunks from the latest retrieval
    generation: str          # buffered answer text from the latest stream_answer attempt
    final_event: dict        # {type:final, answer, tier, grounding, citations, ...} of that attempt
    retries: Annotated[int, operator.add]  # corrective-loop counter (reducer sums increments)
    verdict: str             # last grade label, for the trace
    auth_text: str           # shopper-authored identity, threaded so the order-PII gate holds


class _Ctx:
    """Per-turn dependencies the nodes close over (never in graph state: not serializable, and the
    gates must run on live adapters). Mirrors the deps rag.omni.stream_omni builds."""

    def __init__(self, *, embedder, store, reranker, lang, history, deps, lane, message_id, budget):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.lang = lang
        self.history = history
        self.deps = deps  # the kwargs stream_answer takes, with a BudgetedLLM as llm
        self.lane = lane
        self.message_id = message_id
        self.budget = budget


def _buffered_answer(question: str, ctx: _Ctx) -> tuple[str, dict]:
    """Answer one question through the ONE gated pipeline, buffered (mirrors rag.omni._answer_once).
    This is the single surface where every deterministic gate fires. Returns (text, final_event)."""
    parts: list[str] = []
    final: dict = {}
    for event in stream_answer(question, message_id=ctx.message_id,
                               role_fragment=role_fragment(ctx.lane), lane=ctx.lane, **ctx.deps):
        if event.get("type") == "token":
            parts.append(event.get("text", ""))
        elif event.get("type") == "final":
            final = event
    return ("".join(parts) or final.get("answer", "")), final


def make_graph(ctx: _Ctx):
    """Compile the corrective-RAG StateGraph with the turn's dependencies bound into the nodes."""

    def retrieve_node(state: _State) -> dict:
        # A grading retrieval, gated exactly like the pipeline: auth_text carries the shopper's own
        # words so an order document surfaces only on a valid name+email, never on a probe.
        hits = retrieve(state["question"], ctx.embedder, ctx.store, top_k=8, reranker=ctx.reranker,
                        top_k_in=50, auth_text=state.get("auth_text", ""), lang=ctx.lang)
        return {"documents": build_contexts(hits)}

    def grade_documents(state: _State) -> dict:
        # Model-free relevance grade, reusing the pipeline's own confidence gate so the two never
        # drift. "weak" means the retrieved context does not cover the question well enough.
        abstained, _conf = should_abstain(state["question"], state.get("documents") or [],
                                          DEFAULT_MIN_CONFIDENCE)
        return {"verdict": "weak" if abstained else "relevant"}

    def transform_query(state: _State) -> dict:
        # Rewrite to a better standalone/retrieval query, then increment the loop counter. The
        # rewrite is trusted only if it shares a content word with the conversation
        # (rewrite_followup guards this), else it falls back to the original query.
        rewritten = rewrite_followup(state["question"], ctx.history, ctx.deps["llm"])
        return {"question": rewritten, "retries": 1}

    def generate_node(state: _State) -> dict:
        text, final = _buffered_answer(state["question"], ctx)
        return {"generation": text, "final_event": final}

    def abstain_node(state: _State) -> dict:
        # Terminal: emit the best final already in hand. Because generation went through the
        # pipeline, a low-confidence turn is already the pipeline's honest abstain final.
        return {}

    def decide_after_documents(state: _State) -> str:
        if state.get("verdict") == "weak" and state.get("retries", 0) < MAX_RETRIES:
            return "transform_query"
        return "generate"

    def decide_after_generation(state: _State) -> str:
        final = state.get("final_event") or {}
        tier = final.get("tier")
        grounding = final.get("grounding") or 0.0
        if tier in ("degraded", "error"):
            return "abstain"  # a provider failure; a rewrite will not help
        if tier == "auto" and grounding >= _GROUNDING_FLOOR:
            return "useful"
        if state.get("retries", 0) < MAX_RETRIES:
            return "transform_query"
        return "abstain"

    workflow = StateGraph(_State)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate_node)
    workflow.add_node("abstain", abstain_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_after_documents,
                                   {"transform_query": "transform_query", "generate": "generate"})
    workflow.add_edge("transform_query", "retrieve")  # the corrective loop
    workflow.add_conditional_edges("generate", decide_after_generation,
                                   {"useful": END, "transform_query": "transform_query",
                                    "abstain": "abstain"})
    workflow.add_edge("abstain", END)
    return workflow.compile()


def _budget_final(budget: TurnBudget, reason: str):
    """A safe abstain final for a budget breach or the recursion backstop, shaped exactly like
    rag.omni._emit_with_budget's fallback so the SSE contract never dies mid-stream."""
    text = "Let me keep this quick and accurate and bring in a specialist to help you further."
    yield {"type": "token", "text": text}
    yield {"type": "final", "message_id": uuid.uuid4().hex, "answer": text, "tier": "abstain",
           "confidence": 0.0, "grounding": 0.0, "citations": [], "lane": "budget",
           "budget": {**budget.snapshot(), "stopped": reason}}


def stream_graph(query, *, embedder, store, llm, reranker=None, metric_resolver=None,
                 graph_retriever=None, lang=None, persona=None, history=None, concise=False,
                 auth_identity=None, notes=None, message_id=None, small_llm=None, trace_path=None,
                 review_queue=None, domain=None, budget=None, answer_cache=None,
                 block_order_pii=False):
    """Run the corrective-RAG lane and yield the same event dicts as stream_omni/stream_answer
    (token chunks then one final). Signature mirrors rag.omni.stream_omni so the API dispatch is a
    drop-in third branch. The turn is bounded by one shared TurnBudget."""
    budget = budget or TurnBudget()
    b_llm = BudgetedLLM(llm, budget)
    signed_in = bool(auth_identity and auth_identity[0])
    lane = route(query, history=history, signed_in=signed_in).lane
    mid = message_id or uuid.uuid4().hex
    # Identity for the order-PII gate is the shopper's own words, as stream_answer builds it,
    # or empty when order/account disclosure is blocked (e.g. an anonymous channel).
    identity = " ".join(p for p in (auth_identity or ()) if p)
    auth_text = "" if block_order_pii else (identity + " "
                                            + _user_authored_text(query, history)).strip()
    deps = dict(embedder=embedder, store=store, llm=b_llm, reranker=reranker,
                metric_resolver=metric_resolver, graph_retriever=graph_retriever, lang=lang,
                persona=persona, history=history, concise=concise, auth_identity=auth_identity,
                notes=notes, block_order_pii=block_order_pii)
    if trace_path is not None:
        deps["trace_path"] = trace_path
    ctx = _Ctx(embedder=embedder, store=store, reranker=reranker, lang=lang, history=history,
               deps=deps, lane=lane, message_id=mid, budget=budget)

    app = make_graph(ctx)
    initial: _State = {"question": query, "original_question": query, "retries": 0,
                       "auth_text": auth_text}
    try:
        state = app.invoke(initial, {"recursion_limit": _RECURSION_LIMIT})
    except BudgetExceeded as exc:
        yield from _budget_final(budget, exc.reason)
        return
    except GraphRecursionError:
        # Unreachable: MAX_RETRIES and the budget fire first; a loud safe-final beats a hang.
        yield from _budget_final(budget, "recursion")
        return

    final = state.get("final_event") or {}
    text = state.get("generation") or final.get("answer", "")
    if not final:
        yield from _budget_final(budget, "no-final")
        return
    final = {**final, "message_id": final.get("message_id") or mid, "budget": budget.snapshot()}
    yield {"type": "token", "text": text}
    yield final
