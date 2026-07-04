"""The chat graph (M6.1): the linear pipeline folded into a typed LangGraph state machine.

Nodes reuse the M1 to M5 functions as their bodies, so there is one implementation of retrieval,
evidence, the gate, and generation, orchestrated as a graph. This is the skeleton the supervisor
and specialist agents grow into at M6.2 and M6.3.

    understand -> retrieve -> evidence -> (gate) -> generate | abstain -> end

Stack: the graph is defined with LangGraph and compiles to a langchain-core `Runnable`
(`CompiledStateGraph`), so it is driven through the standard `.invoke()` interface; Langfuse wraps
each run in one root span (see run_chat). LangChain-core is the runtime substrate, LangGraph is the
orchestration, and Langfuse is the observability: one turn maps to one trace.
"""
from __future__ import annotations

import hashlib
import time
import uuid

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from adapters.observability import request_span, update_span
from pipeline.answer import (
    _ABSTAIN,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_TRACE_PATH,
    AnswerResult,
    _build_prompt,
    _estimate_cost,
    _system,
    _used_citations,
    build_contexts,
    grounding_score,
    retrieve,
    should_abstain,
    with_graph_evidence,
    with_metric_evidence,
    write_trace,
)
from rag.state import ChatState
from rag.understand import heuristic_route, rewrite_followup


def build_chat_graph(components: dict, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                     top_k: int = 8, top_k_in: int = 50,
                     trace_path: str = DEFAULT_TRACE_PATH) -> Runnable:
    """Compile the chat graph over a set of components (embedder, store, llm, and optionally
    reranker, metric_resolver, graph_retriever). Build once, invoke per turn. Returns a
    langchain-core Runnable (LangGraph's CompiledStateGraph), driven via .invoke()."""
    embedder = components["embedder"]
    store = components["store"]
    llm = components["llm"]
    reranker = components.get("reranker")
    metric_resolver = components.get("metric_resolver")
    graph_retriever = components.get("graph_retriever")

    def understand(state: ChatState) -> dict:
        started = time.perf_counter()  # at entry, so the rewrite LLM call is inside latency
        rewritten = rewrite_followup(state["query"], state.get("history") or [], llm)
        return {"rewritten_query": rewritten, "route": heuristic_route(rewritten),
                "started": started}

    def retrieve_node(state: ChatState) -> dict:
        query = state["rewritten_query"]
        hits = retrieve(query, embedder, store, top_k, reranker=reranker, top_k_in=top_k_in)
        vector_contexts = build_contexts(hits)
        # gate on vector evidence alone, before any authoritative block is injected
        abstained, confidence = should_abstain(query, vector_contexts, min_confidence)
        return {"vector_contexts": vector_contexts, "abstained": abstained,
                "confidence": confidence}

    def evidence_node(state: ChatState) -> dict:
        query = state["rewritten_query"]
        contexts, has_graph, graph_auth = with_graph_evidence(
            query, [dict(c) for c in state["vector_contexts"]], graph_retriever)
        contexts, has_metric = with_metric_evidence(query, contexts, llm, metric_resolver)
        abstained = state["abstained"] and not (has_metric or graph_auth)
        return {"contexts": contexts, "has_graph": has_graph, "graph_auth": graph_auth,
                "has_metric": has_metric, "abstained": abstained}

    def gate(state: ChatState) -> str:
        return "abstain" if state["abstained"] else "generate"

    def _base_trace(state: ChatState) -> dict:
        return {
            "ts": time.time(),
            "message_id": state.get("message_id"),
            "raw_query": state["query"],
            "query": state["rewritten_query"],
            "route": state["route"],
            "reranked": reranker is not None,
            "metric": state.get("has_metric", False),
            "graph": state.get("has_graph", False),
            "retrieved": [{"id": c["id"], "score": c["score"]} for c in state["contexts"]],
            "confidence": round(state.get("confidence", 0.0), 3),
        }

    def abstain_node(state: ChatState) -> dict:
        trace = _base_trace(state)
        trace.update(tier="abstain", model=None, grounding=0.0, prompt_tokens=0,
                     completion_tokens=0, cost=0.0,
                     latency_ms=round((time.perf_counter() - state["started"]) * 1000, 1))
        write_trace(trace, trace_path)
        return {"answer": _ABSTAIN, "tier": "abstain", "grounding": 0.0, "citations": [],
                "trace": trace}

    def generate_node(state: ChatState) -> dict:
        contexts = state["contexts"]
        prompt = _build_prompt(state["rewritten_query"], contexts)
        from adapters.config import get_settings
        result = llm.generate(prompt, system=_system(get_settings().domain))
        grounding = grounding_score(result.text, contexts)
        citations = [{"n": c["n"], "id": c["id"], "source": c["source"], "doc_type": c["doc_type"]}
                     for c in _used_citations(result.text, contexts)]
        trace = _base_trace(state)
        trace.update(
            tier="auto", model=result.model, grounding=round(grounding, 3),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
            cost=_estimate_cost(result.model, result.prompt_tokens, result.completion_tokens),
            latency_ms=round((time.perf_counter() - state["started"]) * 1000, 1))
        write_trace(trace, trace_path)
        return {"answer": result.text, "tier": "auto", "grounding": grounding,
                "citations": citations, "trace": trace}

    graph = StateGraph(ChatState)
    graph.add_node("understand", understand)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("abstain", abstain_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "evidence")
    graph.add_conditional_edges("evidence", gate, {"abstain": "abstain", "generate": "generate"})
    graph.add_edge("abstain", END)
    graph.add_edge("generate", END)
    return graph.compile()


def run_chat(query: str, *, components: dict, history: list | None = None,
             message_id: str | None = None, graph=None,
             min_confidence: float = DEFAULT_MIN_CONFIDENCE,
             trace_path: str = DEFAULT_TRACE_PATH) -> AnswerResult:
    """Run one turn through the chat graph and return the same AnswerResult the linear pipeline
    returns, so callers and the eval harness are unchanged. Pass a prebuilt graph to avoid
    recompiling per turn."""
    graph = graph or build_chat_graph(components, min_confidence=min_confidence,
                                       trace_path=trace_path)
    mid = message_id or uuid.uuid4().hex
    with request_span("chat.linear", input=query, metadata={"message_id": mid}):
        state = graph.invoke({"query": query, "history": history or [], "message_id": mid})
        update_span(output=state.get("answer"), metadata={"tier": state.get("tier")})
    return AnswerResult(
        answer=state["answer"], tier=state["tier"], confidence=state.get("confidence", 0.0),
        grounding=state.get("grounding", 0.0), citations=state.get("citations", []),
        contexts=state.get("contexts", []), trace=state.get("trace", {}))
