"""The named capabilities the omni lanes compose.

Each is a thin adapter over an existing engine function, so there is still one implementation of
retrieval, the governed metric layer, and graph resolution (the specialists behind the Finding
contract). The only new thing here is get_profile, which turns the shopper's trusted on-device
notes into a short brief a lane can personalize from.

Order and account lookups are deliberately NOT a separate privileged tool. They flow through the
same retrieve() path as product and policy questions, so the deterministic order-PII gate inside
retrieve() decides whether order records are even in the candidate pool. A lane cannot reach around
it to a "fetch this customer's orders" call, because no such call exists.
"""
from __future__ import annotations

from rag.specialists import Finding, graph_finding, metrics_finding, retriever_finding


def get_governed_metric(query: str, *, llm, metric_resolver) -> Finding:
    """A governed aggregate (a return rate, an average price, a count). A resolved, non-null number
    is authoritative and is preferred over anything a model would estimate."""
    return metrics_finding(query, llm=llm, metric_resolver=metric_resolver)


def graph_facts(query: str, *, graph_retriever, extra_texts: tuple = ()) -> Finding:
    """Relational facts about a query-named entity, resolved from the knowledge graph. A
    query-named entity is authoritative; a hop from retrieved text is enrichment."""
    return graph_finding(query, graph_retriever=graph_retriever, extra_texts=extra_texts)


def retrieve_evidence(query: str, *, embedder, store, llm=None, reranker=None,
                      generate_answer: bool = False) -> Finding:
    """Hybrid retrieval over the catalog and knowledge base, self-scored. This is the shared path
    for product facts, policy, recommendations, and a shopper's own order docs. The order-PII gate
    inside retrieve() decides whether order records are in the pool at all, so this is safe to call
    from any lane. generate_answer=False returns evidence only (no model call)."""
    return retriever_finding(query, embedder=embedder, store=store, llm=llm, reranker=reranker,
                             generate_answer=generate_answer)


def get_profile(notes: str | None) -> dict:
    """Summarize the shopper's trusted on-device profile (recipients, interests) that the client
    sends as notes, into a short brief a lane can personalize from. Returns {} when there is
    nothing. The notes are client-provided and sanitized upstream; this only shapes them into a
    brief, it never treats them as instructions."""
    brief = (notes or "").strip()
    return {"brief": brief} if brief else {}


__all__ = ["Finding", "get_governed_metric", "graph_facts", "retrieve_evidence", "get_profile"]
