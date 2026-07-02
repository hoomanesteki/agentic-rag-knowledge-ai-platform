"""M6.2 specialist agents behind one common Finding contract.

Each specialist owns one evidence source and returns a self-scored Finding (answer, evidence
contexts, confidence, citations), so the supervisor (M6.3) can dispatch to the relevant ones and
reconcile what they return. They reuse the M1 to M5 functions, so there is still one
implementation of retrieval, the metric layer, and graph resolution.

- Retriever: hybrid text retrieval, generates and grounds an answer.
- Metrics: routes to the governed metric layer; a resolved number is authoritative.
- Graph: resolves a query-named entity to graph facts; a query-named entity is authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.answer import (
    _SYSTEM,
    DEFAULT_MIN_CONFIDENCE,
    _build_prompt,
    _metric_has_value,
    _used_citations,
    build_contexts,
    grounding_score,
    retrieve,
    should_abstain,
)
from retrieval.metric_router import metric_context, route_metric


@dataclass
class Finding:
    """One specialist's contribution. found is False when the specialist has nothing to say for
    this query (not its slice), so the supervisor can ignore it. authoritative marks governed or
    query-named evidence that may stand on its own. abstained means it found evidence but did not
    produce an answer (the gate blocked it, or no model was available).

    confidence and grounding are separate signals: confidence is retrieval/evidence strength;
    grounding is citation discipline of the answer (a cited-but-wrong answer can still ground).
    Contexts are joined across specialists by `id` (unique: metric:*, graph:*, and chunk ids),
    never by `n`, which is finding-local and must be renumbered when the supervisor merges them.
    """

    specialist: str            # retriever | metrics | graph
    kind: str                  # text | metric | graph
    found: bool = False
    authoritative: bool = False
    abstained: bool = False
    answer: str = ""
    confidence: float = 0.0
    grounding: float = 0.0
    contexts: list = field(default_factory=list)
    citations: list = field(default_factory=list)


def _cite(contexts: list) -> list:
    return [{"n": c["n"], "id": c["id"], "source": c.get("source"), "doc_type": c.get("doc_type")}
            for c in contexts]


def retriever_finding(query: str, *, embedder, store, llm, reranker=None, top_k: int = 8,
                      top_k_in: int = 50,
                      min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> Finding:
    """Hybrid text retrieval, self-scored by lexical overlap and answer grounding."""
    hits = retrieve(query, embedder, store, top_k, reranker=reranker, top_k_in=top_k_in)
    contexts = build_contexts(hits)
    if not contexts:
        return Finding("retriever", "text", found=False)
    abstained, confidence = should_abstain(query, contexts, min_confidence)
    confidence = round(confidence, 3)
    if abstained or llm is None:
        # found context but did not answer (gated, or no model); leave it to the supervisor
        return Finding("retriever", "text", found=True, abstained=True, confidence=confidence,
                       contexts=contexts)
    result = llm.generate(_build_prompt(query, contexts), system=_SYSTEM)
    cited = _used_citations(result.text, contexts)
    grounding = grounding_score(result.text, contexts)
    return Finding("retriever", "text", found=True, answer=result.text, confidence=confidence,
                   grounding=round(grounding, 3), contexts=contexts, citations=_cite(cited))


def metrics_finding(query: str, *, llm, metric_resolver) -> Finding:
    """Route to the governed metric layer. A resolved, non-null number is authoritative."""
    if metric_resolver is None or llm is None:  # routing needs the model to slot-fill
        return Finding("metrics", "metric", found=False)
    result = route_metric(query, llm, metric_resolver)
    if result is None or not _metric_has_value(result):
        return Finding("metrics", "metric", found=False)
    block = metric_context(result)
    block["n"] = 1
    return Finding("metrics", "metric", found=True, authoritative=True,
                   answer="Governed metric: " + result.summary(), confidence=1.0,
                   contexts=[block], citations=_cite([block]))


def graph_finding(query: str, *, graph_retriever, extra_texts: tuple = ()) -> Finding:
    """Resolve a query-named entity (or an entity in retrieved text) to graph facts. A
    query-named entity is authoritative relational grounding; a hop from text is enrichment."""
    if graph_retriever is None:
        return Finding("graph", "graph", found=False)
    block, from_query = graph_retriever.evidence(query, extra_texts)
    if block is None:
        return Finding("graph", "graph", found=False)
    block["n"] = 1
    return Finding("graph", "graph", found=True, authoritative=from_query,
                   answer=block["text"], confidence=1.0 if from_query else 0.5,
                   contexts=[block], citations=_cite([block]))
