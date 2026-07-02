"""M6.3 supervisor: dispatch to specialists, reconcile, synthesize one answer. Offline on the
fakes. Proves the supervisor fans out only when a query needs it, merges governed and text
evidence, resolves a planted conflict by evidence rank, and abstains when nothing answers.
"""
from adapters.base import LLMResult
from adapters.factory import make_embedder, make_graph, make_llm, make_store
from data.lakehouse import build_lakehouse
from data.metrics import MetricResolver
from knowledge.graph_loader import load_graph
from rag.supervisor import run_supervised
from retrieval.graph import make_graph_retriever
from retrieval.sparse import SparseEncoder


class SlotSynthLLM:
    """Slot-fills the metric, and for the synthesis prompt answers citing the first source."""

    model = "fake"

    def generate(self, prompt, *, system=None, max_tokens=512):
        text = ('{"metric": "return_rate_by_size", "params": {"size": "M"}}'
                if "Available metrics:" in prompt else "the return rate is 0.5 [1]")
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")


def _store_with(*texts):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    points = []
    for i, text in enumerate(texts):
        dense = embedder.embed([text])[0]
        sparse = encoder.encode(text)
        points.append({"id": "D{}".format(i), "text": text, "payload": {"doc_type": "review"},
                       "dense": dense, "sparse": {"indices": sparse.indices,
                                                  "values": sparse.values}})
    store.upsert(points)
    return embedder, store


def _resolver(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    return MetricResolver("apparel_ecommerce", db)


def _graph(tmp_path):
    db = str(tmp_path / "lh2.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    store = make_graph("memory")
    load_graph("apparel_ecommerce", db, store)
    return make_graph_retriever("apparel_ecommerce", store)


def test_supervisor_answers_plain_question_with_only_retriever(tmp_path):
    embedder, store = _store_with("the flow legging runs small so size up one")
    components = {"embedder": embedder, "store": store, "llm": make_llm("fake")}
    result = run_supervised("does the flow legging run small", components=components,
                            trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "auto"
    assert result.trace["specialists"] == ["retriever"]  # no fan-out when one source suffices


def test_supervisor_abstains_out_of_scope(tmp_path):
    embedder, store = _store_with("the flow legging runs small")
    components = {"embedder": embedder, "store": store, "llm": make_llm("fake")}
    result = run_supervised("what is the boiling point of water", components=components,
                            trace_path=str(tmp_path / "t.jsonl"))
    assert result.abstained


def test_supervisor_answers_relational_from_graph(tmp_path):
    components = {"embedder": make_embedder("fake"), "store": make_store("memory"),
                 "llm": make_llm("fake"), "graph_retriever": _graph(tmp_path)}
    result = run_supervised("which supplier makes the Aster Cloud Hoodie", components=components,
                            trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "auto"
    assert "graph" in result.trace["specialists"]
    assert "Northloom" in " ".join(c["text"] for c in result.contexts)


def test_supervisor_merges_metric_and_text(tmp_path):
    embedder, store = _store_with("customers say size M returns are common")
    components = {"embedder": embedder, "store": store, "llm": SlotSynthLLM(),
                 "metric_resolver": _resolver(tmp_path)}
    result = run_supervised("what is the return rate for size M", components=components,
                            trace_path=str(tmp_path / "t.jsonl"))
    assert result.trace["metric"] is True
    assert set(result.trace["specialists"]) >= {"metrics", "retriever"}


def test_planted_conflict_is_flagged_and_resolved_by_rank(tmp_path):
    # the governed metric says 0.5; a review claims 0.9. The supervisor must flag the conflict and
    # ground the answer in the governed number (ranked first), not the review.
    embedder, store = _store_with("the return rate for size M is 0.9 according to a shopper")
    components = {"embedder": embedder, "store": store, "llm": SlotSynthLLM(),
                 "metric_resolver": _resolver(tmp_path)}
    result = run_supervised("what is the return rate for size M", components=components,
                            trace_path=str(tmp_path / "t.jsonl"))
    assert result.trace["conflict"] is True
    assert result.trace["conflict_resolved"] is True
    assert result.citations[0]["id"].startswith("metric:")  # governed evidence cited first


def test_conflict_with_wrong_synthesis_falls_back_to_governed(tmp_path):
    # a model that grabs the review's 0.9 instead of the governed 0.5 must not ship as clean: the
    # supervisor replaces it with the governed evidence and marks the conflict unresolved.
    class WrongSynthLLM:
        model = "fake"

        def generate(self, prompt, *, system=None, max_tokens=512):
            text = ('{"metric": "return_rate_by_size", "params": {"size": "M"}}'
                    if "Available metrics:" in prompt else "the return rate is 0.9 [2]")
            return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")

    embedder, store = _store_with("the return rate for size M is 0.9 according to a shopper")
    components = {"embedder": embedder, "store": store, "llm": WrongSynthLLM(),
                 "metric_resolver": _resolver(tmp_path)}
    result = run_supervised("what is the return rate for size M", components=components,
                            trace_path=str(tmp_path / "t.jsonl"))
    assert result.trace["conflict"] is True
    assert result.trace["conflict_resolved"] is False
    assert "0.5" in result.answer and "0.9" not in result.answer  # governed value shipped
