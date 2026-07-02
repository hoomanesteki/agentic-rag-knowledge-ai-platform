"""M6.2 specialists: each owns one evidence source and returns a self-scored Finding, offline
on the fakes. Proves each answers its slice, reports confidence and authority correctly, and
stays quiet (found=False) outside its slice.
"""
from adapters.base import LLMResult
from adapters.factory import make_embedder, make_graph, make_llm, make_store
from data.lakehouse import build_lakehouse
from data.metrics import MetricResolver
from knowledge.graph_loader import load_graph
from rag.specialists import graph_finding, metrics_finding, retriever_finding
from retrieval.graph import make_graph_retriever
from retrieval.sparse import SparseEncoder


class SlotLLM:
    model = "fake"

    def generate(self, prompt, *, system=None, max_tokens=512):
        text = ('{"metric": "return_rate_by_size", "params": {"size": "M"}}'
                if "Available metrics:" in prompt else "the return rate is 0.5 [1]")
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")


def _seed_store():
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    text = "the flow legging runs small so size up one"
    dense = embedder.embed([text])[0]
    sparse = encoder.encode(text)
    store.upsert([{"id": "R2", "text": text, "payload": {"doc_type": "review"},
                   "dense": dense, "sparse": {"indices": sparse.indices, "values": sparse.values}}])
    return embedder, store


def _metric_resolver(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    return MetricResolver("apparel_ecommerce", db)


def _graph_retriever(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    graph = make_graph("memory")
    load_graph("apparel_ecommerce", db, graph)
    return make_graph_retriever("apparel_ecommerce", graph)


def test_retriever_answers_its_slice():
    embedder, store = _seed_store()
    finding = retriever_finding("does the flow legging run small", embedder=embedder, store=store,
                                llm=make_llm("fake"))
    assert finding.found and finding.answer and finding.confidence > 0
    assert finding.citations


def test_retriever_quiet_with_no_evidence():
    finding = retriever_finding("anything", embedder=make_embedder("fake"),
                                store=make_store("memory"), llm=make_llm("fake"))
    assert finding.found is False


def test_retriever_abstains_on_weak_overlap():
    embedder, store = _seed_store()  # only a legging review is indexed
    finding = retriever_finding("what is the boiling point of water", embedder=embedder,
                                store=store, llm=make_llm("fake"))
    assert finding.found and finding.abstained and finding.answer == ""


def test_retriever_without_llm_returns_evidence_not_answer():
    embedder, store = _seed_store()
    finding = retriever_finding("does the flow legging run small", embedder=embedder, store=store,
                                llm=None)
    assert finding.found and finding.abstained and finding.contexts and finding.answer == ""


def test_metrics_without_llm_is_quiet(tmp_path):
    finding = metrics_finding("what is the return rate for size M", llm=None,
                              metric_resolver=_metric_resolver(tmp_path))
    assert finding.found is False


def test_metrics_answers_its_slice(tmp_path):
    finding = metrics_finding("what is the return rate for size M", llm=SlotLLM(),
                              metric_resolver=_metric_resolver(tmp_path))
    assert finding.found and finding.authoritative and finding.confidence == 1.0
    assert "0.5" in finding.answer


def test_metrics_quiet_outside_its_slice(tmp_path):
    finding = metrics_finding("does the legging run small", llm=make_llm("fake"),
                              metric_resolver=_metric_resolver(tmp_path))
    assert finding.found is False  # the pre-gate never routes a non-metric question


def test_graph_answers_its_slice(tmp_path):
    finding = graph_finding("which supplier makes the Aster Cloud Hoodie",
                            graph_retriever=_graph_retriever(tmp_path))
    assert finding.found and finding.authoritative and finding.confidence == 1.0
    assert "Northloom" in finding.answer


def test_graph_hop_from_text_is_not_authoritative(tmp_path):
    finding = graph_finding("what is your refund policy",
                            graph_retriever=_graph_retriever(tmp_path),
                            extra_texts=("the Aster Cloud Hoodie keeps you warm",))
    assert finding.found and finding.authoritative is False and finding.confidence == 0.5


def test_graph_quiet_outside_its_slice(tmp_path):
    finding = graph_finding("what is the boiling point of water",
                            graph_retriever=_graph_retriever(tmp_path))
    assert finding.found is False
