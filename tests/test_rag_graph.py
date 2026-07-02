"""M6.1 chat graph: the LangGraph state machine, offline on the fakes. Proves the four query
types flow through the graph, a follow-up is rewritten to a standalone question, the graph
matches the linear pipeline's verdict, and the route heuristic is measured on the golden set.
"""
import json
from pathlib import Path

from adapters.base import LLMResult
from adapters.factory import make_embedder, make_graph, make_llm, make_store
from data.lakehouse import build_lakehouse
from data.metrics import MetricResolver
from knowledge.graph_loader import load_graph
from pipeline.answer import answer_question
from rag.graph import run_chat
from rag.understand import heuristic_route, rewrite_followup
from retrieval.graph import make_graph_retriever
from retrieval.sparse import SparseEncoder

ROOT = Path(__file__).resolve().parents[1]


def _seed_store():
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    docs = [
        {"id": "R1", "text": "the daytrip belt bag costs 38 dollars and fits a phone",
         "payload": {"doc_type": "review"}},
        {"id": "R2", "text": "the flow legging runs small so size up one",
         "payload": {"doc_type": "review"}},
    ]
    dense = embedder.embed([d["text"] for d in docs])
    store.upsert([{**d, "dense": dv, "sparse": {"indices": sv.indices, "values": sv.values}}
                  for d, dv, sv in zip(docs, dense, [encoder.encode(d["text"]) for d in docs])])
    return {"embedder": embedder, "store": store, "llm": make_llm("fake")}


def test_graph_answers_relevant_question(tmp_path):
    result = run_chat("how much does the daytrip belt bag cost", components=_seed_store(),
                      trace_path=str(tmp_path / "t.jsonl"))
    assert result.tier == "auto"
    assert result.citations
    assert result.trace["route"] in ("factual", "relational", "qualitative", "metric")


def test_graph_abstains_out_of_scope(tmp_path):
    result = run_chat("what is the boiling point of water", components=_seed_store(),
                      trace_path=str(tmp_path / "t.jsonl"))
    assert result.abstained
    assert result.citations == []


def test_graph_matches_linear_pipeline(tmp_path):
    components = _seed_store()
    query = "how much does the daytrip belt bag cost"
    graph_result = run_chat(query, components=components, trace_path=str(tmp_path / "g.jsonl"))
    linear = answer_question(query, embedder=components["embedder"], store=components["store"],
                             llm=components["llm"], trace_path=str(tmp_path / "l.jsonl"))
    assert graph_result.tier == linear.tier  # same functions, same verdict


def test_followup_is_rewritten_to_standalone():
    class RewritingLLM:
        def generate(self, prompt, *, system=None, max_tokens=512):
            if system and "standalone" in system.lower():
                return LLMResult(text="does the flow legging come in size small",
                                 prompt_tokens=1, completion_tokens=1, model="fake")
            return LLMResult(text="answer [1]", prompt_tokens=1, completion_tokens=1, model="fake")

    components = _seed_store()
    components["llm"] = RewritingLLM()
    history = [{"role": "user", "content": "tell me about the flow legging"},
               {"role": "assistant", "content": "it runs small"}]
    result = run_chat("what about small?", components=components, history=history)
    assert result.trace["query"] == "does the flow legging come in size small"


def test_offline_followup_falls_back_when_rewrite_is_unrelated():
    # the echo fake returns an unrelated string, so understand keeps the original question
    history = [{"role": "user", "content": "tell me about the flow legging"}]
    assert rewrite_followup("what about small", history, make_llm("fake")) == "what about small"


def test_rewrite_guard_rejects_stopword_only_overlap():
    # a rewrite sharing only stopwords with the conversation is not trusted (the guard uses
    # content words, not bare tokens)
    class StopwordLLM:
        def generate(self, prompt, *, system=None, max_tokens=512):
            return LLMResult(text="what is the that", prompt_tokens=1, completion_tokens=1,
                             model="fake")

    history = [{"role": "user", "content": "the flow legging"}]
    assert rewrite_followup("do you have that", history, StopwordLLM()) == "do you have that"


def _golden_rows():
    rows = []
    for path in sorted(ROOT.glob("domains/*/eval/golden.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_route_heuristic_accuracy_on_golden_set():
    rows = [r for r in _golden_rows() if r.get("route") and r.get("type") != "out_of_domain"]
    assert rows, "golden set has no routed questions"
    correct = sum(1 for r in rows if heuristic_route(r["question"]) == r["route"])
    accuracy = correct / len(rows)
    # the deterministic router clears the golden set today; keep a floor with margin so adding a
    # tricky question flags a real regression rather than passing silently
    assert accuracy >= 0.75, "route accuracy {:.2f} on {} questions".format(accuracy, len(rows))


def test_graph_uses_metric_and_graph_evidence(tmp_path):
    db = str(tmp_path / "lh.duckdb")
    build_lakehouse("apparel_ecommerce", db)
    graph_store = make_graph("memory")
    load_graph("apparel_ecommerce", db, graph_store)
    components = _seed_store()
    components["metric_resolver"] = MetricResolver("apparel_ecommerce", db)
    components["graph_retriever"] = make_graph_retriever("apparel_ecommerce", graph_store)

    class SlotLLM:
        model = "fake"

        def generate(self, prompt, *, system=None, max_tokens=512):
            if "Available metrics:" in prompt:
                text = '{"metric": "return_rate_by_size", "params": {"size": "M"}}'
            else:
                text = "the return rate for size M is 0.5 [1]"
            return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")

    components["llm"] = SlotLLM()
    metric_result = run_chat("what is the return rate for size M", components=components,
                             trace_path=str(tmp_path / "t.jsonl"))
    assert metric_result.trace["metric"] is True

    graph_result = run_chat("which supplier makes the Aster Cloud Hoodie", components=components,
                            trace_path=str(tmp_path / "t.jsonl"))
    assert graph_result.trace["graph"] is True
