"""M3.1 API: streaming chat, abstain, degraded mode, rate limiting, feedback (offline)."""
import json

from fastapi.testclient import TestClient

from adapters.factory import make_embedder, make_store
from adapters.fakes import EchoLLM
from api.app import create_app
from api.deps import get_components
from retrieval.sparse import SparseEncoder


def _components(llm=None):
    embedder = make_embedder("fake")
    encoder = SparseEncoder()
    store = make_store("memory")
    docs = [{"id": "R1", "text": "the belt bag costs 38 dollars and fits a phone",
             "payload": {"doc_type": "review", "product_id": "P006"}}]
    dense = embedder.embed([d["text"] for d in docs])
    store.upsert([
        {**d, "dense": dv, "sparse": {"indices": sv.indices, "values": sv.values}}
        for d, dv, sv in zip(docs, dense, [encoder.encode(d["text"]) for d in docs])
    ])
    return {"embedder": embedder, "store": store, "llm": llm or EchoLLM(), "reranker": None}


def _client(components, rate_limit="100/minute"):
    app = create_app(rate_limit=rate_limit)
    app.dependency_overrides[get_components] = lambda: components
    return TestClient(app)


def _sse(text):
    return [json.loads(ln[5:].strip()) for ln in text.splitlines() if ln.startswith("data:")]


def test_health():
    assert _client(_components()).get("/health").json()["status"] == "ok"


def test_chat_streams_tokens_then_final():
    resp = _client(_components()).post("/api/chat",
                                       json={"query": "how much does the belt bag cost"})
    assert resp.status_code == 200
    events = _sse(resp.text)
    assert any(e["type"] == "token" for e in events)
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] in ("auto", "abstain")
    assert "message_id" in final


def test_chat_abstains_out_of_scope():
    events = _sse(_client(_components()).post("/api/chat",
                                              json={"query": "capital of france"}).text)
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] == "abstain"
    assert final["citations"] == []


def test_chat_degrades_on_transient_error():
    class Boom:
        model = "x"

        def generate(self, *a, **k):
            raise RuntimeError("groq stream -> HTTP 429: rate limited")

        def stream(self, *a, **k):
            raise RuntimeError("groq stream -> HTTP 429: rate limited")

    events = _sse(_client(_components(llm=Boom())).post(
        "/api/chat", json={"query": "how much does the belt bag cost"}).text)
    assert [e for e in events if e["type"] == "final"][-1]["tier"] == "degraded"


def test_streaming_answer_keeps_citations():
    class CiteStreamLLM:
        model = "fake"

        def generate(self, *a, **k):
            raise NotImplementedError

        def stream(self, prompt, *, system=None, max_tokens=512):
            for piece in ("the belt bag costs 38 dollars ", "[1]."):
                yield piece

    events = _sse(_client(_components(llm=CiteStreamLLM())).post(
        "/api/chat", json={"query": "how much does the belt bag cost"}).text)
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] == "auto"
    assert final["citations"] and final["citations"][0]["n"] == 1
    assert final["grounding"] > 0.0


def test_midstream_failure_degrades_with_message_id():
    class MidFail:
        model = "fake"

        def generate(self, *a, **k):
            raise NotImplementedError

        def stream(self, prompt, *, system=None, max_tokens=512):
            yield "partial "
            raise RuntimeError("groq stream failed: connection reset")

    events = _sse(_client(_components(llm=MidFail())).post(
        "/api/chat", json={"query": "how much does the belt bag cost"}).text)
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] == "degraded"
    assert final.get("message_id")


def test_rate_limit_returns_429():
    client = _client(_components(), rate_limit="2/minute")
    q = {"query": "how much does the belt bag cost"}
    codes = [client.post("/api/chat", json=q).status_code for _ in range(4)]
    assert 429 in codes


def test_empty_query_rejected():
    assert _client(_components()).post("/api/chat", json={"query": "   "}).status_code == 400


def test_cors_allows_configured_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    resp = _client(_components()).get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_feedback_records(tmp_path, monkeypatch):
    import api.app as app_mod
    monkeypatch.setattr(app_mod, "_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    client = _client(_components())
    good = client.post("/api/feedback", json={"message_id": "m1", "verdict": "up"})
    assert good.status_code == 200
    assert (tmp_path / "fb.jsonl").read_text().strip()
    assert client.post("/api/feedback",
                       json={"message_id": "m1", "verdict": "maybe"}).status_code == 400
