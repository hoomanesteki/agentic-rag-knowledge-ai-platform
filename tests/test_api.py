"""M3.1/M3.3 API: streaming chat, degraded mode, rate limiting, feedback, and auth (offline)."""
import json

from fastapi.testclient import TestClient

from adapters.config import get_settings
from adapters.factory import make_embedder, make_store
from adapters.fakes import EchoLLM
from api.app import create_app
from api.auth import create_access_token
from api.deps import get_components
from retrieval.sparse import SparseEncoder

_AUTH = {"Authorization": "Bearer " + create_access_token(
    "demo", "customer", get_settings().jwt_secret)}


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


def _client(components, rate_limit="100/minute", auth_db_path=None):
    app = create_app(rate_limit=rate_limit, auth_db_path=auth_db_path)
    app.dependency_overrides[get_components] = lambda: components
    return TestClient(app)


def _sse(text):
    return [json.loads(ln[5:].strip()) for ln in text.splitlines() if ln.startswith("data:")]


def _chat(client, query):
    return client.post("/api/chat", json={"query": query}, headers=_AUTH)


def test_health():
    assert _client(_components()).get("/health").json()["status"] == "ok"


def test_chat_requires_auth():
    resp = _client(_components()).post("/api/chat", json={"query": "hello"})
    assert resp.status_code == 401


def test_chat_streams_tokens_then_final():
    resp = _chat(_client(_components()), "how much does the belt bag cost")
    assert resp.status_code == 200
    events = _sse(resp.text)
    assert any(e["type"] == "token" for e in events)
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] in ("auto", "abstain")
    assert "message_id" in final


def test_chat_abstains_out_of_scope():
    events = _sse(_chat(_client(_components()), "capital of france").text)
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

    events = _sse(_chat(_client(_components(llm=Boom())), "how much does the belt bag cost").text)
    assert [e for e in events if e["type"] == "final"][-1]["tier"] == "degraded"


def test_streaming_answer_keeps_citations():
    class CiteStreamLLM:
        model = "fake"

        def generate(self, *a, **k):
            raise NotImplementedError

        def stream(self, prompt, *, system=None, max_tokens=512):
            for piece in ("the belt bag costs 38 dollars ", "[1]."):
                yield piece

    events = _sse(_chat(_client(_components(llm=CiteStreamLLM())),
                        "how much does the belt bag cost").text)
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

    events = _sse(_chat(_client(_components(llm=MidFail())),
                        "how much does the belt bag cost").text)
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] == "degraded"
    assert final.get("message_id")


def test_rate_limit_returns_429():
    client = _client(_components(), rate_limit="2/minute")
    codes = [_chat(client, "how much does the belt bag cost").status_code for _ in range(4)]
    assert 429 in codes


def test_empty_query_rejected():
    resp = _client(_components()).post("/api/chat", json={"query": "   "}, headers=_AUTH)
    assert resp.status_code == 400


def test_cors_allows_configured_origin(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    resp = _client(_components()).get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_login_succeeds_with_demo_user(tmp_path):
    client = _client(_components(), auth_db_path=str(tmp_path / "auth.db"))
    resp = client.post("/api/login",
                       json={"username": "demo", "password": get_settings().demo_password})
    assert resp.status_code == 200
    assert resp.json()["access_token"]
    assert resp.json()["role"] == "customer"


def test_login_rejects_bad_password(tmp_path):
    client = _client(_components(), auth_db_path=str(tmp_path / "auth.db"))
    resp = client.post("/api/login", json={"username": "demo", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user_same_as_bad_password(tmp_path):
    client = _client(_components(), auth_db_path=str(tmp_path / "auth.db"))
    resp = client.post("/api/login", json={"username": "nobody", "password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "invalid credentials"  # no user enumeration


def test_chat_rejects_forged_token():
    forged = "Bearer " + create_access_token("demo", "customer", "a-different-secret")
    resp = _client(_components()).post("/api/chat", json={"query": "hi"},
                                       headers={"Authorization": forged})
    assert resp.status_code == 401


def test_chat_rejects_expired_token():
    expired = "Bearer " + create_access_token(
        "demo", "customer", get_settings().jwt_secret, expires_min=-1)
    resp = _client(_components()).post("/api/chat", json={"query": "hi"},
                                       headers={"Authorization": expired})
    assert resp.status_code == 401


def test_feedback_requires_auth_and_records(tmp_path, monkeypatch):
    import api.app as app_mod
    monkeypatch.setattr(app_mod, "_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    client = _client(_components())
    payload = {"message_id": "m1", "verdict": "up"}
    assert client.post("/api/feedback", json=payload).status_code == 401
    assert client.post("/api/feedback", json=payload, headers=_AUTH).status_code == 200
    assert (tmp_path / "fb.jsonl").read_text().strip()
