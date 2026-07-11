"""M3.1/M3.3 API: streaming chat, degraded mode, rate limiting, feedback, and auth (offline)."""
import contextlib
import json
import os

import pytest
from fastapi.testclient import TestClient

from adapters import config
from adapters.config import get_settings
from adapters.factory import make_embedder, make_store
from adapters.fakes import EchoLLM
from api.app import create_app
from api.auth import create_access_token
from api.deps import get_components
from retrieval.sparse import SparseEncoder

_AUTH = {"Authorization": "Bearer " + create_access_token(
    "demo", "customer", get_settings().jwt_secret)}
_ADMIN = {"Authorization": "Bearer " + create_access_token(
    "admin", "admin", get_settings().jwt_secret)}


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
    from adapters.fakes import FakeTranscriber
    return {"embedder": embedder, "store": store, "llm": llm or EchoLLM(), "reranker": None,
            "transcriber": FakeTranscriber()}


def _client(components, rate_limit="100/minute", auth_db_path=None, chat_brain=None):
    app = create_app(rate_limit=rate_limit, auth_db_path=auth_db_path, chat_brain=chat_brain)
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

        def stream(self, prompt, *, system=None, max_tokens=512, usage_out=None):
            for piece in ("the belt bag costs 38 dollars ", "[1]."):
                yield piece
            if usage_out is not None:  # the metering path fills usage after the stream
                usage_out.update({"prompt_tokens": 40, "completion_tokens": 6, "model": "fake"})

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

        def stream(self, prompt, *, system=None, max_tokens=512, usage_out=None):
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


def test_retired_agent_brain_maps_to_omni(tmp_path):
    # the M6 LangGraph brain is retired; CHAT_BRAIN=agent now resolves to its omni successor and
    # still answers a grounded question over SSE rather than erroring on an unknown brain
    from rag.hitl import ReviewQueue
    components = _components()
    components["review_queue"] = ReviewQueue(str(tmp_path / "rq.db"))
    components["domain"] = "apparel_ecommerce"
    client = _client(components, chat_brain="agent")
    events = _sse(_chat(client, "how much does the belt bag cost").text)
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["tier"] == "auto" and final["citations"]


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


def _queue_client(tmp_path, *seed):
    from rag.hitl import ReviewQueue
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    ids = [queue.enqueue(q, domain="apparel_ecommerce") for q in seed]
    components = _components()
    components["review_queue"] = queue
    return _client(components), queue, ids


def test_admin_queue_requires_admin_role(tmp_path):
    client, _queue, _ids = _queue_client(tmp_path, "what is the SLA?")
    assert client.get("/api/admin/queue").status_code == 401           # no token
    assert client.get("/api/admin/queue", headers=_AUTH).status_code == 403   # customer, not admin
    assert client.get("/api/admin/queue", headers=_ADMIN).status_code == 200


def _claim(client, item_id):
    return client.post("/api/admin/queue/{}/claim".format(item_id), headers=_ADMIN)


def test_admin_claims_then_answers(tmp_path):
    client, queue, ids = _queue_client(tmp_path, "what is the SLA?")
    items = client.get("/api/admin/queue", headers=_ADMIN).json()["items"]
    assert items[0]["id"] == ids[0] and items[0]["status"] == "open"
    assert _claim(client, ids[0]).status_code == 200
    # the claimed item stays visible to its claimer (so it can be answered), now marked claimed
    items = client.get("/api/admin/queue", headers=_ADMIN).json()["items"]
    assert items[0]["status"] == "claimed"
    resp = client.post("/api/admin/queue/{}/answer".format(ids[0]),
                       json={"answer": "The SLA is 99.9 percent."}, headers=_ADMIN)
    assert resp.status_code == 200
    assert queue.get(ids[0])["status"] == "closed"


def test_double_claim_conflicts(tmp_path):
    client, _queue, ids = _queue_client(tmp_path, "what is the SLA?")
    assert _claim(client, ids[0]).status_code == 200
    assert _claim(client, ids[0]).status_code == 409


def test_admin_answer_unknown_item_is_404(tmp_path):
    client, _queue, _ids = _queue_client(tmp_path, "what is the SLA?")
    resp = client.post("/api/admin/queue/nope/answer", json={"answer": "x"}, headers=_ADMIN)
    assert resp.status_code == 404


def test_admin_answer_rejects_blank(tmp_path):
    client, _queue, ids = _queue_client(tmp_path, "what is the SLA?")
    _claim(client, ids[0])
    resp = client.post("/api/admin/queue/{}/answer".format(ids[0]),
                       json={"answer": "   "}, headers=_ADMIN)
    assert resp.status_code == 400  # a blank answer must not silently close the item


def test_admin_quality_requires_admin_and_returns_shape():
    client = _client(_components())
    assert client.get("/api/admin/quality", headers=_AUTH).status_code == 403
    resp = client.get("/api/admin/quality", headers=_ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert "overall" in body and "by_language" in body


def test_admin_domain_and_gaps_require_admin():
    client = _client(_components())
    for path in ("/api/admin/domain", "/api/admin/gaps", "/api/admin/health"):
        assert client.get(path, headers=_AUTH).status_code == 403
        assert client.get(path, headers=_ADMIN).status_code == 200
    body = client.get("/api/admin/domain", headers=_ADMIN).json()
    assert "ontology" in body and "metrics" in body and "lineage" in body


def test_admin_flywheel_reindexes_a_resolved_answer(tmp_path):
    from rag.hitl import ReviewQueue
    queue = ReviewQueue(str(tmp_path / "rq.db"))
    # empty domain so the run writes the verified eval to the gitignored traces/, not a real pack
    item_id = queue.enqueue("how long is the refund window", domain="")
    queue.resolve(item_id, "Refunds within 30 days.", "operator")
    components = _components()
    components["review_queue"] = queue
    components["domain"] = ""
    client = _client(components)
    assert client.post("/api/admin/flywheel", headers=_AUTH).status_code == 403
    resp = client.post("/api/admin/flywheel", headers=_ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["closed_items"] == 1 and body["indexed"] == 1 and "threshold" in body


def test_transcribe_returns_text():
    import base64
    client = _client(_components())
    audio = base64.b64encode(b"fake audio bytes").decode()
    resp = client.post("/api/transcribe", json={"audio_base64": audio}, headers=_AUTH)
    assert resp.status_code == 200 and resp.json()["text"] == "offline transcription"


def test_transcribe_requires_auth_and_rejects_bad_audio():
    client = _client(_components())
    assert client.post("/api/transcribe", json={"audio_base64": "AAAA"}).status_code == 401
    resp = client.post("/api/transcribe", json={"audio_base64": "not base64 %%%"}, headers=_AUTH)
    assert resp.status_code == 400  # invalid base64


def test_suggestions_requires_auth_and_returns_domain_prompts():
    # pin DOMAIN so the assertion does not depend on a developer's .env
    with _env(DOMAIN="apparel_ecommerce"):
        client = _client(_components())
        assert client.get("/api/suggestions").status_code == 401  # auth required
        body = client.get("/api/suggestions", headers=_AUTH).json()
        assert body["domain"] == "apparel_ecommerce"
        assert len(body["suggestions"]) >= 3
        first = body["suggestions"][0]
        assert set(first) == {"text", "lang", "kind"} and first["text"]


def test_catalog_is_public_and_returns_a_product_list():
    # the storefront shows products before login, so no auth; shape is a list (possibly empty
    # without a built lakehouse)
    body = _client(_components()).get("/api/catalog").json()
    assert "products" in body and isinstance(body["products"], list)


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


@contextlib.contextmanager
def _env(**overrides):
    # Settings are lru_cached, so set the env, clear the cache, and (critically) restore the env
    # BEFORE the final clear so no later test inherits these settings.
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update({k: v for k, v in overrides.items()})
    config.get_settings.cache_clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        config.get_settings.cache_clear()


# A real production secret: not a placeholder and at least 32 characters.
_STRONG_SECRET = "a-long-random-production-jwt-secret-0123456789"
# Explicit non-default credentials so the production credential gate does not fire on these tests.
# Gate credentials are included so the boot check does not depend on a developer's local .env (CI
# has none, so an empty GATE_USERNAME/GATE_PASSWORD would otherwise fail the production-boot test).
_REAL_CREDS = {"ADMIN_PASSWORD": "a-real-admin-password", "DEMO_PASSWORD": "a-real-demo-password",
               "GATE_USERNAME": "gatekeeper", "GATE_PASSWORD": "a-real-gate-password"}


def test_production_refuses_insecure_jwt_secret():
    # A public deploy with a forgeable admin token must fail to boot, not just warn.
    with _env(SKEIN_ENV="production", JWT_SECRET="dev-insecure-change-me",
              TURNSTILE_SECRET_KEY="a-secret", **_REAL_CREDS):
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            create_app()


def test_production_refuses_the_shipped_placeholder_secret():
    # The .env.example value is 33 chars, so the length check alone misses it; it is denylisted.
    with _env(SKEIN_ENV="production", JWT_SECRET="change-me-to-a-long-random-string",
              TURNSTILE_SECRET_KEY="a-secret", **_REAL_CREDS):
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            create_app()


def test_production_refuses_short_secret():
    with _env(SKEIN_ENV="production", JWT_SECRET="too-short", TURNSTILE_SECRET_KEY="a-secret",
              **_REAL_CREDS):
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            create_app()


def test_production_refuses_missing_turnstile():
    with _env(SKEIN_ENV="production", JWT_SECRET=_STRONG_SECRET, TURNSTILE_SECRET_KEY="",
              **_REAL_CREDS):
        with pytest.raises(RuntimeError, match="TURNSTILE"):
            create_app()


def test_production_refuses_default_admin_password():
    with _env(SKEIN_ENV="production", JWT_SECRET=_STRONG_SECRET, TURNSTILE_SECRET_KEY="a-secret",
              ADMIN_PASSWORD="skein-admin-2026", DEMO_PASSWORD="a-real-demo-password"):
        with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
            create_app()


def test_production_refuses_default_neo4j_password():
    with _env(SKEIN_ENV="production", JWT_SECRET=_STRONG_SECRET, TURNSTILE_SECRET_KEY="a-secret",
              GRAPH_PROVIDER="neo4j", NEO4J_PASSWORD="skein_password", **_REAL_CREDS):
        with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
            create_app()


def test_production_boots_with_real_secret_and_captcha(tmp_path):
    # GRAPH_PROVIDER=memory so the neo4j credential gate is not exercised here (a dev .env may set
    # neo4j with the default password; that path has its own test above).
    with _env(SKEIN_ENV="production", JWT_SECRET=_STRONG_SECRET, TURNSTILE_SECRET_KEY="a-secret",
              GRAPH_PROVIDER="memory", **_REAL_CREDS):
        app = create_app(auth_db_path=str(tmp_path / "auth.db"))
        assert TestClient(app).get("/health").json()["status"] == "ok"


def test_chunked_transfer_encoding_is_rejected():
    resp = _client(_components()).post("/api/chat", json={"query": "hi"}, headers={
        **_AUTH, "Transfer-Encoding": "chunked"})
    assert resp.status_code == 411


def test_demo_readonly_blocks_mutations_but_not_reads(tmp_path):
    with _env(DEMO_READONLY="true"):
        client, _queue, ids = _queue_client(tmp_path, "what is the SLA?")
        # reads still work
        assert client.get("/api/admin/queue", headers=_ADMIN).status_code == 200
        # every mutating admin endpoint is refused, even for a real admin token
        assert client.post("/api/admin/flywheel", headers=_ADMIN).status_code == 403
        assert client.post("/api/admin/queue/{}/claim".format(ids[0]),
                           headers=_ADMIN).status_code == 403
        assert client.post("/api/admin/queue/{}/answer".format(ids[0]),
                           json={"answer": "x"}, headers=_ADMIN).status_code == 403
