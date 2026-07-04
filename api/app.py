"""Customer-facing FastAPI app: streaming chat and feedback, with rate limiting and a
degraded-mode fallback when hosted APIs are unavailable."""
import base64
import binascii
import json
import logging
import os
import time
import uuid
from functools import lru_cache

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from adapters.config import get_settings
from adapters.elevenlabs import ElevenLabsTTS
from adapters.observability import flush, request_span, update_span
from api.auth import (
    DUMMY_HASH,
    UserStore,
    create_access_token,
    decode_token,
    seed_demo_user,
    verify_password,
    verify_turnstile,
)
from api.deps import get_components
from api.ratelimit import RateLimiter
from api.resilience import is_transient
from data.introspect import lineage_view, metrics_view, ontology_view
from data.lakehouse import load_manifest
from evaluation.monitoring import aggregate_gaps, aggregate_health, aggregate_quality, read_jsonl
from pipeline.answer import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_TRACE_PATH,
    _smalltalk,
    stream_answer,
    write_trace,
)
from rag.agent import answer_with_agent
from rag.flywheel import grow_verified_eval, reindex_verified, suggest_threshold

_FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "traces/feedback.jsonl")
_MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB decoded: a short voice clip, not a file upload
_MAX_BODY_BYTES = 15 * 1024 * 1024   # reject an oversized body before parsing it
_ALLOWED_AUDIO_MIME = {"audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/mp3",
                       "audio/wav", "audio/x-wav", "audio/flac"}
_DEGRADED = "The assistant is busy right now. Please try again in a moment."
_RATE_LIMITED = ("I'm getting more questions than the demo's free API tier allows right now. "
                 "Please wait about 20 seconds and ask again.")
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# Placeholders that must never sign real tokens. The length check below catches anything else
# too short, but the shipped .env.example value is 33 chars, so it has to be named explicitly.
_INSECURE_JWT_SECRETS = {"dev-insecure-change-me", "change-me", "change-me-to-a-long-random-string"}
_MIN_JWT_SECRET_LEN = 32
_DEFAULT_ADMIN_PASSWORD = "skein-admin-2026"  # the value committed in .env.example / config.py
_DEFAULT_DEMO_PASSWORD = "Canada54321"
_DEFAULT_NEO4J_PASSWORD = "skein_password"
_log = logging.getLogger("skein.api")


def _is_production(app_env: str) -> bool:
    return app_env in ("production", "prod")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    lang: str | None = Field(default=None, max_length=8)
    session_id: str | None = Field(default=None, max_length=64)
    # "agent" answers in the human specialist's voice after a shopper is escalated.
    persona: str | None = Field(default=None, max_length=16)
    # prior turns [{"role": "user"|"assistant", "content": str}] so a follow-up can be rewritten
    history: list[dict] | None = Field(default=None, max_length=20)


class FeedbackRequest(BaseModel):
    message_id: str = Field(max_length=64)
    verdict: str  # "up" or "down"
    note: str | None = Field(default=None, max_length=2000)


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)


class TranscribeRequest(BaseModel):
    # a short voice clip, base64-encoded (about 13.4M chars caps the decoded audio near 10 MB)
    audio_base64: str = Field(min_length=1, max_length=13_400_000)
    mime: str = Field(default="audio/webm", max_length=100)
    lang: str | None = Field(default=None, max_length=8)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    # "agent" uses the human specialist's voice; anything else uses the assistant voice
    persona: str | None = Field(default=None, max_length=16)


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=72)  # bcrypt only uses the first 72 bytes
    turnstile_token: str | None = Field(default=None, max_length=4096)


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


@lru_cache
def _catalog(domain: str) -> list:
    """The product catalog for the storefront, one card per product (size variants collapsed).
    Reads the governed gold `products` table; returns [] if the domain has no catalog or the
    lakehouse is not built yet."""
    import duckdb
    db = os.getenv("LAKEHOUSE_DB", "lakehouse.duckdb")
    if not os.path.exists(db):
        return []
    con = duckdb.connect(db, read_only=True, config={"enable_external_access": False})
    try:
        has = con.execute("SELECT count(*) FROM information_schema.tables "
                          "WHERE table_name = 'products' AND table_schema = 'main'").fetchone()[0]
        if not has:
            return []
        rows = con.execute(
            "SELECT any_value(product_id) AS id, name, any_value(category) AS category, "
            "any_value(gender) AS gender, min(price) AS price, any_value(color) AS color, "
            "any_value(colors) AS colors, any_value(weather) AS weather, "
            "list(DISTINCT size ORDER BY size) AS sizes, sum(stock) AS stock "
            "FROM products GROUP BY name ORDER BY category, name").fetchall()
    except duckdb.Error:
        return []
    finally:
        con.close()
    return [{"id": r[0], "name": r[1], "category": r[2], "gender": r[3], "price": r[4],
             "color": r[5], "colors": [c for c in (r[6] or "").split("|") if c],
             "weather": r[7], "sizes": list(r[8]), "stock": int(r[9] or 0)} for r in rows]


@lru_cache
def _product(domain: str, pid: str) -> dict | None:
    """One product's full page: the catalog card plus its marketing copy and any reviews."""
    prod = next((p for p in _catalog(domain) if p["id"] == pid), None)
    if not prod:
        return None
    name = prod["name"]
    seed = os.path.join("domains", domain, "seed", "unstructured")
    desc = ""
    for fn in ("products.jsonl", "products_catalog.jsonl"):  # copy whose text names this product
        path = os.path.join(seed, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if name.lower() in d.get("text", "").lower():
                    desc = d["text"]
                    break
        if desc:
            break
    ids: set = set()
    import duckdb
    db = os.getenv("LAKEHOUSE_DB", "lakehouse.duckdb")
    if os.path.exists(db):
        con = duckdb.connect(db, read_only=True, config={"enable_external_access": False})
        try:
            ids = {r[0] for r in con.execute(
                "SELECT product_id FROM products WHERE name = ?", [name]).fetchall()}
        except duckdb.Error:
            ids = set()
        finally:
            con.close()
    reviews = []
    rpath = os.path.join(seed, "reviews.jsonl")
    if ids and os.path.exists(rpath):
        with open(rpath, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("product_id") in ids:
                    reviews.append({"text": d.get("text", ""), "rating": d.get("rating")})
    return {**prod, "description": desc, "reviews": reviews[:6]}


@lru_cache
def _brand(domain: str) -> str:
    """The active domain's display brand, read from its manifest, so the storefront names itself
    from the pack instead of the engine hardcoding it."""
    return str(load_manifest(os.path.join("domains", domain)).get("brand", "") or "")[:80]


@lru_cache
def _suggestions(domain: str) -> list:
    """The active domain's starter prompts, read from its manifest. Only text/lang/kind are
    exposed and the list is capped, so a pack cannot push arbitrary fields to the client."""
    items = load_manifest(os.path.join("domains", domain)).get("suggestions", []) or []
    if not isinstance(items, list):  # a malformed pack (mapping, not list) must not 500 the route
        return []
    return [{"text": str(s.get("text", ""))[:200],
             "lang": str(s.get("lang", "en"))[:8],
             "kind": str(s.get("kind", "fact"))[:16]}
            for s in items[:12] if isinstance(s, dict) and s.get("text")]


def create_app(rate_limit: str | None = None, auth_db_path: str | None = None,
               chat_brain: str | None = None) -> FastAPI:
    app = FastAPI(title="Skein Lite API")
    settings = get_settings()
    brain = chat_brain or settings.chat_brain  # "linear" streams; "agent" runs the M6 brain
    limiter = RateLimiter(rate_limit or settings.rate_limit)
    # Default allows the web app at either localhost or 127.0.0.1 (browsers pick either), so local
    # dev works without CORS surprises. Production sets ALLOWED_ORIGINS to the real origin.
    origins = [o.strip() for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["*"])

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        # reject an oversized body on Content-Length before Starlette reads and parses it, so a
        # voice clip cannot be used to force a huge allocation
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > _MAX_BODY_BYTES:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        # A chunked body carries no Content-Length, so it would slip past the check above and be
        # buffered unbounded before pydantic's field caps run. This app's clients always send a
        # Content-Length, so refuse chunked uploads outright.
        if "chunked" in request.headers.get("transfer-encoding", "").lower():
            return JSONResponse({"detail": "chunked transfer-encoding is not accepted"},
                                status_code=411)
        return await call_next(request)

    login_limiter = RateLimiter("5/minute")  # tighter bucket for the credential endpoint
    # demo-login takes no password (it just mints a demo token), so the strict credential bucket is
    # wrong for it: a page load + StrictMode double-fire + a retry can exhaust 5/min and 429 the
    # visitor into a stuck "connecting" state. Give it a roomy bucket of its own.
    demo_limiter = RateLimiter("60/minute")
    # a separate, roomier bucket for voice: each spoken turn is one /api/chat plus one /api/tts, so
    # sharing the chat bucket would halve the real conversation rate and cut the voice mid-demo
    tts_limiter = RateLimiter("60/minute")
    store = UserStore(auth_db_path or settings.auth_db_path)
    seed_demo_user(store, settings.demo_username, settings.demo_password)
    seed_demo_user(store, settings.admin_username, settings.admin_password, role="admin")

    production = _is_production(settings.app_env)
    # A secret is unsafe if it is a known placeholder or simply too short to resist brute force.
    weak_jwt = (settings.jwt_secret in _INSECURE_JWT_SECRETS
                or len(settings.jwt_secret) < _MIN_JWT_SECRET_LEN)
    if weak_jwt:
        msg = ("JWT_SECRET is weak or a placeholder: anyone can forge a token, including an admin "
               "one. Set JWT_SECRET to a random string of at least {} characters.".format(
                   _MIN_JWT_SECRET_LEN))
        if production:
            # Fail fast instead of booting a forgeable-auth server on a public URL.
            raise RuntimeError(msg + " Refusing to start with SKEIN_ENV=production.")
        _log.error(msg)
    if production and not settings.turnstile_secret:
        raise RuntimeError("TURNSTILE_SECRET_KEY is empty but SKEIN_ENV=production; the login "
                           "captcha would be bypassed. Set it or unset SKEIN_ENV.")
    if production:
        # The default credentials are committed in .env.example, so a public deploy that keeps
        # them is wide open (admin dashboards leak ops data even with DEMO_READONLY on).
        if settings.admin_password == _DEFAULT_ADMIN_PASSWORD:
            raise RuntimeError("ADMIN_PASSWORD is the documented default; set a real one before "
                               "SKEIN_ENV=production.")
        if settings.demo_password == _DEFAULT_DEMO_PASSWORD:
            raise RuntimeError("DEMO_PASSWORD is the documented default; set a real one before "
                               "SKEIN_ENV=production.")
        if settings.graph_provider == "neo4j" and \
                settings.neo4j_password == _DEFAULT_NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_PASSWORD is the documented default; set a real one before "
                               "SKEIN_ENV=production.")
        if not (settings.gate_username and settings.gate_password):
            # Empty gate credentials leave /api/gate-login open, and it mints a chat token.
            raise RuntimeError("GATE_USERNAME and GATE_PASSWORD must be set before "
                               "SKEIN_ENV=production, or the demo gate is open to anyone.")
    if not settings.turnstile_secret:
        _log.warning("TURNSTILE_SECRET_KEY is empty; the login captcha is bypassed (dev only).")
    if settings.demo_readonly:
        _log.info("DEMO_READONLY is on; mutating admin endpoints are disabled.")

    def deny_if_readonly() -> None:
        if settings.demo_readonly:
            raise HTTPException(status_code=403, detail="this is a read-only demo")

    def client_key(request: Request) -> str:
        # request.client.host is the direct peer; behind Cloud Run that is the proxy hop, so every
        # caller would share one bucket and a single abuser would lock everyone out. In production
        # key on the client IP from X-Forwarded-For. Cloud Run's front end appends the verified
        # client IP as the LAST entry and does not strip client-supplied ones, so the last entry is
        # the trustworthy hop (the leftmost is attacker-controlled). Assumes direct run.app; a
        # fronting load balancer adds another hop. The hard cost ceiling is the instance cap.
        if production:
            last = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
            if last:
                return last
        return request.client.host if request.client else "anon"

    def current_user(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="not authenticated",
                                headers={"WWW-Authenticate": "Bearer"})
        try:
            payload = decode_token(authorization.split(" ", 1)[1], settings.jwt_secret)
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="invalid or expired token",
                                headers={"WWW-Authenticate": "Bearer"})
        return {"username": payload.get("sub"), "role": payload.get("role")}

    def require_admin(user: dict = Depends(current_user)) -> dict:
        if user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="admin only")
        return user

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.on_event("shutdown")
    def _flush_traces():
        # The batch processor exports traces on an interval; flush once on shutdown so a final
        # turn is not lost. Not per-request: a per-turn flush would block the response on a slow
        # or unreachable Langfuse. No-op when tracing is off.
        flush()

    @app.post("/api/login")
    def login(body: LoginRequest, request: Request):
        if not login_limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="too many attempts",
                                headers={"Retry-After": "30"})
        if not verify_turnstile(body.turnstile_token, settings.turnstile_secret):
            raise HTTPException(status_code=403, detail="captcha verification failed")
        user = store.get(body.username)
        # Always run a bcrypt compare (dummy hash for unknown users) so timing does not leak
        # whether a username exists.
        password_hash = user["password_hash"] if user else DUMMY_HASH
        if not verify_password(body.password, password_hash) or not user:
            raise HTTPException(status_code=401, detail="invalid credentials")
        token = create_access_token(user["username"], user["role"], settings.jwt_secret)
        return {"access_token": token, "token_type": "bearer", "role": user["role"]}

    @app.post("/api/demo-login")
    def demo_login(request: Request):
        # Frictionless demo: outside production, hand a visitor a demo-user token so they can land
        # on the store and just ask, no password wall. Disabled in production, where the gate is
        # the only way in. Rate-limited so it cannot be used to mint tokens in a loop.
        if production:
            raise HTTPException(status_code=404, detail="not found")
        if not demo_limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="too many attempts",
                                headers={"Retry-After": "10"})
        user = store.get(settings.demo_username)
        if not user:
            raise HTTPException(status_code=404, detail="no demo user")
        token = create_access_token(user["username"], user["role"], settings.jwt_secret)
        return {"access_token": token, "token_type": "bearer", "role": user["role"]}

    @app.post("/api/gate-login")
    def gate_login(body: LoginRequest, request: Request):
        # The public landing page gates the demo behind one shared credential plus a captcha, so a
        # link shared with a reviewer is not open to bots. Rate-limited to deter brute force / DDoS.
        # If no gate credential is configured, the gate is open (local dev).
        if not login_limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="too many attempts",
                                headers={"Retry-After": "30"})
        if not verify_turnstile(body.turnstile_token, settings.turnstile_secret):
            raise HTTPException(status_code=403, detail="captcha verification failed")
        import hmac
        gu, gp = settings.gate_username, settings.gate_password
        # compare bytes, so an accented character is a wrong password, not a 500. Both halves always
        # run (no and-short-circuit) so timing does not leak whether the username matched.
        user_ok = hmac.compare_digest((body.username or "").encode(), gu.encode())
        pass_ok = hmac.compare_digest((body.password or "").encode(), gp.encode())
        if not ((not gu and not gp) or (user_ok & pass_ok)):
            raise HTTPException(status_code=401, detail="invalid credentials")
        return {"access_token": create_access_token("gate", "gate", settings.jwt_secret),
                "token_type": "bearer"}

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request, comp: dict = Depends(get_components),
             user: dict = Depends(current_user)):
        if not limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": "10"})
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="query is required")

        message_id = uuid.uuid4().hex
        started = time.perf_counter()

        def event_stream():
            try:
                if brain == "agent":
                    # safety/greeting intercept BEFORE the brain, so the agent path gets the same
                    # harm-decline and small-talk handling as the linear path (the brain does not
                    # call _smalltalk itself)
                    chat = _smalltalk(req.query, req.persona)
                    if chat is not None:
                        yield _sse({"type": "token", "text": chat})
                        yield _sse({"type": "final", "message_id": message_id, "answer": chat,
                                    "tier": "auto", "confidence": 1.0, "grounding": 1.0,
                                    "citations": []})
                        return
                    # the full M6 brain (supervisor, gate, escalation to the review queue) as a
                    # buffered response. The whole turn runs synchronously here before the first
                    # yield, so the Langfuse span opens and closes within one execution (no cross-
                    # yield context hop) and every LLM generation nests under it. No-op when off.
                    with request_span("chat.agent", input=req.query,
                                      metadata={"message_id": message_id, "lang": req.lang}):
                        result = answer_with_agent(
                            req.query, components=comp, history=req.history or [],
                            message_id=message_id, review_queue=comp.get("review_queue"),
                            domain=comp.get("domain"), lang=req.lang)
                        update_span(output=result.answer, metadata={"tier": result.tier})
                    yield _sse({"type": "token", "text": result.answer})
                    yield _sse({"type": "final", "message_id": message_id,
                                "answer": result.answer, "tier": result.tier,
                                "confidence": round(result.confidence, 3),
                                "grounding": round(result.grounding, 3),
                                "citations": result.citations,
                                "escalation_id": result.trace.get("escalation_id")})
                    return
                # Linear path: tokens stream across threadpool resumes, so we do not open a span
                # around the generator (its OTel context would not survive the hops). The individual
                # generate() calls are still traced by the adapter.
                for event in stream_answer(req.query, message_id=message_id,
                                           embedder=comp["embedder"], store=comp["store"],
                                           llm=comp["llm"], reranker=comp["reranker"],
                                           metric_resolver=comp.get("metric_resolver"),
                                           graph_retriever=comp.get("graph_retriever"),
                                           lang=req.lang, persona=req.persona,
                                           history=req.history):
                    yield _sse(event)
            # Catch broadly: the response is already a 200 SSE stream, so any failure (a hosted
            # SDK error not wrapped as RuntimeError, a mid-stream drop) must surface as an event,
            # never a silent dead stream. is_transient decides degraded vs honest error.
            except Exception as exc:
                latency = round((time.perf_counter() - started) * 1000, 1)
                transient = is_transient(exc)
                write_trace({"ts": time.time(), "message_id": message_id, "query": req.query,
                             "lang": req.lang,
                             "tier": "degraded" if transient else "error", "streamed": True,
                             "error": str(exc)[:200], "latency_ms": latency}, DEFAULT_TRACE_PATH)
                if transient:
                    # a 429 is the metered free tier, not a real outage: say so, so the demo reads
                    # as a rate limit to wait out rather than a broken assistant
                    msg = _RATE_LIMITED if "429" in str(exc) else _DEGRADED
                    yield _sse({"type": "token", "text": msg})
                    yield _sse({"type": "final", "message_id": message_id, "tier": "degraded",
                                "answer": msg, "confidence": 0.0, "grounding": 0.0,
                                "citations": []})
                else:
                    yield _sse({"type": "error", "message_id": message_id,
                                "message": "internal error"})

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers=_SSE_HEADERS)

    @app.get("/api/suggestions")
    def suggestions(user: dict = Depends(current_user)):
        # starter prompts for the active domain, so the chat guides the user instead of showing a
        # blank box; served from the pack, so switching DOMAIN switches the suggestions
        return {"domain": settings.domain, "suggestions": _suggestions(settings.domain)}

    @app.get("/api/catalog")
    def catalog():
        # public: the storefront shows products before login, like a real store. The brand comes
        # from the pack so the engine stays domain agnostic.
        return {
            "domain": settings.domain,
            "brand": _brand(settings.domain),
            "products": _catalog(settings.domain),
        }

    @app.get("/api/product/{pid}")
    def product(pid: str, request: Request):
        # public: a product detail page (basics + marketing copy + reviews). Rate-limited so a
        # scraper cannot cycle ids to hammer the file reads.
        if not limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": "10"})
        p = _product(settings.domain, pid)
        if not p:
            raise HTTPException(status_code=404, detail="product not found")
        return p

    @app.post("/api/transcribe")
    def transcribe(body: TranscribeRequest, request: Request,
                   comp: dict = Depends(get_components), user: dict = Depends(current_user)):
        if not limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": "10"})
        try:
            audio = base64.b64decode(body.audio_base64, validate=True)
        except (ValueError, binascii.Error):
            raise HTTPException(status_code=400, detail="audio_base64 is not valid base64")
        if not audio or len(audio) > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=400, detail="audio is empty or too large")
        # allowlist the mime so a client cannot inject a header value into the upstream multipart
        mime = body.mime.split(";")[0].strip().lower()
        if mime not in _ALLOWED_AUDIO_MIME:
            mime = "audio/webm"
        try:
            text = comp["transcriber"].transcribe(audio, mime=mime, language=body.lang)
        except Exception as exc:  # a hosted STT failure; the client falls back to Web Speech
            _log.warning("transcription failed: %s", str(exc)[:200])
            raise HTTPException(status_code=502, detail="transcription unavailable")
        return {"text": text}

    @app.post("/api/tts")
    def tts(body: TTSRequest, request: Request, user: dict = Depends(current_user)):
        # Premium voice for spoken answers. The API key never leaves the server. With no provider
        # or key configured, return 204 so the browser uses its built-in speechSynthesis instead.
        if not tts_limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": "10"})
        if settings.tts_provider != "elevenlabs" or not settings.elevenlabs_api_key:
            return Response(status_code=204)
        voice = (settings.elevenlabs_agent_voice_id if body.persona == "agent"
                 else settings.elevenlabs_voice_id)
        try:
            audio = ElevenLabsTTS(settings.elevenlabs_api_key,
                                  model=settings.elevenlabs_model).synthesize(body.text, voice)
        except Exception as exc:  # upstream failure: the client falls back to its built-in voice
            _log.warning("tts failed: %s", str(exc)[:200])
            return Response(status_code=204)
        return Response(content=audio, media_type="audio/mpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/feedback")
    def feedback(fb: FeedbackRequest, request: Request, user: dict = Depends(current_user)):
        if not limiter.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": "10"})
        if fb.verdict not in ("up", "down"):
            raise HTTPException(status_code=400, detail="verdict must be 'up' or 'down'")
        os.makedirs(os.path.dirname(_FEEDBACK_PATH) or ".", exist_ok=True)
        # Attribute the feedback to the authenticated user so entries are not anonymous or
        # spoofable; the flywheel (M7.3) needs to know who rated what.
        record = {"ts": time.time(), "username": user["username"], **fb.model_dump()}
        with open(_FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {"status": "recorded"}

    @app.get("/api/admin/quality")
    def admin_quality(_: dict = Depends(require_admin)):
        traces = read_jsonl(DEFAULT_TRACE_PATH, limit=5000)
        feedback = read_jsonl(_FEEDBACK_PATH, limit=5000)
        return aggregate_quality(traces, feedback)

    @app.get("/api/admin/health")
    def admin_health(_: dict = Depends(require_admin)):
        # live platform health from recent traffic (p95 latency, throughput, error rate, cost)
        return aggregate_health(read_jsonl(DEFAULT_TRACE_PATH, limit=5000))

    @app.get("/api/admin/gaps")
    def admin_gaps(_: dict = Depends(require_admin)):
        # questions the system could not answer well: the worklist for what to teach it next
        return {"gaps": aggregate_gaps(read_jsonl(DEFAULT_TRACE_PATH, limit=5000))}

    @app.get("/api/admin/analytics")
    def admin_analytics(_: dict = Depends(require_admin)):
        # store-manager metrics (traffic, top searches/questions, funnel, revenue) from the pack's
        # simulated sessions. Business insights, not ML internals.
        path = os.path.join("domains", settings.domain, "seed", "analytics.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @app.get("/api/admin/domain")
    def admin_domain(_: dict = Depends(require_admin)):
        # read-only structure of the active domain, plus a link to MLflow (wired at M8)
        domain = settings.domain
        return {"domain": domain, "ontology": ontology_view(domain),
                "metrics": metrics_view(domain), "lineage": lineage_view(domain),
                "mlflow_url": settings.mlflow_url or None,
                "langfuse_url": settings.langfuse_url or None}

    @app.post("/api/admin/flywheel")
    def admin_flywheel(comp: dict = Depends(get_components), _: dict = Depends(require_admin)):
        deny_if_readonly()  # re-embedding costs money; off in the public demo
        queue = comp["review_queue"]
        domain = comp.get("domain") or ""
        # only items for this domain, resolved since the last run, so re-embedding is not repeated
        items = queue.closed_since(queue.flywheel_watermark(domain), domain=domain)
        indexed = reindex_verified(items, comp["embedder"], comp["store"])
        # a growing eval set under traces/ (gitignored), never written into a git-tracked pack
        eval_path = "traces/verified_eval_{}.jsonl".format(domain or "default")
        grown = grow_verified_eval(items, eval_path)
        if items:
            queue.advance_flywheel_watermark(domain, max(it["resolved_at"] for it in items))
        quality = aggregate_quality(read_jsonl(DEFAULT_TRACE_PATH, 5000),
                                    read_jsonl(_FEEDBACK_PATH, 5000))
        return {"closed_items": len(items), "indexed": indexed, "grown": grown,
                "threshold": suggest_threshold(quality, DEFAULT_MIN_CONFIDENCE)}

    @app.get("/api/admin/queue")
    def admin_queue(comp: dict = Depends(get_components), user: dict = Depends(require_admin)):
        # open items to claim plus the caller's own claimed items to answer
        return {"items": comp["review_queue"].list_actionable(user["username"])}

    @app.post("/api/admin/queue/{item_id}/claim")
    def admin_claim(item_id: str, comp: dict = Depends(get_components),
                    user: dict = Depends(require_admin)):
        deny_if_readonly()
        queue = comp["review_queue"]
        if queue.get(item_id) is None:
            raise HTTPException(status_code=404, detail="no such item")
        if not queue.claim(item_id, user["username"]):
            raise HTTPException(status_code=409, detail="already claimed or closed")
        return {"status": "claimed", "id": item_id, "by": user["username"]}

    @app.post("/api/admin/queue/{item_id}/answer")
    def admin_answer(item_id: str, body: AnswerRequest, comp: dict = Depends(get_components),
                     user: dict = Depends(require_admin)):
        deny_if_readonly()
        queue = comp["review_queue"]
        if not body.answer.strip():
            raise HTTPException(status_code=400, detail="answer is required")
        if queue.get(item_id) is None:
            raise HTTPException(status_code=404, detail="no such item")
        if not queue.resolve(item_id, body.answer, user["username"]):
            raise HTTPException(status_code=409,
                                detail="not open, or claimed by another operator")
        return {"status": "closed", "id": item_id}

    return app


app = create_app()
