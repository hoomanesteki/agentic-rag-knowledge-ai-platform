"""Customer-facing FastAPI app: streaming chat and feedback, with rate limiting and a
degraded-mode fallback when hosted APIs are unavailable."""
import json
import logging
import os
import time
import uuid

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from adapters.config import get_settings
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
from pipeline.answer import DEFAULT_TRACE_PATH, stream_answer, write_trace
from rag.agent import answer_with_agent

_FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "traces/feedback.jsonl")
_DEGRADED = "The assistant is busy right now. Please try again in a moment."
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_INSECURE_JWT_SECRET = "dev-insecure-change-me"
_log = logging.getLogger("skein.api")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    lang: str | None = Field(default=None, max_length=8)
    session_id: str | None = Field(default=None, max_length=64)
    # prior turns [{"role": "user"|"assistant", "content": str}] so a follow-up can be rewritten
    history: list[dict] | None = Field(default=None, max_length=20)


class FeedbackRequest(BaseModel):
    message_id: str = Field(max_length=64)
    verdict: str  # "up" or "down"
    note: str | None = Field(default=None, max_length=2000)


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=72)  # bcrypt only uses the first 72 bytes
    turnstile_token: str | None = Field(default=None, max_length=4096)


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def create_app(rate_limit: str | None = None, auth_db_path: str | None = None,
               chat_brain: str | None = None) -> FastAPI:
    app = FastAPI(title="Skein Lite API")
    settings = get_settings()
    brain = chat_brain or settings.chat_brain  # "linear" streams; "agent" runs the M6 brain
    limiter = RateLimiter(rate_limit or settings.rate_limit)
    origins = [o.strip() for o in
               os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["*"])

    login_limiter = RateLimiter("5/minute")  # tighter bucket for the credential endpoint
    store = UserStore(auth_db_path or settings.auth_db_path)
    seed_demo_user(store, settings.demo_username, settings.demo_password)

    if settings.jwt_secret == _INSECURE_JWT_SECRET:
        _log.warning("JWT_SECRET is the insecure default; tokens are forgeable. Set JWT_SECRET.")
    if not settings.turnstile_secret:
        _log.warning("TURNSTILE_SECRET_KEY is empty; the login captcha is bypassed (dev only).")

    def client_key(request: Request) -> str:
        # request.client.host is the direct peer; behind a proxy (Cloud Run, M9.3) this is the
        # proxy hop, so trusted X-Forwarded-For parsing must be added at deploy time.
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

    @app.get("/health")
    def health():
        return {"status": "ok"}

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
                    # the full M6 brain (supervisor, gate, escalation to the review queue) as a
                    # buffered response over the same SSE contract
                    result = answer_with_agent(
                        req.query, components=comp, history=req.history or [],
                        message_id=message_id, review_queue=comp.get("review_queue"),
                        domain=comp.get("domain"))
                    yield _sse({"type": "token", "text": result.answer})
                    yield _sse({"type": "final", "message_id": message_id,
                                "answer": result.answer, "tier": result.tier,
                                "confidence": round(result.confidence, 3),
                                "grounding": round(result.grounding, 3),
                                "citations": result.citations,
                                "escalation_id": result.trace.get("escalation_id")})
                    return
                for event in stream_answer(req.query, message_id=message_id,
                                           embedder=comp["embedder"], store=comp["store"],
                                           llm=comp["llm"], reranker=comp["reranker"],
                                           metric_resolver=comp.get("metric_resolver"),
                                           graph_retriever=comp.get("graph_retriever")):
                    yield _sse(event)
            # Catch broadly: the response is already a 200 SSE stream, so any failure (a hosted
            # SDK error not wrapped as RuntimeError, a mid-stream drop) must surface as an event,
            # never a silent dead stream. is_transient decides degraded vs honest error.
            except Exception as exc:
                latency = round((time.perf_counter() - started) * 1000, 1)
                transient = is_transient(exc)
                write_trace({"ts": time.time(), "message_id": message_id, "query": req.query,
                             "tier": "degraded" if transient else "error", "streamed": True,
                             "error": str(exc)[:200], "latency_ms": latency}, DEFAULT_TRACE_PATH)
                if transient:
                    yield _sse({"type": "token", "text": _DEGRADED})
                    yield _sse({"type": "final", "message_id": message_id, "tier": "degraded",
                                "answer": _DEGRADED, "confidence": 0.0, "grounding": 0.0,
                                "citations": []})
                else:
                    yield _sse({"type": "error", "message_id": message_id,
                                "message": "internal error"})

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers=_SSE_HEADERS)

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

    return app


app = create_app()
