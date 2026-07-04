"""Tiny JSON-over-HTTP helper using the standard library, so the adapters need no extra
dependency. Swap for httpx or a vendor SDK later without touching callers.

On failure it raises RuntimeError with the vendor's response body included, because Voyage
and Qdrant put the real reason (bad model, invalid filter, dim mismatch, quota) in the body.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

# Send a real User-Agent. Groq sits behind Cloudflare, which blocks the default "Python-urllib/x.y"
# signature with a 403 (error 1010). Any honest UA gets through, so name the app.
_USER_AGENT = "skein-lite/1.0 (+https://github.com/hoomanesteki/agentic-rag-knowledge-ai-platform)"


def request_json(method: str, url: str, payload: dict | None = None,
                 headers: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", _USER_AGENT)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError("{} {} -> HTTP {}: {}".format(method, url, exc.code, detail)) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        # A read/connect timeout raises socket.timeout (TimeoutError), not URLError, so catch both
        # and normalize to the shared "failed:" RuntimeError. Otherwise a timeout escaped as a bare
        # TimeoutError, which is_transient() did not classify as transient, so the retry wrapper and
        # the reranker fallback were both bypassed and a single slow response lost the turn.
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(
            "{} {} failed: {} (is the service running?)".format(method, url, reason)) from exc
    return json.loads(body) if body else {}
