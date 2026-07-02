"""Tiny JSON-over-HTTP helper using the standard library, so the adapters need no extra
dependency. Swap for httpx or a vendor SDK later without touching callers."""
from __future__ import annotations

import json
import urllib.request


def request_json(method: str, url: str, payload: dict | None = None,
                 headers: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted URLs)
        body = resp.read().decode()
    return json.loads(body) if body else {}
