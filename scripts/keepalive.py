"""Keep the free-tier hosted services warm so they do not idle out between demos.

Neo4j Aura Free pauses after a few days and is deleted around 30; Supabase free pauses after about
a week; Qdrant Cloud free clusters also suspend when idle. This job touches each configured target
with the lightest possible real request and reports a summary. It runs on a schedule from
.github/workflows/keepalive.yml, and is safe to run by hand: `uv run python -m scripts.keepalive`.

Only targets whose env vars are set are checked, so the same script works for any subset of the
hosted stack. It exits non-zero if any checked target fails, so a paused free tier is visible.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable

import httpx

Check = Callable[[], "tuple[bool, str]"]


def _get(url: str, headers: dict | None = None) -> tuple[bool, str]:
    try:
        resp = httpx.get(url, headers=headers, timeout=20.0)
        return resp.status_code < 500, "HTTP {}".format(resp.status_code)
    except httpx.HTTPError as exc:
        return False, str(exc)[:120]


def _touch_api_health(base: str) -> tuple[bool, str]:
    return _get(base.rstrip("/") + "/health")


def _touch_postgres_via_login(base: str, username: str, password: str) -> tuple[bool, str]:
    # A login runs a SELECT on the users table, which lives in Supabase Postgres on deploy. Wrong
    # credentials are fine: the query still executes and keeps the database warm. A 401/403/429 all
    # mean the endpoint answered, so the database was reached.
    try:
        resp = httpx.post(base.rstrip("/") + "/api/login",
                          json={"username": username, "password": password}, timeout=20.0)
        return resp.status_code < 500, "HTTP {}".format(resp.status_code)
    except httpx.HTTPError as exc:
        return False, str(exc)[:120]


def _touch_qdrant(url: str, api_key: str | None) -> tuple[bool, str]:
    headers = {"api-key": api_key} if api_key else None
    return _get(url.rstrip("/") + "/healthz", headers)


def _touch_neo4j() -> tuple[bool, str]:
    # Reuse the real adapter so this exercises the same HTTP Cypher path the app uses.
    from adapters.neo4j_store import Neo4jStore
    try:
        Neo4jStore()._run("RETURN 1 AS ok")
        return True, "RETURN 1 ok"
    except Exception as exc:  # noqa: BLE001 - any backend failure is a failed keepalive
        return False, str(exc)[:120]


def plan(env: dict) -> list[tuple[str, Check]]:
    """Build the list of (name, check) targets from the environment. Pure, so it is testable."""
    targets: list[tuple[str, Check]] = []
    api = env.get("KEEPALIVE_API_URL")
    if api:
        targets.append(("api-health", lambda: _touch_api_health(api)))
        targets.append(("supabase-postgres", lambda: _touch_postgres_via_login(
            api, env.get("DEMO_USERNAME", "demo"), "keepalive-not-a-real-password")))
    qdrant = env.get("QDRANT_URL")
    if qdrant and not qdrant.startswith("http://localhost"):
        targets.append(("qdrant", lambda: _touch_qdrant(qdrant, env.get("QDRANT_API_KEY"))))
    if env.get("GRAPH_PROVIDER") == "neo4j" and env.get("NEO4J_URL", "").startswith("http"):
        if not env.get("NEO4J_URL", "").startswith("http://localhost"):
            targets.append(("neo4j", _touch_neo4j))
    return targets


def main() -> int:
    # Load .env for local runs; on the CI schedule the values come from repo secrets instead.
    from dotenv import load_dotenv
    load_dotenv()
    targets = plan(dict(os.environ))
    if not targets:
        print("keepalive: no hosted targets configured (set KEEPALIVE_API_URL, QDRANT_URL, ...)")
        return 0
    failures = 0
    for name, check in targets:
        ok, detail = check()
        print("keepalive: {:<20} {} ({})".format(name, "ok" if ok else "FAILED", detail))
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
