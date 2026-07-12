"""Skein Lite MCP server: a small, read-only Model Context Protocol surface over the same gated
pipeline the web app uses.

The design principle is "thin second client": every tool delegates to the exact code the FastAPI
routes already call (rag.omni.stream_omni, data.metrics.MetricResolver, data.catalog), so the
deterministic safety gates (order-PII, prompt-injection, customer-enumeration, per-citation
grounding) all run below this surface with zero MCP-side re-implementation. Nothing here re-derives
a gate or trusts the client.

Safe by construction:
  - Read-only only. There is no write, admin, login, TTS, or order-lookup tool.
  - skein_ask runs anonymous AND passes block_order_pii=True, so the pipeline drops every order
    and account document before generation. No order or account data is disclosed over MCP even if
    the query supplies a real name and email, which makes this surface strictly safer than the
    signed-out web flow (where a matching name+email unlocks orders). No tool reaches private data.
  - Tool results are truncated so a large table cannot flood a client's context.
  - Nothing writes to stdout: over stdio, stdout is the JSON-RPC channel, and a stray print would
    corrupt the framing. All diagnostics go through logging (stderr).

Run (stdio, what Claude Desktop / Claude Code / an IDE launch):
    make mcp                         # or: uv run python -m mcp_server.server
Scale-up transport (documented, not on by default):
    uv run python -m mcp_server.server --transport streamable-http
"""
from __future__ import annotations

import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

_log = logging.getLogger("skein.mcp")

mcp = FastMCP("skein-lite")

# Components (embedder, store, llm, metric resolver, ...) are built once, lazily, from the same
# adapters.factory the app uses. Lazy so importing this module (in tests, or to list tools) is cheap
# and offline-safe; factory.py returns fakes when no API keys are present.
_COMPONENTS: dict | None = None


def _components() -> dict:
    global _COMPONENTS
    if _COMPONENTS is None:
        from api.deps import get_components
        _COMPONENTS = get_components()
    return _COMPONENTS


def _domain() -> str:
    return _components()["domain"]


@mcp.tool()
def skein_ask(query: str) -> dict:
    """Ask the store's shopping assistant a shopping or store question and get a short, grounded,
    cited answer drawn from the store's own data.

    It CANNOT access any shopper's orders, account, or personal data: the call runs anonymous and
    with order/account disclosure blocked, so every order and account document is dropped before
    the model sees it, even if the question includes a real name and email. Order, account, and
    "who bought X" questions therefore return only generic help. Good for product, sizing,
    materials, care, shipping, and returns questions. Returns {answer, citations, grounding, tier}.
    """
    try:
        from rag.omni import stream_omni
        c = _components()
        final: dict = {}
        for event in stream_omni(
            query,
            embedder=c["embedder"], store=c["store"], llm=c["llm"],
            reranker=c.get("reranker"), metric_resolver=c.get("metric_resolver"),
            graph_retriever=c.get("graph_retriever"), answer_cache=c.get("answer_cache"),
            review_queue=c.get("review_queue"), domain=c["domain"],
            auth_identity=None, concise=True, block_order_pii=True,
        ):
            if event.get("type") == "final":
                final = event
        return {
            "answer": final.get("answer", ""),
            "citations": (final.get("citations") or [])[:8],
            "grounding": final.get("grounding"),
            "tier": final.get("tier"),
        }
    except Exception as exc:  # noqa: BLE001 - a tool must return a clean error, never crash the client
        _log.warning("skein_ask failed: %s", exc)
        return {"error": "skein_ask failed: {}".format(str(exc)[:200])}


@mcp.tool()
def skein_get_metric(name: str, params: dict | None = None) -> dict:
    """Run one governed, read-only business metric by name and return its rows.

    The name is validated against the domain's metrics.yaml; only a single read-only SELECT over
    the curated gold tables can run (no free-form SQL, no raw PII layers). Call skein_list_metrics
    to see the allowed names and their parameters. Returns {name, params, columns, rows, summary}.
    """
    try:
        res = _components()["metric_resolver"].resolve(name, params or {})
        return {
            "name": res.name, "params": res.params, "columns": res.columns,
            "rows": res.rows[:50], "summary": res.summary(),
        }
    except Exception as exc:  # noqa: BLE001 - unknown metric / bad param returns a clean error
        _log.warning("skein_get_metric(%s) failed: %s", name, exc)
        return {"error": str(exc)[:200], "allowed": _components()["metric_resolver"].names()}


@mcp.tool()
def skein_list_metrics() -> dict:
    """List the governed metrics available to skein_get_metric: name, grain, source, and parameters.
    Metadata only; a metric's SQL template is never exposed."""
    try:
        from data.introspect import metrics_view
        return {"metrics": metrics_view(_domain())}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


@mcp.tool()
def skein_search_products(
    query: str = "", gender: str = "", max_price: float | None = None,
) -> dict:
    """Search the storefront catalog. Optional filters: a free-text query (matched against name,
    category, colour, and weather), a gender ('women' | 'men' | 'unisex'), and a maximum price.
    Reads the same governed gold catalog the website renders. Returns up to 20 product cards."""
    try:
        from data.catalog import catalog
        q = (query or "").lower().strip()
        want_gender = (gender or "").lower().strip()

        def _match(p: dict) -> bool:
            if q:
                hay = " ".join(str(x) for x in (
                    p.get("name"), p.get("category"), p.get("color"), p.get("weather"),
                    " ".join(p.get("colors") or []))).lower()
                if q not in hay:
                    return False
            if want_gender and (p.get("gender") or "").lower() not in (want_gender, "unisex", ""):
                return False
            if max_price is not None and p.get("price") is not None and p["price"] > max_price:
                return False
            return True

        hits = [{
            "id": p["id"], "name": p["name"], "category": p["category"], "gender": p["gender"],
            "price": p["price"], "colors": p["colors"], "sizes": p["sizes"],
        } for p in catalog(_domain()) if _match(p)]
        return {"count": len(hits), "products": hits[:20]}
    except Exception as exc:  # noqa: BLE001
        _log.warning("skein_search_products failed: %s", exc)
        return {"error": str(exc)[:200]}


@mcp.tool()
def skein_get_product_facts(product_id: str) -> dict:
    """Full facts for one product by id: price, sizes, colours, description, and a few honest
    reviews (a positive and a critical one where available). Returns an error field if unknown."""
    try:
        from data.catalog import product
        p = product(_domain(), product_id)
        if not p:
            return {"error": "no such product: {}".format(product_id)}
        return {
            "id": p["id"], "name": p["name"], "price": p["price"], "sizes": p["sizes"],
            "colors": p["colors"], "gender": p["gender"], "category": p["category"],
            "description": p.get("description", ""), "reviews": (p.get("reviews") or [])[:6],
        }
    except Exception as exc:  # noqa: BLE001
        _log.warning("skein_get_product_facts failed: %s", exc)
        return {"error": str(exc)[:200]}


@mcp.resource("skein://metrics")
def metrics_resource() -> str:
    """The governed metric catalog as JSON (metadata only, no SQL)."""
    from data.introspect import metrics_view
    return json.dumps(metrics_view(_domain()), indent=2)


@mcp.resource("skein://ontology")
def ontology_resource() -> str:
    """The knowledge-graph ontology: entity labels and the typed edges between them, as JSON."""
    from data.introspect import ontology_view
    return json.dumps(ontology_view(_domain()), indent=2)


@mcp.prompt()
def shop_for(occasion: str, budget: str = "") -> str:
    """A ready-to-send styling request for the assistant."""
    tail = " with a budget around {}".format(budget) if budget else ""
    return ("I'm shopping for {}{}. What do you recommend from the catalog, and why? "
            "Keep it to two or three picks.".format(occasion, tail))


def main() -> None:
    # stdio by default (what local MCP clients launch); a single --transport flag selects the
    # documented HTTP scale-up. No other CLI surface, matching the reference servers.
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
