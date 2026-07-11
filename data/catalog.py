"""The storefront catalog, read from the governed gold `products` table plus the pack's seed copy.

Extracted from api/app.py so both the web API and the MCP server (mcp_server/server.py) read one
implementation, with no duplicated catalog logic and no second copy of the DuckDB access rules.
Every read opens DuckDB read-only with external access disabled, the same posture as the metric
layer. Returns empty/None rather than raising when the lakehouse is not built yet, so a caller
degrades instead of crashing.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from data.lakehouse import load_manifest


@lru_cache
def catalog(domain: str) -> list:
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
def product(domain: str, pid: str) -> dict | None:
    """One product's full page: the catalog card plus its marketing copy and any reviews."""
    prod = next((p for p in catalog(domain) if p["id"] == pid), None)
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
def brand(domain: str) -> str:
    """The active domain's display brand, read from its manifest, so the storefront names itself
    from the pack instead of the engine hardcoding it."""
    return str(load_manifest(os.path.join("domains", domain)).get("brand", "") or "")[:80]
