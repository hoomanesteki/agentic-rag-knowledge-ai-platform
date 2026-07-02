"""Neo4j knowledge-graph store over the HTTP transaction endpoint.

Uses Neo4j's HTTP Cypher API (POST /db/<db>/tx/commit), so it needs no bolt driver, just the
stdlib JSON-over-HTTP helper the other adapters use. Values are always passed as Cypher
parameters; only labels, edge types, and key names are ever placed in the query string, and
each is validated against a strict identifier allowlist first, so the model (or a pack) can
never inject Cypher. This is the runtime twin of the in-memory fake: same GraphStore contract.

Every node is stamped with a private `_key` property naming its primary-key column, so a fresh
store (for example the retriever's) can report a neighbor's id without knowing the schema.
"""
from __future__ import annotations

import base64
import re

from ._http import request_json
from .base import GraphNeighbor, GraphNode
from .config import get_settings

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEY_PROP = "_key"
_DOMAIN_PROP = "_domain"
_ARROWS = {"out": "-[r]->", "in": "<-[r]-", "both": "-[r]-"}


def _ident(name: str, kind: str) -> str:
    if not name or not _IDENT.match(name):
        raise ValueError("unsafe {} identifier: {!r}".format(kind, name))
    return name


class Neo4jGraphStore:
    def __init__(self, url: str | None = None, user: str | None = None,
                 password: str | None = None, database: str = "neo4j",
                 domain: str | None = None) -> None:
        s = get_settings()
        self.url = (url or s.neo4j_url).rstrip("/")
        self.database = database
        # scope every write and the reset to this domain, so one shared Neo4j can hold several
        # domains without a load wiping the others
        self.domain = domain or s.domain
        self.endpoint = "{}/db/{}/tx/commit".format(self.url, self.database)
        token = base64.b64encode(
            "{}:{}".format(user or s.neo4j_user, password or s.neo4j_password).encode()).decode()
        self.headers = {"Authorization": "Basic " + token, "Accept": "application/json"}

    def _run(self, statement: str, parameters: dict | None = None) -> list[dict]:
        body = request_json(
            "POST", self.endpoint,
            {"statements": [{"statement": statement, "parameters": parameters or {}}]},
            headers=self.headers)
        errors = body.get("errors") or []
        if errors:
            msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise RuntimeError("neo4j error: " + msg)
        results = body.get("results") or [{}]
        return results[0].get("data", [])

    def reset(self) -> None:
        # only this domain's nodes, so a shared graph keeps other domains intact
        self._run("MATCH (n {`_domain`: $d}) DETACH DELETE n", {"d": self.domain})

    def apply_schema(self, statements: list[str]) -> None:
        for stmt in statements:
            stmt = stmt.strip().rstrip(";")
            if stmt:
                self._run(stmt)

    def upsert_nodes(self, label: str, key: str, rows: list[dict]) -> int:
        _ident(label, "label")
        _ident(key, "key")
        if not rows:
            return 0
        stmt = ("UNWIND $rows AS row "
                "MERGE (n:`{L}` {{`{K}`: row.`{K}`}}) "
                "SET n += row SET n.`{P}` = $key SET n.`{D}` = $domain").format(
                    L=label, K=key, P=_KEY_PROP, D=_DOMAIN_PROP)
        self._run(stmt, {"rows": rows, "key": key, "domain": self.domain})
        return len(rows)

    def upsert_edges(self, edge_type: str, from_label: str, from_key: str,
                     to_label: str, to_key: str, pairs: list[tuple[str, str]]) -> int:
        _ident(edge_type, "edge_type")
        _ident(from_label, "label")
        _ident(from_key, "key")
        _ident(to_label, "label")
        _ident(to_key, "key")
        if not pairs:
            return 0
        rows = [{"f": str(f), "t": str(t)} for f, t in pairs]
        # RETURN the real count: a pair whose endpoints do not both exist matches nothing and
        # creates no edge, so len(rows) would overstate what was loaded.
        stmt = ("UNWIND $rows AS row "
                "MATCH (a:`{FL}` {{`{FK}`: row.f}}) "
                "MATCH (b:`{TL}` {{`{TK}`: row.t}}) "
                "MERGE (a)-[:`{ET}`]->(b) "
                "RETURN count(*) AS n").format(
                    FL=from_label, FK=from_key, TL=to_label, TK=to_key, ET=edge_type)
        data = self._run(stmt, {"rows": rows})
        return data[0]["row"][0] if data else 0

    def get_node(self, label: str, key: str, value: str) -> GraphNode | None:
        _ident(label, "label")
        _ident(key, "key")
        stmt = "MATCH (n:`{L}` {{`{K}`: $value}}) RETURN properties(n) AS props LIMIT 1".format(
            L=label, K=key)
        data = self._run(stmt, {"value": str(value)})
        if not data:
            return None
        return _node_from_props(label, data[0]["row"][0])

    def find_nodes(self, label: str, where: dict | None = None,
                   limit: int = 1000) -> list[GraphNode]:
        _ident(label, "label")
        where = where or {}
        for prop in where:
            _ident(prop, "property")
        params = {"w{}".format(i): v for i, v in enumerate(where.values())}
        params["limit"] = limit
        clause = ""
        if where:
            conds = " AND ".join(
                "n.`{}` = $w{}".format(prop, i) for i, prop in enumerate(where))
            clause = "WHERE " + conds + " "
        # deterministic order so a truncation at $limit is stable across runs
        stmt = ("MATCH (n:`{L}`) {clause}RETURN properties(n) AS props "
                "ORDER BY elementId(n) LIMIT $limit").format(L=label, clause=clause)
        data = self._run(stmt, params)
        return [_node_from_props(label, datum["row"][0]) for datum in data]

    def neighbors(self, label: str, key: str, value: str, *, edge_type: str | None = None,
                  direction: str = "both", to_label: str | None = None,
                  limit: int = 50) -> list[GraphNeighbor]:
        if not key or not value:
            return []
        _ident(label, "label")
        _ident(key, "key")
        # edge_type and to_label stay parameters, so no identifier interpolation is needed for
        # them; direction is a fixed enum that picks the pattern, never user text.
        if direction not in _ARROWS:
            raise ValueError("unknown direction: {!r}".format(direction))
        arrow = _ARROWS[direction]
        stmt = (
            "MATCH (a:`{L}` {{`{K}`: $value}}){arrow}(b) "
            "WHERE ($etype IS NULL OR type(r) = $etype) "
            "AND ($tolabel IS NULL OR $tolabel IN labels(b)) "
            "RETURN type(r) AS etype, (startNode(r) = a) AS outgoing, "
            "labels(b) AS blabels, properties(b) AS bprops LIMIT $limit"
        ).format(L=label, K=key, arrow=arrow)
        data = self._run(stmt, {"value": str(value), "etype": edge_type,
                                "tolabel": to_label, "limit": limit})
        out = []
        for datum in data:
            etype, outgoing, blabels, bprops = datum["row"]
            blabel = blabels[0] if blabels else ""  # our loader gives each node one label
            out.append(GraphNeighbor(
                edge_type=etype, direction="out" if outgoing else "in",
                node=_node_from_props(blabel, bprops)))
        return out


def _node_from_props(label: str, props: dict) -> GraphNode:
    props = dict(props or {})
    key = props.pop(_KEY_PROP, "")
    props.pop(_DOMAIN_PROP, None)  # internal scoping property, not part of the node's data
    return GraphNode(label=label, key=key, id=str(props.get(key, "")), properties=props)
