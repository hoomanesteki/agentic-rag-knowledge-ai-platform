"""M5.3 graph retriever: resolve entities named in the query (or in retrieved text) to graph
nodes, then attach their neighborhood as a labeled evidence block.

This is the graph-first path for relational questions ("which supplier makes X") and a
vector-first enrichment for the rest (the retrieved review names a product, so its supplier and
stores come along). It uses only the allowlisted GraphStore traversals (find_nodes, neighbors),
never free-form Cypher, so it is injection-safe. The node name index is built once at
construction because the graph is static after a load; resolution is then in memory.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter

from adapters.base import GraphNode, GraphStore
from data.lakehouse import load_manifest

_log = logging.getLogger("skein.graph_retriever")
_WORD = re.compile(r"[a-z0-9]+")
_MAX_ENTITIES = 5
_MAX_NEIGHBORS = 25
_MAX_INDEX_NODES = 100000  # per label; above this the name index would silently truncate


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _name_tokens(name: str) -> set[str]:
    return {t for t in _tokens(name) if len(t) >= 2}


class GraphRetriever:
    def __init__(self, graph: GraphStore, node_specs: list[dict], name_prop: str = "name",
                 max_entities: int = _MAX_ENTITIES, max_neighbors: int = _MAX_NEIGHBORS) -> None:
        self.graph = graph
        self.name_prop = name_prop
        self.max_entities = max_entities
        self.max_neighbors = max_neighbors

        nodes: list[GraphNode] = []
        for spec in node_specs:
            label_nodes = graph.find_nodes(spec["label"], limit=_MAX_INDEX_NODES + 1)
            if len(label_nodes) > _MAX_INDEX_NODES:
                _log.warning("%s has over %d nodes; the name index keeps the first %d",
                             spec["label"], _MAX_INDEX_NODES, _MAX_INDEX_NODES)
                label_nodes = label_nodes[:_MAX_INDEX_NODES]
            nodes.extend(label_nodes)

        # Dedupe by (label, name): variant-grain rows (a product name repeated per size) share a
        # name, so one representative keeps token frequency honest and avoids crowding out other
        # entities at resolution time.
        reps: dict[tuple[str, str], GraphNode] = {}
        for node in nodes:
            name = str(node.properties.get(name_prop, ""))
            if name:
                reps.setdefault((node.label, name.lower()), node)
        representatives = list(reps.values())
        common = self._common_tokens(representatives)
        # only nodes with a distinctive name are resolvable (tickets have no name, so they are
        # reached by hopping, not by naming them in a query)
        self._index: list[tuple[GraphNode, set[str]]] = []
        for node in representatives:
            distinctive = _name_tokens(str(node.properties.get(name_prop, ""))) - common
            if distinctive:
                self._index.append((node, distinctive))

    def _common_tokens(self, nodes: list[GraphNode]) -> set[str]:
        """Tokens shared by many names (a brand word). An absolute floor keeps a token shared by
        just two nodes from being treated as common in a small catalog."""
        df: Counter = Counter()
        for node in nodes:
            for token in _name_tokens(str(node.properties.get(self.name_prop, ""))):
                df[token] += 1
        cutoff = max(4, len(nodes) // 3)
        return {token for token, n in df.items() if n >= cutoff}

    def resolve(self, text: str) -> list[GraphNode]:
        """Nodes whose full distinctive name appears in the text (high precision). The most
        specific match (largest distinctive name) wins when more than max_entities resolve."""
        query_tokens = _tokens(text)
        matched = [(node, distinctive) for node, distinctive in self._index
                   if distinctive <= query_tokens]
        matched.sort(key=lambda pair: len(pair[1]), reverse=True)
        return [node for node, _ in matched[:self.max_entities]]

    def _render(self, node: GraphNode) -> str:
        name = node.properties.get(self.name_prop, node.id)
        if not node.key:  # a node with no key property cannot be traversed; render it bare
            return "{} ({}).".format(name, node.label)
        neighbors = self.graph.neighbors(node.label, node.key, node.id, limit=self.max_neighbors)
        props = ", ".join(
            "{}={}".format(k, v) for k, v in node.properties.items()
            if k not in (node.key, self.name_prop))
        head = "{} ({}{})".format(name, node.label, "; " + props if props else "")
        rels: dict[str, list[str]] = {}
        for nb in neighbors:
            nb_name = nb.node.properties.get(self.name_prop, nb.node.id)
            rels.setdefault(nb.edge_type, []).append("{} ({})".format(nb_name, nb.node.label))
        rel_str = "; ".join("{} {}".format(etype, ", ".join(vals)) for etype, vals in rels.items())
        return head + (" -> " + rel_str if rel_str else "") + "."

    def evidence(self, query: str, extra_texts: tuple = ()) -> tuple[dict | None, bool]:
        """Returns (block, from_query). block is a single graph evidence block or None if nothing
        resolved. from_query is True only when an entity named in the query itself resolved: that
        is authoritative relational grounding. Entities pulled from retrieved text (the
        vector-first hop) enrich the block but do not, on their own, make it authoritative."""
        resolved = self.resolve(query)
        from_query = bool(resolved)
        seen = {(n.label, n.id) for n in resolved}
        for text in extra_texts:
            if len(resolved) >= self.max_entities:
                break
            for node in self.resolve(text):
                if (node.label, node.id) not in seen:
                    seen.add((node.label, node.id))
                    resolved.append(node)
                    if len(resolved) >= self.max_entities:
                        break
        if not resolved:
            return None, False
        body = " ".join(self._render(node) for node in resolved)
        ids = ",".join("{}:{}".format(n.label, n.id) for n in resolved)
        block = {"id": "graph:" + ids, "text": "Knowledge graph facts. " + body,
                 "source": "graph", "doc_type": "graph", "score": 1.0}
        return block, from_query


def make_graph_retriever(domain: str, graph: GraphStore,
                         domains_dir: str = "domains") -> GraphRetriever | None:
    """Build a retriever from the pack's graph node specs, or None if the pack declares no graph.
    The name property to resolve against is the manifest's graph.name_property (default 'name'),
    so a domain whose entities carry a 'title' still resolves."""
    graph_spec = load_manifest(os.path.join(domains_dir, domain)).get("graph", {}) or {}
    specs = graph_spec.get("nodes", []) or []
    if not specs:
        return None
    return GraphRetriever(graph, specs, name_prop=graph_spec.get("name_property", "name"))
