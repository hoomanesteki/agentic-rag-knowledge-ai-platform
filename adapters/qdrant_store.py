"""Qdrant hybrid store: dense (Voyage) plus sparse (BM25-style TF) named vectors, fused with
RRF. IDF is applied server-side (sparse modifier "idf"), so the client encoder is stateless.

Uses the Qdrant REST API over stdlib HTTP. Point ids are derived from the chunk id so
re-ingest overwrites in place and stays idempotent.
"""
from __future__ import annotations

import uuid

from ._http import request_json
from .config import get_settings

_NS = uuid.NAMESPACE_URL


def point_id(chunk_id: str) -> str:
    """Stable UUID for a chunk id, so upserting the same chunk overwrites it."""
    return str(uuid.uuid5(_NS, chunk_id))


def _to_filter(where: dict) -> dict:
    """Turn a flat {field: value} dict into a Qdrant must/match filter."""
    return {"must": [{"key": k, "match": {"value": v}} for k, v in where.items()]}


class QdrantStore:
    def __init__(self, collection: str, url: str | None = None,
                 api_key: str | None = None) -> None:
        self.collection = collection
        settings = get_settings()
        self.url = (url or settings.qdrant_url).rstrip("/")
        # Qdrant Cloud rejects unauthenticated requests; the header is harmless against a local
        # instance that ignores it. Empty key -> no header, so local dev keeps working.
        key = settings.qdrant_api_key if api_key is None else api_key
        self._headers = {"api-key": key} if key else None

    def _exists(self) -> bool:
        resp = request_json("GET", "{}/collections/{}/exists".format(self.url, self.collection),
                            headers=self._headers)
        return bool(resp.get("result", {}).get("exists"))

    def ensure_collection(self, dense_dim: int) -> None:
        if self._exists():
            return
        request_json("PUT", "{}/collections/{}".format(self.url, self.collection), {
            "vectors": {"dense": {"size": dense_dim, "distance": "Cosine"}},
            "sparse_vectors": {"sparse": {"modifier": "idf"}},
        }, headers=self._headers)

    def upsert(self, points: list[dict]) -> None:
        """Each point: {id, text, payload, dense: [...], sparse: {indices, values}}."""
        body = {"points": [
            {
                "id": point_id(p["id"]),
                "vector": {
                    "dense": p["dense"],
                    "sparse": {"indices": p["sparse"]["indices"],
                               "values": p["sparse"]["values"]},
                },
                "payload": {**p["payload"], "chunk_id": p["id"], "text": p["text"]},
            }
            for p in points
        ]}
        request_json("PUT",
                     "{}/collections/{}/points?wait=true".format(self.url, self.collection),
                     body, headers=self._headers)

    def hybrid_search(self, dense_query: list[float], sparse_query: dict,
                      top_k: int = 8, where: dict | None = None,
                      dense_only: bool = False) -> list[dict]:
        flt = _to_filter(where) if where else None
        url = "{}/collections/{}/points/query".format(self.url, self.collection)
        if dense_only:
            body = {"query": dense_query, "using": "dense", "limit": top_k, "with_payload": True}
            if flt:
                body["filter"] = flt
            return request_json("POST", url, body, headers=self._headers).get(
                "result", {}).get("points", [])

        prefetch_limit = top_k * 4
        prefetch = [
            {"query": dense_query, "using": "dense", "limit": prefetch_limit},
            {"query": {"indices": sparse_query["indices"], "values": sparse_query["values"]},
             "using": "sparse", "limit": prefetch_limit},
        ]
        if flt:
            for branch in prefetch:
                branch["filter"] = flt
        body = {"prefetch": prefetch, "query": {"fusion": "rrf"},
                "limit": top_k, "with_payload": True}
        return request_json("POST", url, body, headers=self._headers).get(
            "result", {}).get("points", [])
