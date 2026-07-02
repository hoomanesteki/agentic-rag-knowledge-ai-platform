"""Voyage reranker behind the Reranker interface (hosted cross-encoder)."""
from __future__ import annotations

from ._http import request_json
from .config import get_settings

_API = "https://api.voyageai.com/v1/rerank"


class VoyageReranker:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.rerank_model
        self.api_key = api_key or settings.voyage_api_key

    def rerank(self, query: str, documents: list[str],
               top_n: int = 8) -> list[tuple[int, float]]:
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set; put it in .env")
        if not documents:
            return []
        documents = list(documents)[:1000]  # Voyage caps documents per rerank request
        body = {"query": query, "documents": documents, "model": self.model,
                "top_k": top_n, "truncation": True}
        resp = request_json("POST", _API, body,
                            {"Authorization": "Bearer " + self.api_key})
        # Do not trust array order; sort by score like the embedder does with its index.
        rows = sorted(resp["data"], key=lambda r: r["relevance_score"], reverse=True)
        return [(row["index"], row["relevance_score"]) for row in rows]
