"""Cohere embeddings and reranker behind the Embedder/Reranker interfaces. One account and key
cover both. Uses the v2 API. Kept dependency-free via the shared HTTP helper."""
from __future__ import annotations

from ._http import request_json
from .config import get_settings

_EMBED_API = "https://api.cohere.com/v2/embed"
_RERANK_API = "https://api.cohere.com/v2/rerank"

# embed-v4.0 lets you pick the output size; the v3 models are fixed. Default v4 to 1536.
_MODEL_DIMS = {
    "embed-v4.0": 1536,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}
_EMBED_BATCH = 96  # Cohere caps texts per embed call at 96
_RERANK_MAX_DOCS = 1000

# Our interface speaks "document"/"query"; Cohere wants the search-tuned variants.
_INPUT_TYPES = {"document": "search_document", "query": "search_query"}


class CohereEmbedder:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.embed_model
        self.api_key = api_key or settings.cohere_api_key
        self._dim = _MODEL_DIMS.get(self.model, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is not set; put it in .env")
        cohere_type = _INPUT_TYPES.get(input_type, "search_document")
        texts = list(texts)
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start:start + _EMBED_BATCH]
            body = {"model": self.model, "texts": batch, "input_type": cohere_type,
                    "embedding_types": ["float"]}
            if "v4" in self.model:  # only v4 accepts a chosen output size
                body["output_dimension"] = self._dim
            resp = request_json("POST", _EMBED_API, body,
                                {"Authorization": "Bearer " + self.api_key})
            out.extend(resp["embeddings"]["float"])  # v2 returns vectors in input order
        return out


class CohereReranker:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.rerank_model
        self.api_key = api_key or settings.cohere_api_key

    def rerank(self, query: str, documents: list[str],
               top_n: int = 8) -> list[tuple[int, float]]:
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is not set; put it in .env")
        if not documents:
            return []
        documents = list(documents)[:_RERANK_MAX_DOCS]
        body = {"model": self.model, "query": query, "documents": documents, "top_n": top_n}
        resp = request_json("POST", _RERANK_API, body,
                            {"Authorization": "Bearer " + self.api_key})
        # Sort by score rather than trusting array order, mirroring the other adapters.
        rows = sorted(resp["results"], key=lambda r: r["relevance_score"], reverse=True)
        return [(row["index"], row["relevance_score"]) for row in rows]
