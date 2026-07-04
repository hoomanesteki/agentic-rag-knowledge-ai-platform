"""Cohere embeddings and reranker behind the Embedder/Reranker interfaces. One account and key
cover both. Uses the v2 API. Kept dependency-free via the shared HTTP helper.

Two-key layering: COHERE_API_KEY is tried first (a free Trial key is fine as the first layer), and
on a 429 (the trial monthly cap or a per-minute rate limit) the request retries once on
COHERE_API_KEY_FALLBACK if set (a paid Production key). If both fail, the retrieval layer degrades
to local sparse search, so the answer still lands."""
from __future__ import annotations

import logging

from ._http import request_json
from .config import get_settings

_log = logging.getLogger("skein.cohere")
_EMBED_API = "https://api.cohere.com/v2/embed"
_RERANK_API = "https://api.cohere.com/v2/rerank"


def _post(url: str, body: dict, api_key: str, fallback_key: str) -> dict:
    """POST to Cohere on the primary key; on a 429, retry once on the fallback key if configured."""
    try:
        return request_json("POST", url, body, {"Authorization": "Bearer " + api_key})
    except RuntimeError as exc:
        if fallback_key and fallback_key != api_key and "HTTP 429" in str(exc):
            _log.warning("Cohere primary key hit 429, retrying on the fallback key")
            return request_json("POST", url, body, {"Authorization": "Bearer " + fallback_key})
        raise

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
        self.fallback_key = settings.cohere_api_key_fallback
        if self.model in _MODEL_DIMS:
            self._dim = _MODEL_DIMS[self.model]
        elif "v4" in self.model:
            self._dim = 1536  # v4 lets us pin the size, and we request exactly this below
        else:
            # guessing a dim risks a silent Qdrant mismatch later; fail loudly at construction
            raise ValueError(
                "unknown Cohere embed model {!r}; add it to _MODEL_DIMS".format(self.model))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is not set; put it in .env")
        if input_type not in _INPUT_TYPES:  # a query embedded as a document silently hurts recall
            raise ValueError(
                "input_type must be 'query' or 'document', got {!r}".format(input_type))
        cohere_type = _INPUT_TYPES[input_type]
        texts = list(texts)
        out: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start:start + _EMBED_BATCH]
            body = {"model": self.model, "texts": batch, "input_type": cohere_type,
                    "embedding_types": ["float"]}
            if "v4" in self.model:  # only v4 accepts a chosen output size
                body["output_dimension"] = self._dim
            resp = _post(_EMBED_API, body, self.api_key, self.fallback_key)
            out.extend(resp["embeddings"]["float"])  # v2 returns vectors in input order
        return out


class CohereReranker:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.rerank_model
        self.api_key = api_key or settings.cohere_api_key
        self.fallback_key = settings.cohere_api_key_fallback

    def rerank(self, query: str, documents: list[str],
               top_n: int = 8) -> list[tuple[int, float]]:
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is not set; put it in .env")
        if not documents:
            return []
        documents = list(documents)[:_RERANK_MAX_DOCS]
        body = {"model": self.model, "query": query, "documents": documents, "top_n": top_n}
        resp = _post(_RERANK_API, body, self.api_key, self.fallback_key)
        # Sort by score rather than trusting array order, mirroring the other adapters.
        rows = sorted(resp["results"], key=lambda r: r["relevance_score"], reverse=True)
        return [(row["index"], row["relevance_score"]) for row in rows]
