"""Voyage embeddings behind the Embedder interface (hosted, dense, multilingual)."""
from __future__ import annotations

import os

from ._http import request_json
from .config import get_settings

_API = "https://api.voyageai.com/v1/embeddings"
_MODEL_DIMS = {"voyage-3-large": 1024, "voyage-3": 1024, "voyage-3.5": 1024}


class VoyageEmbedder:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.model = model or os.getenv("EMBED_MODEL", "voyage-3-large")
        self.api_key = api_key or settings.voyage_api_key
        self._dim = _MODEL_DIMS.get(self.model, 1024)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set; put it in .env")
        body = {"model": self.model, "input": list(texts), "input_type": input_type}
        resp = request_json("POST", _API, body,
                            {"Authorization": "Bearer " + self.api_key})
        return [row["embedding"] for row in resp["data"]]
