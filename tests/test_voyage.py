"""M1.2 Voyage embedder: the HTTP is mocked so this runs offline with no key or network."""
import pytest

import adapters.voyage as voyage_mod
from adapters.voyage import VoyageEmbedder


def test_embed_parses_response(monkeypatch):
    def fake_request(method, url, payload=None, headers=None, timeout=60):
        assert method == "POST"
        assert headers and "Authorization" in headers
        return {"data": [{"embedding": [0.1, 0.2]} for _ in payload["input"]]}

    monkeypatch.setattr(voyage_mod, "request_json", fake_request)
    embedder = VoyageEmbedder(model="voyage-3-large", api_key="test-key")
    assert embedder.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]


def test_missing_key_raises():
    embedder = VoyageEmbedder(model="voyage-3-large", api_key="placeholder")
    embedder.api_key = ""  # force empty regardless of any .env on the machine
    with pytest.raises(RuntimeError):
        embedder.embed(["a"])
