"""M1.2 Voyage embedder: HTTP mocked, so this asserts the real request shape offline."""
import pytest

import adapters.voyage as voyage_mod
from adapters.voyage import VoyageEmbedder


def test_embed_sends_model_and_input_type_and_parses(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload, headers=headers)
        return {"data": [{"embedding": [float(i)], "index": i}
                         for i, _ in enumerate(payload["input"])]}

    monkeypatch.setattr(voyage_mod, "request_json", fake)
    out = VoyageEmbedder(model="voyage-3-large", api_key="k").embed(["a", "b", "c"],
                                                                    input_type="query")
    assert out == [[0.0], [1.0], [2.0]]
    assert seen["url"].endswith("/embeddings")
    assert seen["payload"]["model"] == "voyage-3-large"
    assert seen["payload"]["input_type"] == "query"
    assert seen["headers"]["Authorization"] == "Bearer k"


def test_embed_batches_and_sorts_by_index(monkeypatch):
    calls = {"n": 0}

    def fake(method, url, payload=None, headers=None, timeout=60):
        calls["n"] += 1
        rows = [{"embedding": [float(i)], "index": i} for i, _ in enumerate(payload["input"])]
        return {"data": list(reversed(rows))}  # reversed, to prove we sort by index

    monkeypatch.setattr(voyage_mod, "request_json", fake)
    out = VoyageEmbedder(model="voyage-3-large", api_key="k").embed([str(i) for i in range(200)])
    assert calls["n"] == 2  # 128 + 72
    assert out[0] == [0.0]
    assert out[127] == [127.0]
    assert out[128] == [0.0]


def test_missing_key_raises():
    embedder = VoyageEmbedder(model="voyage-3-large", api_key="placeholder")
    embedder.api_key = ""  # force empty regardless of any .env on the machine
    with pytest.raises(RuntimeError):
        embedder.embed(["a"])
