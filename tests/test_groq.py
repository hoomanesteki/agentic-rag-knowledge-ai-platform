"""M1.3 Groq client: HTTP mocked, so this asserts the real request shape offline."""
import io
import urllib.request

import pytest

import adapters.groq as groq_mod
from adapters.groq import GroqClient


def test_generate_builds_messages_and_parses_usage(monkeypatch):
    seen = {}

    def fake(method, url, payload=None, headers=None, timeout=60):
        seen.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "hi there"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3}}

    monkeypatch.setattr(groq_mod, "request_json", fake)
    result = GroqClient(model="m", api_key="k").generate("q", system="s", max_tokens=64)
    assert result.text == "hi there"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 3
    assert result.model == "m"
    assert seen["url"].endswith("/chat/completions")
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "s"}
    assert seen["payload"]["messages"][1] == {"role": "user", "content": "q"}
    assert seen["headers"]["Authorization"] == "Bearer k"


def test_missing_key_raises():
    client = GroqClient(model="m", api_key="placeholder")
    client.api_key = ""  # force empty regardless of any .env on the machine
    with pytest.raises(RuntimeError):
        client.generate("q")


def test_stream_parses_sse(monkeypatch):
    chunks = b"".join([
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b": keepalive comment\n",
        b"\n",
        b'data: {"choices":[{"delta":{"content":" [1]"}}]}\r\n',
        b'data: {"choices":[{"delta":{}}]}\n',      # finish chunk, empty delta
        b'data: {"choices":[]}\n',                   # usage chunk, no choices
        b"data: [DONE]\n",
    ])
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: io.BytesIO(chunks))
    out = list(GroqClient(model="m", api_key="k").stream("q", system="s"))
    assert out == ["Hello", " [1]"]


def test_stream_normalizes_midstream_error(monkeypatch):
    class Stalling(io.BytesIO):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n'
            raise TimeoutError("read timed out")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Stalling(b""))
    with pytest.raises(RuntimeError):
        list(GroqClient(model="m", api_key="k").stream("q"))
