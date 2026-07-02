"""Groq chat completions behind the LLMClient interface (hosted, OpenAI-compatible, fast)."""
from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator

from ._http import request_json
from .base import LLMResult
from .config import get_settings

_DEFAULT_BASE = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqClient:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or os.getenv("GROQ_MODEL_LARGE", _DEFAULT_MODEL)
        self.api_key = api_key or settings.groq_api_key
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL", _DEFAULT_BASE)).rstrip("/")

    def _messages(self, prompt: str, system: str | None) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 512) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set; put it in .env")
        body = {"model": self.model, "messages": self._messages(prompt, system),
                "max_tokens": max_tokens, "temperature": 0}
        resp = request_json("POST", self.base_url + "/chat/completions", body,
                            {"Authorization": "Bearer " + self.api_key})
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        return LLMResult(text=text,
                         prompt_tokens=usage.get("prompt_tokens", 0),
                         completion_tokens=usage.get("completion_tokens", 0),
                         model=self.model)

    def stream(self, prompt: str, *, system: str | None = None,
               max_tokens: int = 512) -> Iterator[str]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set; put it in .env")
        body = {"model": self.model, "messages": self._messages(prompt, system),
                "max_tokens": max_tokens, "temperature": 0, "stream": True}
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            resp = urllib.request.urlopen(req, timeout=60)  # noqa: S310
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise RuntimeError("groq stream -> HTTP {}: {}".format(exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("groq stream failed: {}".format(exc.reason)) from exc
        with resp:
            try:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        choice = json.loads(data)["choices"][0]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
            except (OSError, http.client.HTTPException) as exc:
                # mid-stream stall/reset/truncation: normalize so the endpoint can degrade
                raise RuntimeError("groq stream failed: {}".format(exc)) from exc
