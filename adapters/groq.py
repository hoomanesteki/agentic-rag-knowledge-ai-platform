"""Groq chat completions behind the LLMClient interface (hosted, OpenAI-compatible, fast)."""
from __future__ import annotations

import os

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

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 512) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set; put it in .env")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": 0}
        resp = request_json("POST", self.base_url + "/chat/completions", body,
                            {"Authorization": "Bearer " + self.api_key})
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        return LLMResult(text=text,
                         prompt_tokens=usage.get("prompt_tokens", 0),
                         completion_tokens=usage.get("completion_tokens", 0),
                         model=self.model)
