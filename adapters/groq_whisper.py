"""Groq hosted Whisper behind the Transcriber interface (speech to text, OpenAI-compatible).

Uses httpx for the one multipart upload; the browser records a short clip, the API forwards the
bytes here, and the transcript comes back. On failure it raises RuntimeError with the vendor body
(never the key), so the browser can fall back to its live Web Speech recognition.
"""
from __future__ import annotations

import os

import httpx

from .config import get_settings

_DEFAULT_BASE = "https://api.groq.com/openai/v1"
# the STT endpoint infers the audio format from the filename extension, so it must be present
_EXT = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/mpeg": "mp3",
        "audio/mp3": "mp3", "audio/wav": "wav", "audio/x-wav": "wav", "audio/flac": "flac"}


class GroqWhisper:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.whisper_model
        self.api_key = api_key or settings.groq_api_key
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL", _DEFAULT_BASE)).rstrip("/")

    def transcribe(self, audio: bytes, *, mime: str = "audio/webm",
                   language: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set; cannot transcribe")
        data = {"model": self.model, "response_format": "json"}
        if language:
            data["language"] = language
        filename = "audio.{}".format(_EXT.get(mime.split(";")[0].strip(), "webm"))
        try:
            resp = httpx.post(
                self.base_url + "/audio/transcriptions",
                headers={"Authorization": "Bearer " + self.api_key},
                data=data, files={"file": (filename, audio, mime)}, timeout=60.0)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise RuntimeError("groq whisper HTTP {}: {}".format(
                exc.response.status_code, body)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError("groq whisper request failed: {}".format(exc)) from exc
        return (resp.json().get("text") or "").strip()
