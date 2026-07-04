"""ElevenLabs text-to-speech, so the spoken assistant has a real, human-sounding voice.

Returns MP3 bytes for a piece of text. The API key stays server-side; the browser calls the
app's /api/tts endpoint, never ElevenLabs directly. Kept dependency-free (stdlib urllib) to match
the other adapters. If anything fails, the caller falls back to the browser's built-in voice.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
_USER_AGENT = "skein-lite/1.0 (+https://github.com/hoomanesteki/agentic-rag-knowledge-ai-platform)"


class ElevenLabsTTS:
    def __init__(self, api_key: str, model: str = "eleven_flash_v2_5",
                 output_format: str = "mp3_44100_128") -> None:
        self.api_key = api_key
        self.model = model
        self.output_format = output_format

    def synthesize(self, text: str, voice_id: str, timeout: int = 30) -> bytes:
        """Return MP3 audio for `text` in the given voice. Raises RuntimeError on failure so the
        endpoint can fall back to the browser voice rather than 500."""
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is not set")
        url = "{}/{}?output_format={}".format(_BASE, voice_id, self.output_format)
        payload = json.dumps({
            "text": text,
            "model_id": self.model,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.75,
                               "style": 0.25, "use_speaker_boost": True},
        })
        req = urllib.request.Request(url, data=payload.encode("utf-8"), method="POST")
        req.add_header("xi-api-key", self.api_key)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "audio/mpeg")
        req.add_header("User-Agent", _USER_AGENT)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError("ElevenLabs TTS HTTP {}: {}".format(exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("ElevenLabs TTS failed: {}".format(exc.reason)) from exc
