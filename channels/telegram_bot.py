"""The store assistant on Telegram: a messaging concierge that is a real MCP client of the tools.

The bot launches mcp_server over stdio and answers every message by calling the `skein_ask` tool,
so it is the exact read-only surface any MCP host uses: anonymous, order and account PII blocked,
grounded and cited. It never touches the pipeline directly. Voice in and voice out reuse the app's
own adapters (Groq Whisper for speech-to-text, ElevenLabs for text-to-speech); when text-to-speech
is not configured, it simply replies in text.

The reply logic lives in AssistantBrain, which takes injected callables (ask, transcribe,
synthesize), so it is unit-tested offline with fakes and no network or subprocess.

Run:
    make telegram          # needs TELEGRAM_BOT_TOKEN from @BotFather
Real answers need GROQ_API_KEY + COHERE_API_KEY, `make up`, and an ingest (same as the web app).
Voice in needs TRANSCRIBE_PROVIDER=groq; voice out needs ELEVENLABS_API_KEY (else it replies text).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass

import httpx

_API = "https://api.telegram.org/bot{token}/{method}"
_FILE = "https://api.telegram.org/file/bot{token}/{path}"

# Generic fallbacks: no brand or persona name lives in engine code (leak-linted). The welcome line
# is built from the active pack at startup and injected into the brain.
_EMPTY = ("I don't have enough to answer that from the store's information. Try rephrasing, or ask "
          "me about a product, sizing, shipping, or returns.")
_NO_VOICE = "I couldn't make out that voice note. Mind typing it, or sending it again?"
_SNAG = "Sorry, I hit a brief snag on my end. Give it a moment and ask me again."


@dataclass
class Reply:
    """What to send back: always a text line, plus optional spoken audio (mp3 bytes)."""
    text: str
    voice: bytes | None = None


def _extract_answer(result) -> dict:
    """Pull the {answer, citations, ...} dict out of an MCP CallToolResult, tolerant of shape."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        if "answer" in structured:
            return structured
        if isinstance(structured.get("result"), dict):
            return structured["result"]
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return {"answer": text}
            if isinstance(parsed, dict):
                return parsed
    return {"answer": ""}


class AssistantBrain:
    """Channel-agnostic reply logic. `ask` is async (query:str) -> dict with an "answer"; transcribe
    and synthesize are optional sync callables, so voice degrades to text when they are absent."""

    def __init__(self, ask, *, transcribe=None, synthesize=None,
                 welcome: str | None = None) -> None:
        self._ask = ask
        self._transcribe = transcribe
        self._synthesize = synthesize
        self._welcome = welcome or ("Hi, I'm the shopping assistant. Ask me about products, "
                                    "sizing, shipping, or returns, by text or a voice note.")

    async def reply_to_text(self, text: str) -> Reply:
        data = await self._ask(text)
        return Reply(text=(data.get("answer") or "").strip() or _EMPTY)

    async def reply_to_voice(self, audio: bytes, mime: str = "audio/ogg") -> Reply:
        if not self._transcribe:
            return Reply(text="Voice isn't enabled here right now, but type your question and I'll "
                              "help.")
        heard = (self._transcribe(audio, mime=mime) or "").strip()
        if not heard:
            return Reply(text=_NO_VOICE)
        data = await self._ask(heard)
        answer = (data.get("answer") or "").strip() or _EMPTY
        voice = None
        if self._synthesize:
            try:
                voice = self._synthesize(answer)  # None or bytes; failures fall back to text
            except Exception:  # noqa: BLE001 - a TTS failure must never drop the text reply
                voice = None
        return Reply(text=answer, voice=voice)

    async def handle_message(self, message: dict, download=None) -> Reply | None:
        """Turn one Telegram `message` object into a Reply, or None to ignore it. `download` is an
        async (file_id) -> bytes used to fetch a voice note."""
        text = message.get("text")
        if text:
            if text.strip().lower() in ("/start", "/help"):
                return Reply(text=self._welcome)
            return await self.reply_to_text(text)
        clip = message.get("voice") or message.get("audio")
        if clip and download is not None:
            audio = await download(clip["file_id"])
            if not audio:
                return Reply(text=_NO_VOICE)
            return await self.reply_to_voice(audio, mime=clip.get("mime_type") or "audio/ogg")
        return None


class TelegramClient:
    """The Bot API calls the bot needs: long-poll, send a message, send spoken audio, fetch a file.
    Thin wrapper over httpx so there is no heavyweight bot framework dependency."""

    def __init__(self, token: str, http: httpx.AsyncClient) -> None:
        self._token = token
        self._http = http

    def _url(self, method: str) -> str:
        return _API.format(token=self._token, method=method)

    async def get_updates(self, offset: int, timeout: int = 25) -> list:
        resp = await self._http.get(self._url("getUpdates"),
                                    params={"offset": offset, "timeout": timeout},
                                    timeout=timeout + 10)
        resp.raise_for_status()
        return resp.json().get("result", [])

    async def send_message(self, chat_id, text: str) -> None:
        await self._http.post(self._url("sendMessage"),
                              json={"chat_id": chat_id, "text": text[:4096]})

    async def send_audio(self, chat_id, audio: bytes) -> None:
        # sendAudio plays inline and accepts mp3 reliably (sendVoice would require ogg/opus).
        await self._http.post(self._url("sendAudio"),
                              data={"chat_id": chat_id, "title": "Voice reply"},
                              files={"audio": ("reply.mp3", audio, "audio/mpeg")})

    async def download(self, file_id: str) -> bytes | None:
        meta = await self._http.get(self._url("getFile"), params={"file_id": file_id})
        path = (meta.json().get("result") or {}).get("file_path")
        if not path:
            return None
        got = await self._http.get(_FILE.format(token=self._token, path=path))
        return got.content if got.status_code == 200 else None


def _pack_welcome(domain: str) -> str:
    """Build the greeting from the active pack, so no persona or brand name is hardcoded here."""
    from data.lakehouse import load_manifest
    manifest = load_manifest(os.path.join("domains", domain))
    persona = manifest.get("persona") or {}
    assistant = (persona.get("assistant") or "the shopping assistant").strip()
    brand = (manifest.get("brand") or "our store").strip()
    return ("Hi, I'm {a}, the shopping assistant for {b}. Ask me about products, sizing, "
            "materials, shipping, or returns, by text or a voice note. For order and account "
            "details, please use the website, where I can verify it's you.").format(
                a=assistant, b=brand)


def _make_synthesize(settings):
    """Return a synthesize(text)->bytes callable if ElevenLabs is configured, else None."""
    key = getattr(settings, "elevenlabs_api_key", "")
    if not key:
        return None
    from adapters.elevenlabs import ElevenLabsTTS
    tts = ElevenLabsTTS(key, model=settings.elevenlabs_model)
    voice_id = settings.elevenlabs_voice_id

    def synthesize(text: str) -> bytes | None:
        try:
            return tts.synthesize(text[:1200], voice_id)
        except Exception:  # noqa: BLE001 - never let a TTS failure break a reply
            return None

    return synthesize


async def _poll_loop(client: TelegramClient, brain: AssistantBrain) -> None:
    offset = 0
    print("The assistant is live on Telegram. Message the bot to start.", file=sys.stderr)
    while True:
        try:
            updates = await client.get_updates(offset)
        except Exception as exc:  # noqa: BLE001 - transient network errors must not kill the bot
            print("poll error: {}".format(str(exc)[:160]), file=sys.stderr)
            await asyncio.sleep(3)
            continue
        for update in updates:
            offset = max(offset, update.get("update_id", offset) + 1)
            message = update.get("message") or update.get("edited_message")
            if not message or "chat" not in message:
                continue
            chat_id = message["chat"]["id"]
            try:
                reply = await brain.handle_message(message, download=client.download)
            except Exception as exc:  # noqa: BLE001 - one bad turn should not stop the bot
                print("handle error: {}".format(str(exc)[:160]), file=sys.stderr)
                reply = Reply(text=_SNAG)
            if reply is None:
                continue
            if reply.text:
                await client.send_message(chat_id, reply.text)
            if reply.voice:
                try:
                    await client.send_audio(chat_id, reply.voice)
                except Exception as exc:  # noqa: BLE001 - text already went; audio is a bonus
                    print("voice send failed: {}".format(str(exc)[:160]), file=sys.stderr)


async def _run() -> int:
    from dotenv import load_dotenv
    load_dotenv()  # read the token and keys from .env, the same way the rest of the app does
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN (get one from @BotFather) to run the Telegram bot.",
              file=sys.stderr)
        return 1

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from adapters.config import get_settings
    from adapters.factory import make_transcriber

    settings = get_settings()
    # Speech-to-text only when a real provider is set; the fake transcriber would "hear" a canned
    # string, so leave voice-in off in that case and let the brain ask the shopper to type.
    transcribe = None
    if getattr(settings, "transcribe_provider", "fake") != "fake":
        transcribe = make_transcriber().transcribe
    synthesize = _make_synthesize(settings)
    welcome = _pack_welcome(settings.domain)

    # The MCP server, launched over stdio as a subprocess and kept for the bot's lifetime, so a turn
    # does not pay a cold start. Same interpreter and env as this process, so it inherits the keys.
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"],
                                   env={**os.environ, "PYTHONPATH": os.getcwd()})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def ask(query: str) -> dict:
                result = await session.call_tool("skein_ask", {"query": query})
                return _extract_answer(result)

            brain = AssistantBrain(ask, transcribe=transcribe, synthesize=synthesize,
                                   welcome=welcome)
            async with httpx.AsyncClient() as http:
                await _poll_loop(TelegramClient(token, http), brain)
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_run()))
    except KeyboardInterrupt:
        print("\nbye", file=sys.stderr)


if __name__ == "__main__":
    main()
