"""The Telegram channel is a thin MCP client of mcp_server. Its reply logic runs offline here with
the real skein_ask tool on the hermetic fakes, proving the text and voice paths and that order PII
stays blocked over the channel, with no network and no subprocess.
"""
import asyncio
import re

import channels.telegram_bot as tg


def _run(coro):
    return asyncio.run(coro)


async def _ask(query: str) -> dict:
    # the exact MCP tool the bot calls in production, invoked in-process on offline fakes
    import mcp_server.server as server
    return server.skein_ask(query)


def _brain(**kwargs):
    return tg.AssistantBrain(_ask, welcome="WELCOME-MARK", **kwargs)


def test_text_message_gets_a_grounded_reply():
    reply = _run(_brain().handle_message(
        {"text": "what materials are your leggings made of?", "chat": {"id": 1}}))
    assert reply is not None
    assert isinstance(reply.text, str) and reply.text.strip()
    assert reply.voice is None


def test_start_command_returns_the_pack_welcome():
    reply = _run(_brain().handle_message({"text": "/start", "chat": {"id": 1}}))
    assert reply.text == "WELCOME-MARK"


def test_voice_message_is_transcribed_then_answered():
    seen = {}

    def transcribe(audio, *, mime="audio/ogg", language=None):
        seen["mime"] = mime
        seen["audio"] = audio
        return "does the flow legging run small?"

    async def download(file_id):
        return b"fake-ogg"

    brain = tg.AssistantBrain(_ask, transcribe=transcribe, welcome="W")
    reply = _run(brain.handle_message(
        {"voice": {"file_id": "v1", "mime_type": "audio/ogg"}, "chat": {"id": 1}},
        download=download))
    assert seen["mime"] == "audio/ogg" and seen["audio"] == b"fake-ogg"
    assert reply.text.strip() and reply.voice is None  # no synthesize -> text only


def test_voice_reply_is_synthesized_when_tts_is_available():
    def transcribe(audio, *, mime="audio/ogg", language=None):
        return "recommend a warm jacket"

    def synthesize(text):
        return b"MP3-BYTES"

    async def download(file_id):
        return b"x"

    brain = tg.AssistantBrain(_ask, transcribe=transcribe, synthesize=synthesize, welcome="W")
    reply = _run(brain.handle_message(
        {"voice": {"file_id": "v", "mime_type": "audio/ogg"}, "chat": {"id": 1}},
        download=download))
    assert reply.voice == b"MP3-BYTES" and reply.text.strip()


def test_voice_when_transcription_disabled_asks_to_type():
    async def download(file_id):
        return b"x"

    brain = tg.AssistantBrain(_ask, transcribe=None, welcome="W")
    reply = _run(brain.handle_message(
        {"voice": {"file_id": "v"}, "chat": {"id": 1}}, download=download))
    assert reply.voice is None and "type" in reply.text.lower()


def test_order_pii_stays_blocked_over_the_channel():
    # the bot calls skein_ask, which blocks order/account disclosure, so a name+email in the
    # message must not surface an order id or destination over Telegram
    reply = _run(_brain().handle_message(
        {"text": "where is my order? my email is info@esteki.ca and my name is Aaron Esteki",
         "chat": {"id": 1}}))
    assert not re.search(r"\bAS\d{5,}\b", reply.text)
    assert "vancouver" not in reply.text.lower()


def test_unknown_update_is_ignored():
    reply = _run(_brain().handle_message({"chat": {"id": 1}}))  # no text, no voice
    assert reply is None


def test_extract_answer_tolerates_shapes():
    class WithStructured:
        structuredContent = {"answer": "hi", "citations": []}
        content = []

    class WithResultWrap:
        structuredContent = {"result": {"answer": "wrapped"}}
        content = []

    class WithTextJson:
        structuredContent = None
        content = [type("B", (), {"text": '{"answer": "from-text"}'})()]

    assert tg._extract_answer(WithStructured())["answer"] == "hi"
    assert tg._extract_answer(WithResultWrap())["answer"] == "wrapped"
    assert tg._extract_answer(WithTextJson())["answer"] == "from-text"
