"""Neutralize prompt-injection attempts in retrieved (user-generated) text before it reaches
the model. Retrieved content is data, never instructions.

Layers: NFKC-normalize and strip zero-width characters (defeats homoglyph and invisible-char
bypasses), collapse whitespace (so a chunk cannot forge prompt structure), then redact
instruction-like spans up to the sentence end (so legitimate text after an injection
survives). English and French, since the corpus is bilingual.

This is defense in depth alongside the system prompt and the untrusted-data reminder, not a
replacement. Known limits (covered by the system prompt, revisited at M6/M8): a payload split
into a separate sentence from its trigger survives, and novel phrasings or other languages are
not matched. Over-matching legitimate review sentences (for example "disregard the previous
review") is accepted as the safe trade-off.
"""
from __future__ import annotations

import re
import unicodedata

# zero-width joiners/spaces and BOM, used to break up trigger words invisibly
_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None)
_TAIL = r"[^.!?]*[.!?]?"  # match to the end of the offending sentence

_INJECTION_PATTERNS = [
    # English
    re.compile(r"(ignore|disregard)\s+(all\s+)?(the\s+)?(previous|prior|above)\b"
               r"(\s+instructions?)?" + _TAIL, re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|the\s+above|previous)" + _TAIL, re.IGNORECASE),
    re.compile(r"(you\s+are\s+now|from\s+now\s+on)\b" + _TAIL, re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)" + _TAIL, re.IGNORECASE),
    re.compile(r"act\s+as\s+(an?\s+)?(assistant|admin|manager|system)" + _TAIL, re.IGNORECASE),
    re.compile(r"(system|developer)\s*(prompt|message)\s*:" + _TAIL, re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:" + _TAIL, re.IGNORECASE),
    # French
    re.compile(r"ignore[rz]?\s+(toutes?\s+)?(les\s+)?instructions\s+"
               r"(pr[ée]c[ée]dentes|ci-dessus)" + _TAIL, re.IGNORECASE),
    re.compile(r"oublie[rz]?\s+(tout|toutes|le\s+reste)" + _TAIL, re.IGNORECASE),
    re.compile(r"(tu\s+es|vous\s+[êe]tes)\s+maintenant" + _TAIL, re.IGNORECASE),
    re.compile(r"(nouvelles?\s+instructions?\s*:|d[ée]sormais\b)" + _TAIL, re.IGNORECASE),
]
_REDACTED = "[removed]"


def sanitize_context(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    cleaned = " ".join(text.split())  # collapse whitespace, kill forged newlines
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub(_REDACTED, cleaned)
    return " ".join(cleaned.split())  # tidy any doubled spaces from redaction
