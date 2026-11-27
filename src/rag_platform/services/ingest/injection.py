"""Built-in prompt-injection pattern list (derived from the 30-item corpus; not read at runtime)."""
# ruff: noqa: E501

import re
import unicodedata

from rag_platform.services.ingest.reasons import MAX_TEXT_CHARS, IngestError, Reason

_HOMOGLYPHS = str.maketrans(
    {"Ι": "I", "і": "i", "ο": "o", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "ѕ": "s"}
)
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))

_PATTERNS = [
    r"\b(?:ignore|disregard|forget|override)\W+(?:(?:all|any|the|your|my|these|those|every|of|previous|prior|above|earlier|preceding|system|developer)\W+){0,4}(?:instructions?|prompts?|rules|guidelines|messages|context|guardrails|directions|training)\b",
    r"\b(?:ignore|disregard|forget)\W+(?:all\W+|everything\W+)?(?:of\W+)?(?:the\W+)?above\b",
    r"\bnew instructions?\s*:",
    r"\bfrom now on\b.{0,40}\b(?:respond|answer|reply)\b",
    r"\b(?:reveal|repeat|print|show)\b.{0,40}\b(?:system prompt|hidden prompt|text above|api keys?)\b",
    r"\bdo not cite\b|\bskip the refusal\b|\bdisable citations\b",
    r"\bnote to (?:ai|the )?assistant\b|\bai (?:assistant|systems?) (?:reading|must)\b",
    r"\b(?:assistant|system|developer)\s*(?:message)?\s*:\s*(?:ignore|you are|the user|understood|reply|disable)",
    r"<!--\s*assistant\b|\[hidden instruction\]|display\s*:\s*none[^>]{0,200}>\s*assistant",
    r"\bthe assistant must\b|\bmust trust this paragraph\b",
    r"\bcommand\s*:\s*output\b",
    r"\byou are now\b.{0,40}(?:dan|maintenance mode|without any restrictions|administrator)",
    r"\bpretend you are\b",
    r"\b(?:debug|maintenance|developer) mode\b.{0,40}\b(?:on|disabled?)\b|safety checks are disabled",
    r"\b(?:send|post|forward)\b.{0,40}\b(?:conversation|chat|context)\b.{0,40}https?://",
    r"!\[[^\]]{0,200}\]\(https?://[^)]{0,300}\?[^)]{0,300}\)",
    r"\b(?:every|all) (?:documents?|chunks?).{0,40}\b(?:regardless of tenant|you can see)\b|\bother (?:bank )?tenant\b",
    r"\b(?:base64|rot13|hex)\b.{0,40}\b(?:follow|obey|apply|decode)\b",
    r"\blist all chunk ids\b",
    r"\bcanary-\d+",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _PATTERNS]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).translate(_HOMOGLYPHS)
    return re.sub(r"\s+", " ", t)


_SPACED = re.compile(r"\b(?:[^\W\d_] ){3,}[^\W\d_]\b")


def find_injection(text: str) -> int | None:
    """Index of the first matching pattern, or None. Never returns matched text."""
    if len(text) > MAX_TEXT_CHARS:
        raise IngestError(Reason.RESOURCE_LIMIT)
    t = normalize(text)
    views = (t, _SPACED.sub(lambda m: m.group(0).replace(" ", ""), t))  # letter-spaced evasion
    for i, p in enumerate(_COMPILED):
        if any(p.search(v) for v in views):
            return i
    return None
