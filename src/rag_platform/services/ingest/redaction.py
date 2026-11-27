"""PII redaction with typed placeholders. Conservative regexes; cards are Luhn-checked."""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from rag_platform.services.ingest.reasons import MAX_TEXT_CHARS, IngestError, Reason

CARD, ACCOUNT, SIN, EMAIL, PHONE = "[CARD]", "[ACCOUNT]", "[SIN]", "[EMAIL]", "[PHONE]"

_EMAIL = re.compile(r"(?<![\w.%+'-])[\w.%+'-]+@[\w-]+(?:\.[\w-]+)*\.[^\W\d_]{2,}")
_CARD = re.compile(r"(?<![\d-])\d(?:[\s.-]{0,3}\d){12,18}(?!\d)")
_SIN = re.compile(r"(?<![\d-])(?:\d{3}[- ]\d{3}[- ]\d{3}|\d{3}-\d{2}-\d{4})(?![\d-])")
_SIN_KW = re.compile(r"(?i)\b(?:sin|ssn|social insurance(?: number)?)\b[^\d\n]{0,12}(\d{9})(?!\d)")
_PHONE = re.compile(
    r"(?<![\w-])\+\d{1,3}[ .-]?(?:\(\d{1,4}\)[ .-]?)?\d{1,4}(?:[ .-]?\d{2,4}){2,4}(?![\d-])"
    r"|(?<![\d-])(?:\+?1[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\d-])"
)
_ACCT_KW = re.compile(
    r"(?i)\b(?:acct|account|a/c)\b(?:\s*(?:no|num|number|#))?\.?[^\d\n]{0,12}(\d(?:[ -]?\d){5,})(?!\d)"  # noqa: E501
)
_ACCT_BARE = re.compile(r"(?<![\d-])\d{10,}(?![\d-])")
_FORMAT_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))


def normalise(text: str) -> str:
    """Strip zero-width/format characters and fold full-width forms (NFKC) before matching."""
    return unicodedata.normalize("NFKC", text).translate(_FORMAT_CHARS)


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def count_tokens(text: str) -> int:
    """Whitespace token count; cheap and deterministic (density and sizing only)."""
    return len(text.split())


def redact(text: str) -> RedactionResult:
    if len(text) > MAX_TEXT_CHARS:  # all regexes are linear; the cap bounds total work
        raise IngestError(Reason.RESOURCE_LIMIT)
    counts: Counter[str] = Counter()
    text = normalise(text)  # returned text is the normalised form; counts refer to it

    def sub(pattern: re.Pattern[str], placeholder: str, s: str, group: int = 0) -> str:
        def repl(m: re.Match[str]) -> str:
            counts[placeholder] += 1
            if group:
                a, b = m.span(group)
                return m.group(0)[: a - m.start()] + placeholder + m.group(0)[b - m.start() :]
            return placeholder

        return pattern.sub(repl, s)

    def card_repl(m: re.Match[str]) -> str:
        if luhn_ok(re.sub(r"\D", "", m.group(0))):
            counts[CARD] += 1
            return CARD
        return m.group(0)

    text = _EMAIL.sub(lambda m: (counts.update([EMAIL]), EMAIL)[1], text)
    text = _CARD.sub(card_repl, text)
    text = sub(_SIN_KW, SIN, text, group=1)
    text = sub(_SIN, SIN, text)
    text = sub(_PHONE, PHONE, text)
    text = sub(_ACCT_KW, ACCOUNT, text, group=1)
    text = sub(_ACCT_BARE, ACCOUNT, text)
    return RedactionResult(text, dict(counts))
