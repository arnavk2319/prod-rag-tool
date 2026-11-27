"""Ingest decision: parse, check, redact, then outcome and reason. Logs ids and counts only.

Density rules (pii_density_high): more than 5 redactions per 1,000 tokens once the document has
at least 200 tokens, or an absolute >= 25 redactions in one document regardless of length.
"""

import hashlib
from collections.abc import Collection
from dataclasses import dataclass

import structlog

from rag_platform.services.ingest.injection import find_injection
from rag_platform.services.ingest.parsers import MAX_UPLOAD_BYTES, parse
from rag_platform.services.ingest.reasons import MAX_TEXT_CHARS, IngestError, Outcome, Reason
from rag_platform.services.ingest.redaction import count_tokens, redact

MAX_REDACTIONS_PER_1K_TOKENS = 5.0
MAX_REDACTIONS_ABSOLUTE = 25  # any document with this many redactions, however short
DENSITY_MIN_TOKENS = 200  # below this, density is too noisy to judge (short notes)
_GARBLE_CHARS = frozenset("ÃÂÐÑ�")
_GARBLE_RATIO = 0.10

_log = structlog.get_logger(__name__)

_OUTCOME = {
    Reason.PARSE_ERROR: Outcome.REJECTED,
    Reason.FILE_TOO_LARGE: Outcome.REJECTED,
    Reason.TYPE_MISMATCH: Outcome.REJECTED,
    Reason.DUPLICATE_HASH: Outcome.REJECTED,
    Reason.RESOURCE_LIMIT: Outcome.REJECTED,
}


@dataclass(frozen=True)
class IngestDecision:
    outcome: Outcome
    reason: Reason | None
    doc_hash: str
    text: str = ""  # redacted text; empty unless accepted
    redactions: int = 0
    redaction_counts: tuple[tuple[str, int], ...] = ()


def is_garbled(text: str) -> bool:
    if not text:
        return False
    sample = text[:20000]
    bad = sum(1 for c in sample if c in _GARBLE_CHARS or (ord(c) < 32 and c not in "\n\r\t"))
    return bad / len(sample) > _GARBLE_RATIO


def _fail(reason: Reason, doc_hash: str) -> IngestDecision:
    d = IngestDecision(_OUTCOME.get(reason, Outcome.QUARANTINED), reason, doc_hash)
    _log.info(
        "ingest_decision", outcome=d.outcome.value, reason=reason.value, doc_hash=doc_hash[:12]
    )
    return d


def assess(
    data: bytes,
    filename: str,
    *,
    known_hashes: Collection[str] = (),
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> IngestDecision:
    doc_hash = hashlib.sha256(data).hexdigest()
    try:
        doc = parse(data, filename, max_bytes=max_bytes)
    except IngestError as e:
        return _fail(e.reason, doc_hash)
    if doc_hash in known_hashes:
        return _fail(Reason.DUPLICATE_HASH, doc_hash)
    if is_garbled(doc.text):
        return _fail(Reason.GARBLED_TEXT, doc_hash)
    if not doc.text.strip():
        return _fail(Reason.NO_TEXT_LAYER, doc_hash)
    if len(doc.text) + len(doc.hidden_text) >= MAX_TEXT_CHARS:
        return _fail(Reason.RESOURCE_LIMIT, doc_hash)
    if find_injection(doc.text + "\n" + doc.hidden_text) is not None:
        return _fail(Reason.PROMPT_INJECTION_SUSPECTED, doc_hash)
    red = redact(doc.text)
    tokens = count_tokens(red.text)
    dense = (
        tokens >= DENSITY_MIN_TOKENS and red.total * 1000 / tokens > MAX_REDACTIONS_PER_1K_TOKENS
    )
    if dense or red.total >= MAX_REDACTIONS_ABSOLUTE:
        _log.info(
            "ingest_decision",
            outcome="quarantined",
            reason="pii_density_high",
            doc_hash=doc_hash[:12],
            redactions=red.total,
            tokens=tokens,
        )
        return IngestDecision(
            Outcome.QUARANTINED,
            Reason.PII_DENSITY_HIGH,
            doc_hash,
            redactions=red.total,
            redaction_counts=tuple(sorted(red.counts.items())),
        )
    reason = Reason.PII_REDACTED if red.total else None
    _log.info(
        "ingest_decision",
        outcome="accepted",
        reason=reason.value if reason else None,
        doc_hash=doc_hash[:12],
        redactions=red.total,
        tokens=tokens,
    )
    return IngestDecision(
        Outcome.ACCEPTED,
        reason,
        doc_hash,
        text=red.text,
        redactions=red.total,
        redaction_counts=tuple(sorted(red.counts.items())),
    )
