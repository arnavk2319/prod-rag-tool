"""Ingest outcomes and reason codes; values match the ingest fixtures expected.yaml."""

from enum import StrEnum

# Longer text is refused (resource_limit), never partly scanned: skipping would leak PII.
MAX_TEXT_CHARS = 8_000_000


class Outcome(StrEnum):
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class Reason(StrEnum):
    PII_REDACTED = "pii_redacted"  # accepted, after redaction
    NO_TEXT_LAYER = "no_text_layer"  # scanned-like PDF
    GARBLED_TEXT = "garbled_text"
    DUPLICATE_HASH = "duplicate_hash"
    PARSE_ERROR = "parse_error"
    PROMPT_INJECTION_SUSPECTED = "prompt_injection_suspected"
    FILE_TOO_LARGE = "file_too_large"
    TYPE_MISMATCH = "type_mismatch"
    ENCRYPTED_PDF = "encrypted_pdf"
    PII_DENSITY_HIGH = "pii_density_high"
    RESOURCE_LIMIT = "resource_limit"  # zip-bomb-like input


class IngestError(Exception):
    """Raised by parsers; carries the reason code. Message never includes document text."""

    def __init__(self, reason: Reason) -> None:
        super().__init__(reason.value)
        self.reason = reason
