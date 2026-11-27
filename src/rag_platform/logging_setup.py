"""structlog JSON logging with trace_id and content redaction."""

import logging
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

# Keys whose values may hold user content or PII; never emitted.
SENSITIVE_KEYS = frozenset(
    {
        "question",
        "query",
        "answer",
        "document",
        "document_text",
        "text",
        "content",
        "prompt",
        "completion",
        "password",
        "token",
        "authorization",
        "jwt_signing_key",
    }
)
REDACTED = "[redacted]"


def _scrub(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            k: REDACTED if str(k).lower() in SENSITIVE_KEYS else _scrub(v) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_scrub(v) for v in value]
    return value


def redact_processor(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if key.lower() in SENSITIVE_KEYS:
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _scrub(event_dict[key])
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_processor,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=False,
    )
