"""Shared helpers for route modules: error responses, headers, stub body."""

import re
from typing import Annotated, Any

from fastapi import Depends, Header, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from rag_platform.api.schemas import ErrorEnvelope

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

bearer = HTTPBearer(scheme_name="BearerAuth", auto_error=False)

Limit = Annotated[int, Query(ge=1, le=200, description="Page size (default 50, max 200)")]
Cursor = Annotated[str | None, Query(max_length=512, description="Opaque cursor")]


async def x_request_id_header(
    x_request_id: Annotated[
        str | None, Header(alias="X-Request-ID", description="Echoed on the response")
    ] = None,
) -> None:
    return None


async def idempotency_key_header(
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            max_length=255,
            description="Replay-safe key, scoped per tenant and user, valid 24 hours",
        ),
    ] = None,
) -> None:
    return None


_REQUEST_ID_HEADER = {
    "X-Request-ID": {
        "description": "Request id (echo of the request header, or generated)",
        "schema": {"type": "string"},
    }
}
_RETRY_AFTER = {
    "Retry-After": {
        "description": "Seconds to wait before retrying",
        "schema": {"type": "integer", "minimum": 0},
    }
}
_ERRORS = {
    400: "Bad request",
    401: "Missing or invalid credentials (unauthenticated)",
    403: "Not allowed (forbidden)",
    404: "Not found",
    409: "Conflict, including idempotency key reuse with a different body",
    413: "Payload too large",
    422: "Validation failed (validation_failed)",
    429: "Rate or spend limit reached (rate_limited)",
    500: "Internal error",
    503: "Dependency unavailable",
}


def error_responses() -> dict[int | str, dict[str, Any]]:
    out: dict[int | str, dict[str, Any]] = {}
    for status, description in _ERRORS.items():
        headers = {**_REQUEST_ID_HEADER, **(_RETRY_AFTER if status == 429 else {})}
        out[status] = {"model": ErrorEnvelope, "description": description, "headers": headers}
    return out


def spec(
    operation_id: str,
    response_model: type[BaseModel] | None,
    *,
    summary: str,
    status: int = 200,
    auth: bool = True,
    idempotent: bool = False,
) -> dict[str, Any]:
    """Keyword arguments for a route decorator; see ADR-0004."""
    responses = error_responses()
    responses[status] = {"description": "Success", "headers": _REQUEST_ID_HEADER}
    deps: list[Any] = [Depends(x_request_id_header)]
    if auth:
        deps.append(Depends(bearer))
    if idempotent:
        deps.append(Depends(idempotency_key_header))
    return {
        "operation_id": operation_id,
        "response_model": response_model,
        "status_code": status,
        "summary": summary,
        "responses": responses,
        "dependencies": deps,
    }


def not_implemented(request: Request) -> JSONResponse:
    """501 with the shared error envelope; stubs never return fake success data."""
    supplied = request.headers.get("X-Request-ID", "")
    rid = (
        supplied if _REQUEST_ID.fullmatch(supplied) else str(getattr(request.state, "trace_id", ""))
    )
    body = {"error": {"code": "internal_error", "message": "not implemented", "request_id": rid}}
    return JSONResponse(body, status_code=501, headers={"X-Request-ID": rid})
