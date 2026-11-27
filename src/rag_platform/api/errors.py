"""Single JSON error envelope for every failure."""

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from rag_platform.domain.errors import AppError, Internal, NotFound, ValidationFailed

REQUEST_ID_HEADER = "X-Request-ID"
_STATUS_CODE = {
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_failed",
    429: "rate_limited",
}


def code_for_status(status: int) -> str:
    """Map an HTTP status to a code in the contract enum."""
    if status in _STATUS_CODE:
        return _STATUS_CODE[status]
    if status in (502, 503, 504):
        return "dependency_unavailable"
    return "internal_error" if status >= 500 else "validation_failed"


def _request_id() -> str:
    return str(structlog.contextvars.get_contextvars().get("trace_id", ""))


def envelope(err: AppError) -> dict[str, dict[str, Any]]:
    body: dict[str, Any] = {"code": err.code, "message": err.message, "request_id": _request_id()}
    if err.details:
        body["details"] = err.details
    return {"error": body}


def to_response(err: AppError) -> JSONResponse:
    body = envelope(err)
    headers = {REQUEST_ID_HEADER: body["error"]["request_id"]}
    if err.retry_after is not None:
        headers["Retry-After"] = str(err.retry_after)
    return JSONResponse(body, status_code=err.status_code, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    async def app_error(_: Request, exc: Exception) -> JSONResponse:
        return to_response(exc if isinstance(exc, AppError) else Internal())

    async def validation(_: Request, exc: Exception) -> JSONResponse:
        return to_response(ValidationFailed("invalid request"))

    async def http(_: Request, exc: Exception) -> JSONResponse:
        status = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
        if status == 404:
            return to_response(NotFound("route not found"))
        err = AppError("http error")
        err.status_code = status
        err.code = code_for_status(status)
        return to_response(err)

    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Runs outside TraceMiddleware (contextvars already cleared): use request.state.
        trace_id = str(getattr(request.state, "trace_id", ""))
        structlog.get_logger().error("unhandled", error_type=type(exc).__name__, trace_id=trace_id)
        body = envelope(Internal("internal error"))
        body["error"]["request_id"] = trace_id
        return JSONResponse(
            body, status_code=500, headers={"X-Trace-Id": trace_id, REQUEST_ID_HEADER: trace_id}
        )

    app.add_exception_handler(AppError, app_error)
    app.add_exception_handler(RequestValidationError, validation)
    app.add_exception_handler(StarletteHTTPException, http)
    app.add_exception_handler(Exception, unhandled)
