"""Single JSON error envelope for every failure."""

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from rag_platform.domain.errors import AppError, Internal, NotFound, ValidationFailed


def envelope(err: AppError) -> dict[str, dict[str, str]]:
    trace_id = str(structlog.contextvars.get_contextvars().get("trace_id", ""))
    return {"error": {"code": err.code, "message": err.message, "trace_id": trace_id}}


def to_response(err: AppError) -> JSONResponse:
    return JSONResponse(envelope(err), status_code=err.status_code)


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
        err.code = "http_error"
        return to_response(err)

    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Runs outside TraceMiddleware (contextvars already cleared): use request.state.
        trace_id = str(getattr(request.state, "trace_id", ""))
        structlog.get_logger().error("unhandled", error_type=type(exc).__name__, trace_id=trace_id)
        body = envelope(Internal("internal error"))
        body["error"]["trace_id"] = trace_id
        return JSONResponse(body, status_code=500, headers={"X-Trace-Id": trace_id})

    app.add_exception_handler(AppError, app_error)
    app.add_exception_handler(RequestValidationError, validation)
    app.add_exception_handler(StarletteHTTPException, http)
    app.add_exception_handler(Exception, unhandled)
