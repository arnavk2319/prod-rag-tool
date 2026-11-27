"""Trace-id and request metrics middleware."""

import re
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rag_platform.metrics import HTTP_LATENCY, HTTP_REQUESTS

TRACE_HEADER = "X-Trace-Id"
_VALID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(TRACE_HEADER, "")
        trace_id = incoming if _VALID.match(incoming) else uuid.uuid4().hex
        request.state.trace_id = trace_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
            structlog.get_logger().info("request", method=request.method, path=path, status=status)
            structlog.contextvars.clear_contextvars()
        response.headers[TRACE_HEADER] = trace_id
        return response
