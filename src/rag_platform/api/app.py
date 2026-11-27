"""FastAPI application factory."""

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from rag_platform.api.errors import install_error_handlers
from rag_platform.api.middleware import TraceMiddleware
from rag_platform.logging_setup import configure_logging
from rag_platform.settings import get_settings


def create_app() -> FastAPI:
    configure_logging(get_settings().log_level)
    app = FastAPI(title="rag-platform")
    install_error_handlers(app)
    app.add_middleware(TraceMiddleware)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
