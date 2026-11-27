"""FastAPI application factory."""

from fastapi import FastAPI

from rag_platform.api.errors import install_error_handlers
from rag_platform.api.middleware import TraceMiddleware
from rag_platform.api.routes import all_routers
from rag_platform.logging_setup import configure_logging
from rag_platform.settings import get_settings


def create_app() -> FastAPI:
    configure_logging(get_settings().log_level)
    app = FastAPI(title="rag-platform", version="1.0.0", separate_input_output_schemas=False)
    install_error_handlers(app)
    app.add_middleware(TraceMiddleware)

    for router in all_routers():
        app.include_router(router)
    return app


app = create_app()
