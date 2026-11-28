"""Unversioned ops probes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_platform.api.routes._common import not_implemented, spec
from rag_platform.api.schemas import HealthStatus

router = APIRouter(tags=["ops"])


@router.get("/healthz", **spec("healthz", HealthStatus, summary="Liveness probe", auth=False))
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", **spec("readyz", HealthStatus, summary="Readiness probe", auth=False))
async def readyz(request: Request) -> JSONResponse:
    return not_implemented(request)


_METRICS = spec("metrics", None, summary="Prometheus metrics (internal only)", auth=False)
_METRICS["responses"][200]["content"] = {"text/plain": {"schema": {"type": "string"}}}


@router.get("/metrics", **_METRICS)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
