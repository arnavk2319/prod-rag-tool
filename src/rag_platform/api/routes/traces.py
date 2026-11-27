"""Trace browsing."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_platform.api.routes._common import Cursor, Limit, not_implemented, spec
from rag_platform.api.schemas import TraceDetail, TracePage

router = APIRouter(prefix="/v1", tags=["traces"])


@router.get("/traces", **spec("list_traces", TracePage, summary="List traces"))
async def list_traces(request: Request, limit: Limit = 50, cursor: Cursor = None) -> JSONResponse:
    return not_implemented(request)


@router.get("/traces/{trace_id}", **spec("get_trace", TraceDetail, summary="Get a trace"))
async def get_trace(trace_id: str, request: Request) -> JSONResponse:
    return not_implemented(request)
