"""Documents, chunks and ingest jobs."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_platform.api.routes._common import Cursor, Limit, not_implemented, spec
from rag_platform.api.schemas import (
    ActionResult,
    Chunk,
    Document,
    DocumentAccepted,
    DocumentCreate,
    DocumentPage,
    DocumentStatus,
    Job,
)

router = APIRouter(prefix="/v1", tags=["documents"])


@router.post(
    "/documents",
    **spec(
        "create_document",
        DocumentAccepted,
        summary="Upload a document",
        status=202,
        idempotent=True,
    ),
)
async def create_document(body: DocumentCreate, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get("/documents", **spec("list_documents", DocumentPage, summary="List documents"))
async def list_documents(
    request: Request,
    status: DocumentStatus | None = None,
    limit: Limit = 50,
    cursor: Cursor = None,
) -> JSONResponse:
    return not_implemented(request)


@router.get("/documents/{id}", **spec("get_document", Document, summary="Get a document"))
async def get_document(id: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.delete("/documents/{id}", **spec("delete_document", ActionResult, summary="Delete"))
async def delete_document(id: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get(
    "/documents/{id}/chunks/{chunk_id}",
    **spec("get_chunk", Chunk, summary="Get a chunk for the source pane"),
)
async def get_chunk(id: str, chunk_id: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get("/jobs/{id}", **spec("get_job", Job, summary="Get an ingest job"))
async def get_job(id: str, request: Request) -> JSONResponse:
    return not_implemented(request)
