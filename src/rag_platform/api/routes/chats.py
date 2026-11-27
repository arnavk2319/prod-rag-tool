"""Chats, messages and per-message traces."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_platform.api.routes._common import Cursor, Limit, not_implemented, spec
from rag_platform.api.schemas import (
    ActionResult,
    Chat,
    ChatCreate,
    ChatDetail,
    ChatPage,
    ChatUpdate,
    Message,
    MessageCreate,
    TraceDetail,
)

router = APIRouter(prefix="/v1", tags=["chats"])


@router.post(
    "/chats",
    **spec("create_chat", Chat, summary="Create a chat", status=201, idempotent=True),
)
async def create_chat(body: ChatCreate, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get("/chats", **spec("list_chats", ChatPage, summary="List chats"))
async def list_chats(request: Request, limit: Limit = 50, cursor: Cursor = None) -> JSONResponse:
    return not_implemented(request)


@router.get("/chats/{id}", **spec("get_chat", ChatDetail, summary="Get a chat"))
async def get_chat(id: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.patch("/chats/{id}", **spec("update_chat", Chat, summary="Rename a chat"))
async def update_chat(id: str, body: ChatUpdate, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.delete("/chats/{id}", **spec("delete_chat", ActionResult, summary="Delete a chat"))
async def delete_chat(id: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/chats/{id}/messages",
    **spec("create_message", Message, summary="Ask a question", idempotent=True),
)
async def create_message(id: str, body: MessageCreate, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get(
    "/chats/{id}/messages/{mid}/trace",
    **spec("get_message_trace", TraceDetail, summary="Trace of one answer"),
)
async def get_message_trace(id: str, mid: str, request: Request) -> JSONResponse:
    return not_implemented(request)
