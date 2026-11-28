"""Route modules, one per API area."""

from fastapi import APIRouter

from rag_platform.api.routes import admin, auth, chats, documents, ops, traces


def all_routers() -> list[APIRouter]:
    return [
        ops.router,
        auth.router,
        documents.router,
        chats.router,
        traces.router,
        admin.router,
    ]
