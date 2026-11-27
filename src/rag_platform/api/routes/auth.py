"""Config, login and identity."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_platform.api.routes._common import not_implemented, spec
from rag_platform.api.schemas import ConfigResponse, LoginRequest, LoginResponse, Me

router = APIRouter(prefix="/v1", tags=["auth"])


@router.get(
    "/config",
    **spec("get_config", ConfigResponse, summary="Non-sensitive runtime config", auth=False),
)
async def get_config(request: Request) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/auth/login", **spec("login", LoginResponse, summary="Exchange credentials", auth=False)
)
async def login(body: LoginRequest, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get("/me", **spec("get_me", Me, summary="Current user"))
async def get_me(request: Request) -> JSONResponse:
    return not_implemented(request)
