"""Admin: index versions, quarantine, registry, flags, audit, system health."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rag_platform.api.routes._common import Cursor, Limit, not_implemented, spec
from rag_platform.api.schemas import (
    ActionResult,
    AuditPage,
    AuditVerifyResult,
    Flag,
    FlagUpdate,
    IndexVersion,
    IndexVersionPage,
    QuarantinePage,
    RegistryPointers,
    RegistryPointersUpdate,
    SystemHealth,
)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get(
    "/index/versions",
    **spec("list_index_versions", IndexVersionPage, summary="List index versions"),
)
async def list_index_versions(
    request: Request, limit: Limit = 50, cursor: Cursor = None
) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/index/versions/{version}/stage",
    **spec("stage_index_version", IndexVersion, summary="Stage an index version"),
)
async def stage_index_version(version: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/index/versions/{version}/commit",
    **spec("commit_index_version", IndexVersion, summary="Commit a staged version (gate passed)"),
)
async def commit_index_version(version: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/index/versions/{version}/rollback",
    **spec("rollback_index_version", IndexVersion, summary="Roll back to a version"),
)
async def rollback_index_version(version: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get("/quarantine", **spec("list_quarantine", QuarantinePage, summary="List quarantine"))
async def list_quarantine(
    request: Request, limit: Limit = 50, cursor: Cursor = None
) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/quarantine/{id}:release",
    **spec("release_quarantine", ActionResult, summary="Release a quarantined document"),
)
async def release_quarantine(id: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/quarantine/{id}:delete",
    **spec("delete_quarantine", ActionResult, summary="Delete a quarantined document"),
)
async def delete_quarantine(id: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get(
    "/registry/pointers",
    **spec("get_registry_pointers", RegistryPointers, summary="Get registry pointers"),
)
async def get_registry_pointers(request: Request) -> JSONResponse:
    return not_implemented(request)


@router.put(
    "/registry/pointers",
    **spec("put_registry_pointers", RegistryPointers, summary="Set registry pointers"),
)
async def put_registry_pointers(body: RegistryPointersUpdate, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get("/flags/{key}", **spec("get_flag", Flag, summary="Get a flag"))
async def get_flag(key: str, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.put("/flags/{key}", **spec("put_flag", Flag, summary="Set a flag"))
async def put_flag(key: str, body: FlagUpdate, request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get("/audit", **spec("list_audit", AuditPage, summary="List audit entries"))
async def list_audit(request: Request, limit: Limit = 50, cursor: Cursor = None) -> JSONResponse:
    return not_implemented(request)


@router.post(
    "/audit:verify",
    **spec("verify_audit", AuditVerifyResult, summary="Verify the audit hash chain"),
)
async def verify_audit(request: Request) -> JSONResponse:
    return not_implemented(request)


@router.get(
    "/system/health", **spec("get_system_health", SystemHealth, summary="Per-service health")
)
async def get_system_health(request: Request) -> JSONResponse:
    return not_implemented(request)
