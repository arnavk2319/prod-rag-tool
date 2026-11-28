"""Shared psycopg helper: one transaction per call, role and tenant set per transaction."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

SET_TENANT = "SELECT set_config('app.tenant_id', %s, true)"


def normalize_dsn(dsn: str) -> str:
    return dsn.replace("postgresql+psycopg://", "postgresql://")


@asynccontextmanager
async def tx(
    dsn: str, *, role: str | None = "rag_app", tenant_id: str | None = None
) -> AsyncIterator[psycopg.AsyncConnection[dict[str, Any]]]:
    """Commits on success, rolls back on any exception. `role` is configuration, never input."""
    async with await psycopg.AsyncConnection.connect(
        normalize_dsn(dsn), row_factory=dict_row
    ) as conn:
        if role:
            # nosemgrep: role is a constructor argument quoted with sql.Identifier
            await conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
        if tenant_id is not None:
            await conn.execute(SET_TENANT, (tenant_id,))
        yield conn
