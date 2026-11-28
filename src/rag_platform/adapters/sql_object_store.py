"""ObjectStore over Postgres (`raw_objects`, RLS by tenant): raw and quarantined uploads."""

import hashlib
from typing import Literal

from rag_platform.adapters._pg import tx

RawStatus = Literal["raw", "quarantined"]

_PUT = (
    "INSERT INTO raw_objects (sha256, tenant_id, mime, size, status, content) "
    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (tenant_id, sha256) DO NOTHING"
)
_GET = "SELECT content FROM raw_objects WHERE tenant_id = %s AND sha256 = %s"
_DELETE = "DELETE FROM raw_objects WHERE tenant_id = %s AND sha256 = %s"
_MARK = "UPDATE raw_objects SET status = %s WHERE tenant_id = %s AND sha256 = %s"
_STATUS = "SELECT status FROM raw_objects WHERE tenant_id = %s AND sha256 = %s"


class PostgresObjectStore:
    def __init__(self, dsn: str, *, role: str | None = "rag_app") -> None:
        self._dsn = dsn
        self._role = role

    async def put(
        self, tenant_id: str, data: bytes, mime: str, *, status: RawStatus = "raw"
    ) -> str:
        sha = hashlib.sha256(data).hexdigest()
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            await c.execute(_PUT, (sha, tenant_id, mime, len(data), status, data))
        return sha

    async def get(self, tenant_id: str, sha256: str) -> bytes | None:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            row = await (await c.execute(_GET, (tenant_id, sha256))).fetchone()
        return bytes(row["content"]) if row else None

    async def delete(self, tenant_id: str, sha256: str) -> bool:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            cur = await c.execute(_DELETE, (tenant_id, sha256))
            return cur.rowcount > 0

    async def mark_quarantined(self, tenant_id: str, sha256: str) -> bool:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            cur = await c.execute(_MARK, ("quarantined", tenant_id, sha256))
            return cur.rowcount > 0

    async def status(self, tenant_id: str, sha256: str) -> RawStatus | None:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            row = await (await c.execute(_STATUS, (tenant_id, sha256))).fetchone()
        return row["status"] if row else None
