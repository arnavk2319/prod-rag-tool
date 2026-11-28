"""Ingest job state in `ingest_jobs` (RLS by tenant). Idempotent create, monotone status."""

from rag_platform.adapters._pg import tx
from rag_platform.services.ingest.jobs import IngestJob, JobStatus

_CREATE = (
    "INSERT INTO ingest_jobs (job_id, tenant_id, raw_sha256, status) VALUES (%s, %s, %s, 'queued') "
    "ON CONFLICT (job_id) DO NOTHING"
)
_STATUS = "SELECT status FROM ingest_jobs WHERE tenant_id = %s AND job_id = %s"
# A terminal status is never overwritten by a late duplicate; running only moves forward from
# queued or running (a re-claim after a crash).
_SET = (
    "UPDATE ingest_jobs SET status = %s, error = %s, updated_at = now() "
    "WHERE tenant_id = %s AND job_id = %s AND status <> ALL(%s)"
)
_DOC = "UPDATE ingest_jobs SET doc_id = %s, updated_at = now() WHERE tenant_id = %s AND job_id = %s"
_REQUEUE = (
    "UPDATE ingest_jobs SET status = 'queued', error = NULL, updated_at = now() "
    "WHERE tenant_id = %s AND job_id = %s"
)
_QUARANTINE = (
    "INSERT INTO quarantine_items (item_id, tenant_id, job_id, raw_sha256, reason, pii_class) "
    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (item_id) DO NOTHING"
)


class SqlJobStore:
    def __init__(self, dsn: str, *, role: str | None = "rag_app") -> None:
        self._dsn = dsn
        self._role = role

    async def create(self, job: IngestJob) -> bool:
        async with tx(self._dsn, role=self._role, tenant_id=job.tenant_id) as c:
            cur = await c.execute(_CREATE, (job.job_id, job.tenant_id, job.raw_sha256))
            return cur.rowcount == 1

    async def status(self, tenant_id: str, job_id: str) -> str | None:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            row = await (await c.execute(_STATUS, (tenant_id, job_id))).fetchone()
        return row["status"] if row else None

    async def set_status(
        self, tenant_id: str, job_id: str, status: str, *, error: str | None = None
    ) -> None:
        blocked = sorted(JobStatus.TERMINAL)
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            await c.execute(_SET, (status, error, tenant_id, job_id, blocked))

    async def doc_id(self, tenant_id: str, job_id: str, doc_id: str) -> None:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            await c.execute(_DOC, (doc_id, tenant_id, job_id))

    async def requeue(self, tenant_id: str, job_id: str) -> None:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            await c.execute(_REQUEUE, (tenant_id, job_id))

    async def quarantine(
        self, tenant_id: str, job_id: str, raw_sha256: str, reason: str, pii_class: str | None
    ) -> None:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            await c.execute(
                _QUARANTINE, (f"q-{job_id}", tenant_id, job_id, raw_sha256, reason, pii_class)
            )
