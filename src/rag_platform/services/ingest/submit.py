"""Accept an upload: size check, store raw bytes, create the job once, publish it.

Same bytes for the same tenant map to the same job id, so a re-submit creates no second job,
document or chunk. A failed or rejected job, or a document that was deleted, may be re-run.
"""

from dataclasses import dataclass
from typing import Protocol

from rag_platform.domain.documents import Document
from rag_platform.domain.errors import ValidationFailed
from rag_platform.domain.policy import PolicyRepository
from rag_platform.services.ingest.jobs import (
    IngestJob,
    JobQueue,
    JobStatus,
    JobStore,
    job_id_for,
    next_attempt,
)
from rag_platform.services.ingest.parsers import MAX_UPLOAD_BYTES
from rag_platform.services.ingest.reasons import Reason


class _Store(Protocol):
    async def put(self, tenant_id: str, data: bytes, mime: str) -> str: ...


class _Docs(Protocol):
    async def get_by_hash(self, tenant_id: str, doc_hash: str) -> Document | None: ...


@dataclass(frozen=True)
class Submitted:
    job_id: str
    raw_sha256: str
    duplicate: bool  # True: an earlier submit of these bytes already owns the job


async def submit_upload(
    *,
    store: _Store,
    jobs: JobStore,
    queue: JobQueue,
    docs: _Docs,
    tenant_id: str,
    filename: str,
    data: bytes,
    acl_labels: tuple[str, ...],
    mime: str = "application/octet-stream",
    max_bytes: int = MAX_UPLOAD_BYTES,
    policies: PolicyRepository | None = None,
) -> Submitted:
    if len(data) > max_bytes:
        raise ValidationFailed(Reason.FILE_TOO_LARGE.value)
    sha = await store.put(tenant_id, data, mime)
    job = IngestJob(
        job_id=job_id_for(tenant_id, sha),
        tenant_id=tenant_id,
        raw_sha256=sha,
        filename=filename,
        acl_labels=acl_labels,
    )
    created = await jobs.create(job)
    if not created:
        status = await jobs.status(tenant_id, job.job_id)
        doc = await docs.get_by_hash(tenant_id, sha)
        rerun = status in (JobStatus.FAILED, JobStatus.REJECTED) or (
            status in JobStatus.TERMINAL and doc is not None and doc.status == "retired"
        )
        if rerun:
            job = job.model_copy(
                update={"attempt": await next_attempt(policies, job.job_id) if policies else 0}
            )
            await jobs.requeue(tenant_id, job.job_id)
        elif status != JobStatus.QUEUED:
            return Submitted(job.job_id, sha, duplicate=True)
    await queue.publish(job)  # a duplicate message for a queued job is harmless: process() dedups
    return Submitted(job.job_id, sha, duplicate=not created and not rerun)
