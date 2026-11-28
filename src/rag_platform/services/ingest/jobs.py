"""Ingest job message, job states and the queue port. Pure; no I/O."""

import hashlib
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag_platform.domain.policy import PolicyRepository, PolicyStatus

MAX_ATTEMPTS = 1000


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"
    FAILED = "failed"

    TERMINAL = frozenset({"done", "quarantined", "rejected", "failed"})


class IngestJob(BaseModel):
    """What travels on the stream. Validated on both ends; never contains document text."""

    model_config = ConfigDict(frozen=True)
    job_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    tenant_id: str = Field(min_length=1, max_length=128)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filename: str = Field(min_length=1, max_length=255)
    acl_labels: tuple[str, ...] = Field(min_length=1, max_length=32)
    attempt: int = Field(default=0, ge=0, le=MAX_ATTEMPTS)  # new attempt = new staged version


def job_id_for(tenant_id: str, raw_sha256: str) -> str:
    """Same upload (tenant + sha256) -> same job id, so a re-submit can never add a second job."""
    return hashlib.sha256(f"{tenant_id}:{raw_sha256}".encode()).hexdigest()[:32]


def doc_id_for(raw_sha256: str) -> str:
    return f"doc-{raw_sha256[:16]}"


def index_version_for(job_id: str, attempt: int = 0) -> str:
    """Attempt 0 keeps the original name; a re-run after rejection or delete gets `-a<n>`."""
    return f"iv-{job_id}" if attempt == 0 else f"iv-{job_id}-a{attempt}"


@dataclass(frozen=True)
class QueuedJob:
    message_id: str
    job: IngestJob


class JobQueue(Protocol):
    async def publish(self, job: IngestJob) -> str: ...

    async def claim(
        self, consumer: str, *, block_ms: int = 0, min_idle_ms: int = 60_000
    ) -> QueuedJob | None:
        """Next job for this consumer: a stale pending one (crashed worker) first, then new."""
        ...

    async def ack(self, message_id: str) -> None: ...


class JobStore(Protocol):
    async def create(self, job: IngestJob) -> bool:
        """Insert as queued; False when the job already exists (idempotent submit)."""
        ...

    async def status(self, tenant_id: str, job_id: str) -> str | None: ...

    async def set_status(
        self, tenant_id: str, job_id: str, status: str, *, error: str | None = None
    ) -> None: ...

    async def doc_id(self, tenant_id: str, job_id: str, doc_id: str) -> None: ...

    async def requeue(self, tenant_id: str, job_id: str) -> None: ...

    async def quarantine(
        self, tenant_id: str, job_id: str, raw_sha256: str, reason: str, pii_class: str | None
    ) -> None:
        """Record one quarantine item per job (idempotent)."""
        ...


async def next_attempt(policies: PolicyRepository, job_id: str) -> int:
    """Attempt number for a re-run: resume a version still being built, else start a new one.

    A staged (indexed), committed or rejected version never resumes: a re-run after a delete or a
    failed verify must stage a fresh version. A crash redelivery keeps the message's attempt.
    """
    latest: int | None = None
    for n in range(MAX_ATTEMPTS + 1):
        if await policies.get_for_version(index_version_for(job_id, n)) is None:
            break
        latest = n
    if latest is None:
        return 0
    p = await policies.get_for_version(index_version_for(job_id, latest))
    if p is not None and p.status in (PolicyStatus.DECLARED, PolicyStatus.VERIFIED):
        return latest
    return min(latest + 1, MAX_ATTEMPTS)
