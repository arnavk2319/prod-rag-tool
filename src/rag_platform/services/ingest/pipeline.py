"""Ingest pipeline: parse, redact, parent/child chunk, embed children, stage an index version.

One job builds one staged version named from the job id (`iv-<job_id>`), so a redelivered job
resumes the same version instead of creating another. Flow (ADR-0002, ADR-0006):

  assess (parse + redact + PII checks) -> register document -> declare policy -> copy forward the
  unchanged documents of the base version (INSERT ... SELECT, no embedder) -> chunk + embed the new
  documents (children only, in batches) -> verify -> mark_indexed.

A change kind that affects embeddings or chunks (REBUILD_KINDS), or an embedder or chunker that
differs from the base version, forces a rebuild: nothing is copied and every active document is
re-parsed from its raw object and re-embedded. The alias, gate and rollback belong to T-011:
this module never touches `index_alias` or any earlier version.
No document text is logged; only ids, counts and reason codes.
"""

import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from rag_platform.domain.documents import DocRow, DocStatus, Document, Outcome
from rag_platform.domain.policy import (
    REBUILD_KINDS,
    ChangeKind,
    PipelinePolicy,
    PolicyDocument,
    PolicyDraft,
    PolicyRepository,
    PolicyRuleViolation,
    PolicyStatus,
    RejectReason,
)
from rag_platform.domain.ports import Chunk, Embedder, ParentChunk
from rag_platform.services.ingest.build import DocMeta, build_records
from rag_platform.services.ingest.chunker import OVERLAP_RATIO, Sizes, chunk_document
from rag_platform.services.ingest.jobs import (
    IngestJob,
    JobStatus,
    JobStore,
    doc_id_for,
    index_version_for,
)
from rag_platform.services.ingest.parsers import MAX_UPLOAD_BYTES
from rag_platform.services.ingest.quarantine import assess
from rag_platform.services.ingest.reasons import Outcome as IngestOutcome
from rag_platform.services.ingest.reasons import Reason
from rag_platform.services.policy_gate import ChunkSummary, PolicyGate

_log = structlog.get_logger(__name__)


class RawStore(Protocol):
    async def get(self, tenant_id: str, sha256: str) -> bytes | None: ...
    async def delete(self, tenant_id: str, sha256: str) -> bool: ...
    async def mark_quarantined(self, tenant_id: str, sha256: str) -> bool: ...


class Documents(Protocol):
    async def register(
        self,
        tenant_id: str,
        doc_hash: str,
        doc_id: str,
        title: str,
        *,
        acl_labels: Sequence[str] = ...,
        pii_class: str = ...,
    ) -> tuple[Document, Outcome]: ...
    async def set_status(self, tenant_id: str, doc_id: str, status: DocStatus) -> Document: ...
    async def list_active(self, tenant_id: str, doc_ids: Sequence[str]) -> list[DocRow]: ...


class IndexWriter(Protocol):
    async def ensure_partition(self, index_version: str) -> None: ...
    async def clear_version(self, index_version: str, tenants: Sequence[str]) -> None: ...
    async def copy_forward(
        self, from_version: str, to_version: str, tenant_id: str, doc_ids: Sequence[str]
    ) -> None: ...
    async def add_document(
        self, parents: Sequence[ParentChunk], chunks: Sequence[Chunk]
    ) -> None: ...
    async def summarize(self, index_version: str, tenants: Sequence[str]) -> ChunkSummary: ...
    async def live_version(self) -> str | None: ...
    def version_lock(self, index_version: str) -> AbstractAsyncContextManager[None]:
        """Mutual exclusion for one staged version across workers (released on crash)."""
        ...


class VersionRejected(Exception):
    """The staged version failed verify (or was rejected earlier). Carries only a reason code."""


@dataclass(frozen=True)
class PipelineConfig:
    embed_batch_size: int = 64
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    change_kind: ChangeKind = ChangeKind.CORPUS_ONLY
    redaction_rules_version: str = "redaction-v1"
    embedder_version: str = "1"
    prompt_version: str = "none"  # ingest does not use a prompt
    generator_chain_ref: str = "none"
    sizes: Sizes | None = None
    actor: str = "ingest-worker"


class BatchedEmbedder:
    """Splits one large embed call into batches of `size`; same model id."""

    def __init__(self, inner: Embedder, size: int) -> None:
        self._inner = inner
        self._size = max(1, size)
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        out: list[tuple[float, ...]] = []
        for i in range(0, len(texts), self._size):
            self.calls += 1
            out += await self._inner.embed(texts[i : i + self._size])
        return out


@dataclass
class _Plan:
    policy: PipelinePolicy
    base_version: str | None
    copy: bool
    carried: dict[str, list[str]] = field(default_factory=dict)  # tenant -> doc ids to copy
    to_embed: list[tuple[DocRow, str]] = field(default_factory=list)


class IngestPipeline:
    def __init__(
        self,
        *,
        store: RawStore,
        docs: Documents,
        jobs: JobStore,
        writer: IndexWriter,
        gate: PolicyGate,
        policies: PolicyRepository,
        embedder: Embedder,
        config: PipelineConfig | None = None,
        base_version: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        self._store = store
        self._docs = docs
        self._jobs = jobs
        self._writer = writer
        self._gate = gate
        self._policies = policies
        self._embedder = embedder
        self._cfg = config or PipelineConfig()
        self._sizes = self._cfg.sizes or Sizes.from_env()
        self._base_version = base_version or writer.live_version

    # ---- public --------------------------------------------------------------------------

    async def process(self, job: IngestJob) -> str:
        """Run one job to a terminal status. Idempotent: a terminal job is returned untouched.

        Exceptions (not BaseException) end in `failed` with the exception class name only. A kill
        leaves the job `running`; the redelivered message resumes the same staged version.
        """
        status = await self._jobs.status(job.tenant_id, job.job_id)
        if status is None:
            await self._jobs.create(job)
        elif status in JobStatus.TERMINAL:
            return status
        await self._jobs.set_status(job.tenant_id, job.job_id, JobStatus.RUNNING)
        try:
            final, error = await self._run(job)
        except Exception as e:  # noqa: BLE001 - recorded, never raised into the worker loop
            final, error = JobStatus.FAILED, type(e).__name__
            _log.warning("ingest_job_failed", job_id=job.job_id, error=error)
        await self._jobs.set_status(job.tenant_id, job.job_id, final, error=error)
        _log.info("ingest_job_finished", job_id=job.job_id, status=final, reason=error)
        return final

    async def delete_document(self, tenant_id: str, doc_id: str) -> None:
        """Retire the document and remove its raw object. The next `reindex` omits it."""
        doc = await self._docs.set_status(tenant_id, doc_id, "retired")
        await self._store.delete(tenant_id, doc.doc_hash)

    async def reindex(self, run_id: str | None = None) -> str:
        """Stage a new version from the base version without retired documents.

        Pass the same `run_id` after a crash to resume. Raises VersionRejected on failed verify.
        """
        version = index_version_for(run_id or uuid.uuid4().hex)
        await self._stage(version, f"reindex {version}", [])
        return version

    # ---- job ----------------------------------------------------------------------------

    async def _run(self, job: IngestJob) -> tuple[str, str | None]:
        data = await self._store.get(job.tenant_id, job.raw_sha256)
        if data is None:
            return JobStatus.FAILED, "raw_object_missing"
        decision = assess(data, job.filename, max_bytes=self._cfg.max_upload_bytes)
        if decision.outcome is IngestOutcome.QUARANTINED:
            reason = decision.reason.value if decision.reason else "unknown"
            await self._store.mark_quarantined(job.tenant_id, job.raw_sha256)
            pii = "dense" if decision.reason is Reason.PII_DENSITY_HIGH else None
            await self._jobs.quarantine(job.tenant_id, job.job_id, job.raw_sha256, reason, pii)
            return JobStatus.QUARANTINED, reason
        if decision.outcome is IngestOutcome.REJECTED:
            await self._store.delete(job.tenant_id, job.raw_sha256)  # refused bytes are not kept
            return JobStatus.REJECTED, decision.reason.value if decision.reason else "rejected"
        pii_class = "redacted" if decision.reason is Reason.PII_REDACTED else "none"
        doc, _outcome = await self._docs.register(
            job.tenant_id, job.raw_sha256, doc_id_for(job.raw_sha256), job.filename,
            acl_labels=job.acl_labels, pii_class=pii_class,
        )  # fmt: skip
        await self._jobs.doc_id(job.tenant_id, job.job_id, doc.doc_id)
        rows = await self._docs.list_active(job.tenant_id, [doc.doc_id])
        if not rows:
            return JobStatus.FAILED, "document_not_active"
        try:
            await self._stage(
                index_version_for(job.job_id, job.attempt),
                f"ingest job {job.job_id}",
                [(rows[0], decision.text)],
            )
        except VersionRejected:
            return JobStatus.REJECTED, RejectReason.VERIFY_FAILED.value
        return JobStatus.DONE, None

    # ---- staging ------------------------------------------------------------------------

    async def _plan(self, version: str, reason: str, new: list[tuple[DocRow, str]]) -> _Plan:
        existing = await self._policies.get_for_version(version)
        new_keys = {(r.doc.tenant_id, r.doc.doc_id) for r, _ in new}
        if existing is not None:  # resume after a crash: same declared policy and documents
            if existing.status is PolicyStatus.REJECTED:
                raise VersionRejected
            base_policy = (
                await self._policies.get(existing.parent_policy_id)
                if existing.parent_policy_id
                else None
            )
            source = existing.documents
            kind = existing.change_kind
        else:
            base = await self._base_version()
            base_policy = await self._policies.get_for_version(base) if base else None
            if base and base_policy is None:
                raise RuntimeError("base_version_without_policy")
            source = base_policy.documents if base_policy else ()
            kind = self._effective_kind(base_policy)
        by_tenant: dict[str, list[str]] = defaultdict(list)
        for d in source:
            if (d.tenant_id, d.doc_id) not in new_keys:
                by_tenant[d.tenant_id].append(d.doc_id)
        active: list[DocRow] = []
        for t in sorted(by_tenant):
            active += await self._docs.list_active(t, by_tenant[t])
        copy = base_policy is not None and kind not in REBUILD_KINDS
        plan = _Plan(
            policy=existing,  # type: ignore[arg-type]  # replaced below when declaring
            base_version=base_policy.index_version if base_policy else None,
            copy=copy,
        )
        if copy:
            for r in active:
                plan.carried.setdefault(r.doc.tenant_id, []).append(r.doc.doc_id)
            plan.to_embed = list(new)
        else:
            plan.to_embed = list(new) + [(r, await self._text_of(r)) for r in active]
        if existing is None:
            everything = [r for r, _ in new] + active
            plan.policy = await self._gate.declare(
                self._draft(version, reason, kind, base_policy, everything),
                declared_by=self._cfg.actor,
                copy_forward=copy,
            )
        return plan

    def _effective_kind(self, base: PipelinePolicy | None) -> ChangeKind:
        kind = self._cfg.change_kind
        if base is not None and kind not in REBUILD_KINDS:
            if base.embedder_id != self._embedder.model_id:
                return ChangeKind.EMBEDDER
            if base.chunker_version != self._sizes.version:
                return ChangeKind.CHUNKER
        return kind

    def _draft(
        self,
        version: str,
        reason: str,
        kind: ChangeKind,
        base: PipelinePolicy | None,
        docs: list[DocRow],
    ) -> PolicyDraft:
        return PolicyDraft(
            change_kind=kind,
            change_reason=reason,
            parent_policy_id=base.policy_id if base else None,
            ingest_run_id=version.removeprefix("iv-"),
            index_version=version,
            embedder_id=self._embedder.model_id,
            embedder_version=self._cfg.embedder_version,
            embedder_dim=self._dim,
            chunker_version=self._sizes.version,
            chunk_tokens=self._sizes.child,
            chunk_overlap=round(self._sizes.child * OVERLAP_RATIO),
            parent_tokens=self._sizes.parent,
            redaction_rules_version=self._cfg.redaction_rules_version,
            prompt_version=self._cfg.prompt_version,
            generator_chain_ref=self._cfg.generator_chain_ref,
            documents=tuple(
                PolicyDocument(
                    tenant_id=r.doc.tenant_id,
                    doc_id=r.doc.doc_id,
                    doc_version=r.doc.version,
                    doc_hash=r.doc.doc_hash,
                )
                for r in docs
            ),
        )

    @property
    def _dim(self) -> int:
        return int(getattr(self._embedder, "dim", 0)) or 0

    async def _text_of(self, row: DocRow) -> str:
        """Rebuild path: re-parse and re-redact the raw object of an already registered document."""
        data = await self._store.get(row.doc.tenant_id, row.doc.doc_hash)
        if data is None:
            raise RuntimeError("raw_object_missing")
        d = assess(data, row.doc.title, max_bytes=self._cfg.max_upload_bytes)
        if d.outcome is not IngestOutcome.ACCEPTED:
            raise RuntimeError("document_no_longer_accepted")
        return d.text

    async def _stage(self, version: str, reason: str, new: list[tuple[DocRow, str]]) -> None:
        if not self._dim:  # embedder without a `dim` attribute: probe once, before declaring
            probe = await self._embedder.embed(["dimension probe"])
            self._embedder = _WithDim(self._embedder, len(probe[0]))
        async with self._writer.version_lock(version):
            await self._stage_locked(version, reason, new)

    async def _stage_locked(self, version: str, reason: str, new: list[tuple[DocRow, str]]) -> None:
        plan = await self._plan(version, reason, new)
        policy = plan.policy
        if policy.status in (PolicyStatus.INDEXED, PolicyStatus.COMMITTED):
            return  # already staged by an earlier delivery
        tenants = sorted({d.tenant_id for d in policy.documents})
        await self._writer.ensure_partition(version)
        await self._writer.clear_version(version, tenants)  # a retry starts from zero rows
        if plan.copy and plan.base_version:
            for t in sorted(plan.carried):
                await self._writer.copy_forward(plan.base_version, version, t, plan.carried[t])
        batched = BatchedEmbedder(self._embedder, self._cfg.embed_batch_size)
        for row, text in plan.to_embed:
            chunked = chunk_document(row.doc.doc_hash, text, self._sizes)
            meta = DocMeta(
                tenant_id=row.doc.tenant_id,
                doc_id=row.doc.doc_id,
                doc_hash=row.doc.doc_hash,
                index_version=version,
                acl_labels=row.acl_labels,
                effective_date=row.effective_date,
                doc_version=row.doc.version,
                expires_date=row.expires_date,
            )
            parents, chunks = await build_records(chunked, meta, batched)
            await self._writer.add_document(parents, chunks)
        summary = await self._writer.summarize(version, tenants)
        if policy.status is PolicyStatus.DECLARED:
            try:
                await self._gate.verify(policy.policy_id, summary)
            except PolicyRuleViolation as e:  # gate already recorded `rejected`; alias untouched
                raise VersionRejected(RejectReason.VERIFY_FAILED.value) from e
        await self._gate.mark_indexed(policy.policy_id)


class _WithDim:
    """Embedder wrapper that adds a probed `dim`."""

    def __init__(self, inner: Embedder, dim: int) -> None:
        self._inner = inner
        self.dim = dim

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return await self._inner.embed(texts)
