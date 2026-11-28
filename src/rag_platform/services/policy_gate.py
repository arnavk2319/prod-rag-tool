"""PolicyGate: declare intent before embedding, verify after, gate the commit (ADR-0006)."""

import logging
import uuid
from dataclasses import dataclass

from rag_platform.domain.errors import NotFound
from rag_platform.domain.policy import (
    EVAL_REQUIRED_KINDS,
    INDEX_STATUS,
    REBUILD_KINDS,
    EvalRunReader,
    IllegalTransition,
    PipelinePolicy,
    PolicyDraft,
    PolicyRepository,
    PolicyRuleViolation,
    PolicyStatus,
    RegistryMirror,
    RejectReason,
    policy_hash,
)
from rag_platform.domain.ports import AuditEvent, AuditLog, Clock

log = logging.getLogger(__name__)
SYSTEM_TENANT = "_system"


@dataclass(frozen=True)
class ChunkSummary:
    """What the job actually wrote: distinct (embedding_model, chunker_version) and row count."""

    pairs: frozenset[tuple[str, str]]
    rows: int
    dim: int  # embedding dimension of the written vectors; mismatch fails verify (mandatory)
    # (tenant_id, doc_id) that have at least one written chunk (mandatory)
    documents_with_rows: frozenset[tuple[str, str]]


class PolicyGate:
    def __init__(
        self,
        repo: PolicyRepository,
        audit: AuditLog,
        clock: Clock,
        mirror: RegistryMirror,
        evals: EvalRunReader | None = None,
    ) -> None:
        self._repo = repo
        self._audit = audit
        self._clock = clock
        self._mirror = mirror
        self._evals = evals

    async def _get(self, policy_id: str) -> PipelinePolicy:
        p = await self._repo.get(policy_id)
        if p is None:
            raise NotFound("policy not found")
        return p

    async def _move(
        self,
        policy_id: str,
        to: PolicyStatus,
        eval_run_id: str | None = None,
        *,
        action: str | None = None,
        actor: str = "",
        **extra: str,
    ) -> PipelinePolicy:
        cur = await self._get(policy_id)
        if eval_run_id:
            extra["eval_run_id"] = eval_run_id  # id only: lineage is visible outside the row
        event = self._event(action, cur, actor, **extra) if action else None
        return await self._repo.transition(
            policy_id,
            expected_status=cur.status,
            expected_index_status=INDEX_STATUS[cur.status],
            new_status=to,
            at=self._clock.now(),
            eval_run_id=eval_run_id,
            audit_event=event,
            audit_log=self._audit,
        )

    def _event(self, action: str, p: PipelinePolicy, actor: str, **extra: str) -> AuditEvent:
        detail = {
            "policy_id": p.policy_id,
            "policy_hash": p.policy_hash,
            "index_version": p.index_version,
            "change_kind": p.change_kind.value,
            **extra,
        }
        return AuditEvent(
            at=self._clock.now(),
            actor=actor,
            action=action,
            tenant_id=SYSTEM_TENANT,
            detail=detail,
        )

    async def declare(
        self, draft: PolicyDraft, *, declared_by: str, copy_forward: bool = False
    ) -> PipelinePolicy:
        if not draft.change_reason.strip():
            raise PolicyRuleViolation("change_reason is mandatory")
        if copy_forward and draft.change_kind in REBUILD_KINDS:
            raise PolicyRuleViolation(
                f"{draft.change_kind.value} change forces a full rebuild; copy-forward is forbidden"
            )
        policy = PipelinePolicy(
            **draft.model_dump(),
            policy_id=str(uuid.uuid4()),
            policy_hash=policy_hash(draft),
            declared_by=declared_by,
            declared_at=self._clock.now(),
        )
        await self._repo.create(
            policy,
            audit_event=self._event("policy.declare", policy, declared_by),
            audit_log=self._audit,
        )
        await sync_one(self._repo, self._mirror, policy)
        return policy

    async def verify(self, policy_id: str, summary: ChunkSummary) -> PipelinePolicy:
        p = await self._get(policy_id)
        problem = self._verify_problem(p, summary)
        if problem:
            await self.reject(policy_id, RejectReason.VERIFY_FAILED, actor="gate")
            raise PolicyRuleViolation(problem)
        return await self._move(policy_id, PolicyStatus.VERIFIED)

    @staticmethod
    def _verify_problem(p: PipelinePolicy, s: ChunkSummary) -> str | None:
        if not p.documents and s.rows == 0 and not s.pairs:
            return None  # explicit zero expected documents (all deleted): an empty version is valid
        if s.rows <= 0:
            return "no chunks were written"
        if len(s.pairs) != 1:
            return "mixed embedding models or chunker versions in one index version"
        (model, chunker) = next(iter(s.pairs))
        if model != p.embedder_id:
            return "written embedding model differs from the declared embedder"
        if chunker != p.chunker_version:
            return "written chunker version differs from the declared chunker"
        if s.dim != p.embedder_dim:
            return "written embedding dimension differs from the declared dimension"
        declared = {(d.tenant_id, d.doc_id) for d in p.documents}
        if declared - s.documents_with_rows:
            return "declared documents have no written chunks"
        return None

    async def mark_indexed(self, policy_id: str) -> PipelinePolicy:
        return await self._move(policy_id, PolicyStatus.INDEXED)

    async def mark_committed(
        self, policy_id: str, *, eval_run_id: str | None = None, actor: str = "commit-gate"
    ) -> PipelinePolicy:
        p = await self._get(policy_id)
        if p.status is not PolicyStatus.INDEXED:
            raise IllegalTransition("commit requires a verified and indexed policy")
        run = eval_run_id or p.eval_run_id
        if p.change_kind in EVAL_REQUIRED_KINDS:
            if not run:
                raise PolicyRuleViolation("retrieval-affecting change needs a linked eval run")
            if self._evals is None or not await self._evals.passed(run):
                raise PolicyRuleViolation("linked eval run did not pass")
        done = await self._move(
            policy_id, PolicyStatus.COMMITTED, eval_run_id=run, action="policy.commit", actor=actor
        )
        await sync_one(self._repo, self._mirror, done)
        return done

    async def mark_rolled_back(self, policy_id: str, *, actor: str = "operator") -> PipelinePolicy:
        done = await self._move(
            policy_id, PolicyStatus.ROLLED_BACK, action="policy.rollback", actor=actor
        )
        await sync_one(self._repo, self._mirror, done)
        return done

    async def reject(
        self, policy_id: str, reason: RejectReason, *, actor: str = "operator"
    ) -> PipelinePolicy:
        done = await self._move(
            policy_id,
            PolicyStatus.REJECTED,
            action="policy.reject",
            actor=actor,
            reason_code=RejectReason(reason).value,
        )
        return done


async def sync_one(repo: PolicyRepository, mirror: RegistryMirror, p: PipelinePolicy) -> bool:
    """Mirror one policy's current state; set mlflow_synced on success. Never raises."""
    try:
        version = p.mlflow_model_version
        if version is None:
            version = await mirror.register_candidate(p)
            if version is None:
                return False
            await repo.record_mlflow_version(p.policy_id, version)  # retries reuse it
        cur = p.model_copy(update={"mlflow_model_version": version})
        ok = True
        if p.status is PolicyStatus.COMMITTED:
            ok = await mirror.promote_champion(cur)
        elif p.status is PolicyStatus.ROLLED_BACK:
            ok = await mirror.set_rolled_back(cur, await repo.get_previous_committed(p))
        if ok:
            await repo.mark_synced(p.policy_id, version)
        return ok
    except Exception:  # noqa: BLE001 - best effort by design
        log.warning("mirror sync failed policy=%s", p.policy_id)
        return False


async def sync_pending(repo: PolicyRepository, mirror: RegistryMirror, limit: int = 50) -> int:
    """Outbox runner: retry unsynced policies. Never raises; returns the synced count."""
    try:
        pending = await repo.list_unsynced(limit)
    except Exception:  # noqa: BLE001
        return 0
    return sum([int(await sync_one(repo, mirror, p)) for p in pending])
