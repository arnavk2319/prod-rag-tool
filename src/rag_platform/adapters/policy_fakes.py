"""In-memory PolicyRepository, mirror and eval reader for tests and the eval harness."""

import asyncio
from datetime import datetime

from rag_platform.domain.errors import Conflict, NotFound
from rag_platform.domain.policy import (
    ALLOWED,
    INDEX_STATUS,
    IllegalTransition,
    PipelinePolicy,
    PolicyStatus,
)
from rag_platform.domain.ports import AuditEvent, AuditLog

_TS = {
    PolicyStatus.VERIFIED: "verified_at",
    PolicyStatus.INDEXED: "indexed_at",
    PolicyStatus.COMMITTED: "committed_at",
    PolicyStatus.ROLLED_BACK: "rolled_back_at",
    PolicyStatus.REJECTED: "rejected_at",
}


class FakePolicyRepository:
    """Atomic under a lock: policy row and index_versions status move together."""

    def __init__(self) -> None:
        self._rows: dict[str, PipelinePolicy] = {}
        self.index_status: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def all(self) -> list[PipelinePolicy]:
        return list(self._rows.values())

    async def create(
        self,
        policy: PipelinePolicy,
        *,
        audit_event: AuditEvent | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        async with self._lock:
            if policy.policy_id in self._rows:
                raise Conflict("policy exists")
            if any(
                r.index_version == policy.index_version and r.policy_hash == policy.policy_hash
                for r in self._rows.values()
            ):
                raise Conflict("policy hash already declared for this index version")
            if audit_event is not None and audit_log is not None:
                await audit_log.record(audit_event)  # raises before anything is applied
            self._rows[policy.policy_id] = policy
            self.index_status[policy.index_version] = INDEX_STATUS[policy.status]

    async def get(self, policy_id: str) -> PipelinePolicy | None:
        return self._rows.get(policy_id)

    async def get_for_version(self, index_version: str) -> PipelinePolicy | None:
        hits = [r for r in self._rows.values() if r.index_version == index_version]
        return max(hits, key=lambda r: r.declared_at) if hits else None

    async def transition(
        self,
        policy_id: str,
        *,
        expected_status: PolicyStatus,
        expected_index_status: str,
        new_status: PolicyStatus,
        at: datetime,
        eval_run_id: str | None = None,
        audit_event: AuditEvent | None = None,
        audit_log: AuditLog | None = None,
    ) -> PipelinePolicy:
        async with self._lock:
            cur = self._rows.get(policy_id)
            if cur is None:
                raise NotFound("policy not found")
            if (
                cur.status is not expected_status
                or self.index_status.get(cur.index_version) != expected_index_status
                or new_status not in ALLOWED[cur.status]
            ):
                raise IllegalTransition(f"{cur.status.value} -> {new_status.value} not allowed")
            update: dict[str, object] = {"status": new_status, _TS[new_status]: at}
            if eval_run_id is not None:
                update["eval_run_id"] = eval_run_id
            if new_status in (PolicyStatus.COMMITTED, PolicyStatus.ROLLED_BACK):
                update["mlflow_synced"] = False
            nxt = cur.model_copy(update=update)
            if audit_event is not None and audit_log is not None:
                await audit_log.record(audit_event)  # raises before anything is applied
            self._rows[policy_id] = nxt
            self.index_status[cur.index_version] = INDEX_STATUS[new_status]
            return nxt

    async def list_unsynced(self, limit: int) -> list[PipelinePolicy]:
        rows = sorted(
            (r for r in self._rows.values() if not r.mlflow_synced),
            key=lambda r: (r.committed_at is None, r.committed_at or r.declared_at, r.declared_at),
        )
        return rows[:limit]

    async def record_mlflow_version(self, policy_id: str, mlflow_model_version: str) -> None:
        async with self._lock:
            cur = self._rows.get(policy_id)
            if cur is not None:
                self._rows[policy_id] = cur.model_copy(
                    update={"mlflow_model_version": mlflow_model_version}
                )

    async def get_previous_committed(self, policy: PipelinePolicy) -> PipelinePolicy | None:
        if policy.committed_at is None:
            return None
        found: list[PipelinePolicy] = []
        seen: set[str] = set()
        pid = policy.parent_policy_id
        while pid is not None and pid not in seen:
            seen.add(pid)
            anc = self._rows.get(pid)
            if anc is None:
                break
            if (
                anc.status is PolicyStatus.COMMITTED
                and anc.committed_at is not None
                and anc.committed_at <= policy.committed_at
            ):
                found.append(anc)
            pid = anc.parent_policy_id
        return max(found, key=lambda r: r.committed_at or r.declared_at) if found else None

    async def mark_synced(self, policy_id: str, mlflow_model_version: str | None) -> None:
        async with self._lock:
            cur = self._rows.get(policy_id)
            if cur is not None:
                self._rows[policy_id] = cur.model_copy(
                    update={"mlflow_synced": True, "mlflow_model_version": mlflow_model_version}
                )


class FakeRegistryMirror:
    def __init__(self, *, down: bool = False) -> None:
        self.down = down
        self.versions: dict[str, str] = {}  # policy_id -> model version
        self.aliases: dict[str, str] = {}  # alias -> model version
        self.register_calls = 0

    async def register_candidate(self, policy: PipelinePolicy) -> str | None:
        if self.down:
            return None
        self.register_calls += 1
        v = self.versions.setdefault(policy.policy_id, str(len(self.versions) + 1))
        self.aliases["candidate"] = v
        return v

    async def promote_champion(self, policy: PipelinePolicy) -> bool:
        if self.down or policy.policy_id not in self.versions:
            return False
        self.aliases["champion"] = self.versions[policy.policy_id]
        return True

    async def set_rolled_back(
        self, policy: PipelinePolicy, previous: PipelinePolicy | None
    ) -> bool:
        """Champion returns to `previous`; False (retry later) when it is unknown."""
        if self.down or previous is None or previous.mlflow_model_version is None:
            return False
        self.aliases["champion"] = previous.mlflow_model_version
        return True


class FakeEvalRuns:
    def __init__(self, passing: set[str] | None = None) -> None:
        self._passing = passing or set()

    async def passed(self, eval_run_id: str) -> bool:
        return eval_run_id in self._passing
