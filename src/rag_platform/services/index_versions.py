"""Index version commit: data gate, one atomic alias swap, rollback, retention (ADR-0002, D-09).

`commit(version)`: refuse unless the policy is `indexed`; run the data gate and write a
`gate_runs` row; on failure reject the version and leave `index_alias` untouched; on success
swap the alias and call `PolicyGate.mark_committed()` in ONE transaction (the policy repository
is lent that connection), then mirror to the registry after the commit. `rollback()` points the
alias back at `parent_version` and calls `mark_rolled_back()` in the same transaction.
"""

from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

from rag_platform.domain.errors import Conflict, NotFound
from rag_platform.domain.policy import (
    PipelinePolicy,
    PolicyRuleViolation,
    PolicyStatus,
    RejectReason,
)
from rag_platform.services.policy_gate import PolicyGate

log = structlog.get_logger(__name__)

EXPECTED_DIM = 384
SELF_RETRIEVAL_MIN = 0.98
SELF_RETRIEVAL_SAMPLE = 500
KEEP_VERSIONS = 3


@dataclass
class GateMeasures:
    """Raw counts from one gate pass (no text, no document content)."""

    rows_actual: int = 0
    rows_recorded: int | None = None
    bad_dim: int = 0
    nonfinite: int = 0
    zero_vectors: int = 0
    self_total: int = 0
    self_hits: int = 0
    orphan_children: int = 0
    parent_rows: int = 0
    parents_referenced: int = 0
    unlisted_chunk_rows: int = 0  # rows of tenants not in the policy (owner connection)
    unlisted_parent_rows: int = 0


class IndexVersionStore(Protocol):
    """What the service needs from storage (Postgres adapter: SqlIndexVersionStore).

    `c` is the connection of the open alias transaction, opaque to the service.
    """

    async def measure(
        self, index_version: str, tenants: Iterable[str], *, dim: int, sample: int
    ) -> GateMeasures: ...
    async def record_gate_run(
        self, index_version: str, passed: bool, metrics: dict[str, Any]
    ) -> str: ...
    def alias_tx(self) -> AbstractAsyncContextManager[Any]: ...
    async def live(self, c: Any) -> tuple[str, str | None] | None: ...
    async def version_row(self, c: Any, index_version: str) -> dict[str, Any] | None: ...
    async def has_partition(self, c: Any, index_version: str) -> bool: ...
    async def set_alias(self, c: Any, index_version: str, previous: str | None) -> None: ...
    async def retain(self, keep: int) -> list[str]: ...


class BindablePolicies(Protocol):
    """PolicyRepository whose calls can join a caller's transaction (SqlPolicyRepository)."""

    async def get_for_version(self, index_version: str) -> PipelinePolicy | None: ...
    def bind(self, conn: Any) -> AbstractAsyncContextManager[None]: ...


class GateFailed(Conflict):
    """The data gate failed; the alias did not move. `details` holds the metrics (counts only)."""


@dataclass(frozen=True)
class GateResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...] = field(default_factory=tuple)
    gate_run_id: str = ""


def evaluate(m: GateMeasures, *, expected_rows: int | None = None) -> GateResult:
    """Pure decision over the measured counts."""
    ratio = 1.0 if m.self_total == 0 else m.self_hits / m.self_total
    violations = m.bad_dim + m.nonfinite + m.zero_vectors
    metrics: dict[str, Any] = {
        "rows_actual": m.rows_actual,
        "rows_recorded": m.rows_recorded,
        "vector_violations": violations,
        "bad_dim": m.bad_dim,
        "nonfinite": m.nonfinite,
        "zero_vectors": m.zero_vectors,
        "self_retrieval": round(ratio, 4),
        "self_retrieval_total": m.self_total,
        "orphan_children": m.orphan_children,
        "parent_rows": m.parent_rows,
        "parents_referenced": m.parents_referenced,
        "unlisted_tenant_rows": m.unlisted_chunk_rows + m.unlisted_parent_rows,
    }
    failures: list[str] = []
    if m.rows_recorded is None or m.rows_recorded != m.rows_actual:
        failures.append("row_parity")
    if violations:
        failures.append("vector_sanity")
    if ratio < SELF_RETRIEVAL_MIN:
        failures.append("self_retrieval")
    if m.orphan_children:
        failures.append("orphan_children")
    if m.unlisted_chunk_rows or m.unlisted_parent_rows:
        failures.append("unlisted_tenant_rows")
    if m.parent_rows != m.parents_referenced:
        failures.append("parent_row_parity")
    return GateResult(not failures, metrics, tuple(failures))


class IndexVersionService:
    def __init__(
        self,
        store: IndexVersionStore,
        policies: BindablePolicies,
        gate: PolicyGate,
        *,
        on_swap: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._policies = policies
        self._gate = gate
        self._on_swap = on_swap  # test seam: runs inside the swap transaction, before commit

    async def run_gate(self, index_version: str, policy: PipelinePolicy) -> GateResult:
        tenants = {d.tenant_id for d in policy.documents}
        m = await self._store.measure(
            index_version, tenants, dim=EXPECTED_DIM, sample=SELF_RETRIEVAL_SAMPLE
        )
        res = evaluate(m)
        run_id = await self._store.record_gate_run(
            index_version, res.passed, {**res.metrics, "failures": list(res.failures)}
        )
        log.info("index.gate", index_version=index_version, passed=res.passed, gate_run_id=run_id,
                 failures=list(res.failures))  # fmt: skip
        return GateResult(res.passed, res.metrics, res.failures, run_id)

    async def commit(self, index_version: str, *, actor: str = "commit-gate") -> GateResult:
        policy = await self._policies.get_for_version(index_version)
        if policy is None:
            raise NotFound("index version has no policy")
        if policy.status is not PolicyStatus.INDEXED:
            raise PolicyRuleViolation("commit requires a policy in status indexed")
        res = await self.run_gate(index_version, policy)
        if not res.passed:
            await self._gate.reject(policy.policy_id, RejectReason.DATA_GATE_FAILED, actor="gate")
            raise GateFailed(
                "index data gate failed: " + ",".join(res.failures),
                details={"gate_run_id": res.gate_run_id, **res.metrics},
            )
        async with self._store.alias_tx() as c, self._policies.bind(c):
            row = await self._store.version_row(c, index_version)
            if row is None:
                raise NotFound("index version not found")
            live = await self._store.live(c)
            if row["parent_version"] != (live[0] if live else None):
                raise Conflict("the live version moved since this version was built")
            done = await self._gate.mark_committed(policy.policy_id, actor=actor, sync=False)
            await self._store.set_alias(c, index_version, live[0] if live else None)
            if self._on_swap is not None:
                await self._on_swap()
        await self._gate.resync(done)
        log.info("index.commit", index_version=index_version, previous=live[0] if live else None)
        return res

    async def rollback(self, *, actor: str = "operator") -> str:
        """Point the alias at the live version's parent; returns the version now live."""
        async with self._store.alias_tx() as c, self._policies.bind(c):
            live = await self._store.live(c)
            if live is None:
                raise Conflict("no live version to roll back")
            row = await self._store.version_row(c, live[0])
            target = row["parent_version"] if row else None
            if target is None:
                raise Conflict("the live version has no parent to roll back to")
            prev = await self._store.version_row(c, target)
            if prev is None or prev["status"] != "committed":
                raise Conflict("the parent version is not a committed version")
            if not await self._store.has_partition(c, target):
                raise Conflict("the parent version was already removed by retention")
            policy = await self._policies.get_for_version(live[0])
            if policy is None:
                raise NotFound("live version has no policy")
            done = await self._gate.mark_rolled_back(policy.policy_id, actor=actor, sync=False)
            await self._store.set_alias(c, target, live[0])
        await self._gate.resync(done)
        log.info("index.rollback", from_version=live[0], to_version=target)
        return str(target)

    async def retain(self, keep: int = KEEP_VERSIONS) -> list[str]:
        removed = await self._store.retain(keep)
        log.info("index.retention", removed=len(removed))
        return removed
