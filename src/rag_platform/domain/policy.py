"""Pipeline policy: the declared intent behind one index version (ADR-0006).

Pure domain: models, canonical hash, state machine and the ports the gate depends on.
"""

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from rag_platform.domain.errors import Conflict, ValidationFailed
from rag_platform.domain.ports import AuditEvent, AuditLog


class ChangeKind(StrEnum):
    EMBEDDER = "embedder"
    CHUNKER = "chunker"
    PARENT_CHILD = "parent_child"
    REDACTION = "redaction"
    RERANKER = "reranker"
    PROMPT = "prompt"
    CORPUS_ONLY = "corpus_only"


class PolicyStatus(StrEnum):
    DECLARED = "declared"
    VERIFIED = "verified"
    INDEXED = "indexed"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


# Kinds that change what retrieval returns: they need a passing eval run before commit.
RETRIEVAL_KINDS = frozenset(
    {
        ChangeKind.EMBEDDER,
        ChangeKind.CHUNKER,
        ChangeKind.PARENT_CHILD,
        ChangeKind.REDACTION,
        ChangeKind.RERANKER,
    }
)
# Kinds that need a linked, passing eval run before commit (prompt changes re-run evals too).
EVAL_REQUIRED_KINDS = RETRIEVAL_KINDS | {ChangeKind.PROMPT}
# Kinds that force a full rebuild: copy-forward of an earlier version's chunks is forbidden.
REBUILD_KINDS = RETRIEVAL_KINDS

# What the matching index_versions row holds after each policy state.
INDEX_STATUS: dict[PolicyStatus, str] = {
    PolicyStatus.DECLARED: "building",
    PolicyStatus.VERIFIED: "building",
    PolicyStatus.INDEXED: "staged",
    PolicyStatus.COMMITTED: "committed",
    PolicyStatus.ROLLED_BACK: "rolled_back",
    PolicyStatus.REJECTED: "rejected",
}

ALLOWED: dict[PolicyStatus, frozenset[PolicyStatus]] = {
    PolicyStatus.DECLARED: frozenset({PolicyStatus.VERIFIED, PolicyStatus.REJECTED}),
    PolicyStatus.VERIFIED: frozenset({PolicyStatus.INDEXED, PolicyStatus.REJECTED}),
    PolicyStatus.INDEXED: frozenset({PolicyStatus.COMMITTED, PolicyStatus.REJECTED}),
    PolicyStatus.COMMITTED: frozenset({PolicyStatus.ROLLED_BACK}),
    PolicyStatus.ROLLED_BACK: frozenset(),
    PolicyStatus.REJECTED: frozenset(),
}


class RejectReason(StrEnum):
    """Closed set of reject reasons: the only reason text that reaches audit rows."""

    VERIFY_FAILED = "verify_failed"
    EVAL_FAILED = "eval_failed"
    DATA_GATE_FAILED = "data_gate_failed"
    OPERATOR_REJECTED = "operator_rejected"


class IllegalTransition(Conflict):
    code = "conflict"


class PolicyRuleViolation(ValidationFailed):
    code = "validation_failed"


class PolicyDocument(BaseModel):
    model_config = ConfigDict(frozen=True)
    tenant_id: str
    doc_id: str
    doc_version: int
    doc_hash: str


class PolicyDraft(BaseModel):
    """The declared (hashed) fields. Immutable after declaration."""

    model_config = ConfigDict(frozen=True)
    change_kind: ChangeKind
    change_reason: str
    parent_policy_id: str | None = None
    ingest_run_id: str
    index_version: str
    embedder_id: str
    embedder_version: str
    embedder_dim: int
    chunker_version: str
    chunk_tokens: int
    chunk_overlap: int
    parent_tokens: int | None = None
    redaction_rules_version: str
    reranker_id: str | None = None
    prompt_version: str
    generator_chain_ref: str
    documents: tuple[PolicyDocument, ...] = ()

    @field_validator("change_reason")
    @classmethod
    def _canon(cls, v: str) -> str:
        """Unicode NFC, then strip edge whitespace; inner whitespace stays significant."""
        return unicodedata.normalize("NFC", v).strip()


class PipelinePolicy(PolicyDraft):
    """Draft plus lifecycle columns (the only part that changes after declaration)."""

    policy_id: str
    policy_hash: str
    status: PolicyStatus = PolicyStatus.DECLARED
    declared_by: str
    eval_run_id: str | None = None  # attached later; not hashed so the hash stays stable
    declared_at: datetime
    verified_at: datetime | None = None
    indexed_at: datetime | None = None
    committed_at: datetime | None = None
    rolled_back_at: datetime | None = None
    rejected_at: datetime | None = None
    mlflow_model_version: str | None = None
    mlflow_synced: bool = False


_HASH_FIELDS = frozenset(PolicyDraft.model_fields)


def _no_floats(x: Any) -> None:
    if isinstance(x, float):
        raise ValueError("floats are not allowed in a policy")
    if isinstance(x, dict):
        for v in x.values():
            _no_floats(v)
    elif isinstance(x, list):
        for v in x:
            _no_floats(v)


def canonical_json(policy: PolicyDraft) -> bytes:
    """Sorted keys, no whitespace, UTF-8, documents sorted, lifecycle fields excluded."""
    data = policy.model_dump(mode="json", include=set(_HASH_FIELDS))
    data["documents"] = sorted(
        data["documents"], key=lambda d: (d["tenant_id"], d["doc_id"], d["doc_version"])
    )
    _no_floats(data)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def policy_hash(policy: PolicyDraft) -> str:
    return hashlib.sha256(canonical_json(policy)).hexdigest()


def draft_of(policy: PipelinePolicy) -> PolicyDraft:
    return PolicyDraft(**{k: getattr(policy, k) for k in _HASH_FIELDS})


def hash_matches(policy: PipelinePolicy) -> bool:
    return policy_hash(draft_of(policy)) == policy.policy_hash


class PolicyRepository(Protocol):
    async def create(
        self,
        policy: PipelinePolicy,
        *,
        audit_event: AuditEvent | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        """Insert policy and its building index_versions row; Conflict on duplicate.

        The audit event is part of the same unit of work: if writing it fails, nothing is
        applied. The in-memory fake writes it through `audit_log`; a SQL adapter inserts the
        (hash-chained) audit row in the same transaction and ignores `audit_log`.
        """
        ...

    async def get(self, policy_id: str) -> PipelinePolicy | None: ...

    async def get_for_version(self, index_version: str) -> PipelinePolicy | None: ...

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
        """One atomic step: policy and index_versions row both move, or IllegalTransition.

        The optional audit event is written in the same unit of work: a failing audit write
        leaves status unchanged. The fake writes through `audit_log` inside its lock; a SQL
        adapter inserts the hash-chained audit row in the same transaction (ignoring `audit_log`).
        """
        ...

    async def list_unsynced(self, limit: int) -> list[PipelinePolicy]:
        """Unsynced policies ordered by committed_at (NULLS LAST) then declared_at.

        A SQL adapter must use the same ordering so promotions replay in commit order.
        """
        ...

    async def record_mlflow_version(self, policy_id: str, mlflow_model_version: str) -> None:
        """Persist the registered MLflow version right away; no status change, stays unsynced."""
        ...

    async def get_previous_committed(self, policy: PipelinePolicy) -> PipelinePolicy | None:
        """Most recent earlier committed policy of the lineage (via parent_policy_id).

        Ordered by committed_at; None when the lineage has no earlier committed policy.
        """
        ...

    async def mark_synced(self, policy_id: str, mlflow_model_version: str | None) -> None: ...


class RegistryMirror(Protocol):
    """Best-effort MLflow mirror. Implementations must never raise into callers."""

    async def register_candidate(self, policy: PipelinePolicy) -> str | None:
        """Create the registered-model version with alias candidate; its id, or None on failure."""
        ...

    async def promote_champion(self, policy: PipelinePolicy) -> bool: ...

    async def set_rolled_back(
        self, policy: PipelinePolicy, previous: PipelinePolicy | None
    ) -> bool:
        """Point the champion alias at `previous`'s MLflow version.

        Returns False (not synced, outbox retries) when `previous` is None or has no MLflow
        version: the champion alias is never deleted on unknown history.
        """
        ...


class EvalRunReader(Protocol):
    async def passed(self, eval_run_id: str) -> bool:
        """True only when the run exists and has no failed metric."""
        ...


def documents_sorted(docs: Sequence[PolicyDocument]) -> list[PolicyDocument]:
    return sorted(docs, key=lambda d: (d.tenant_id, d.doc_id, d.doc_version))
