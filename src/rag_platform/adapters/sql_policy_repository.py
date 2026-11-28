"""PolicyRepository over Postgres (psycopg3, ADR-0006).

One connection and one transaction per call. A transition updates the policy row and its
index_versions row together and appends the hash-chained audit row (SQL function
`audit_append`, migration 0003) in the same transaction: any failure rolls everything back.
Policy documents sit behind RLS, so they are written and read tenant by tenant.
"""

import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, LiteralString

import psycopg
from psycopg import sql
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rag_platform.domain.errors import Conflict, NotFound
from rag_platform.domain.policy import (
    ALLOWED,
    INDEX_STATUS,
    IllegalTransition,
    PipelinePolicy,
    PolicyDocument,
    PolicyStatus,
)
from rag_platform.domain.ports import AuditEvent, AuditLog

_TS_COLUMN = {
    PolicyStatus.VERIFIED: "verified_at",
    PolicyStatus.INDEXED: "indexed_at",
    PolicyStatus.COMMITTED: "committed_at",
    PolicyStatus.ROLLED_BACK: "rolled_back_at",
    PolicyStatus.REJECTED: "rejected_at",
}
_RESYNC = frozenset({PolicyStatus.COMMITTED, PolicyStatus.ROLLED_BACK})

_COLS: LiteralString = (
    "policy_id::text AS policy_id, policy_hash, ingest_run_id, index_version, "
    "parent_policy_id::text AS parent_policy_id, change_kind, change_reason, embedder_id, "
    "embedder_version, embedder_dim, chunker_version, chunk_tokens, chunk_overlap, parent_tokens, "
    "redaction_rules_version, reranker_id, prompt_version, generator_chain_ref, doc_tenants, "
    "eval_run_id, status, declared_by, declared_at, verified_at, indexed_at, committed_at, "
    "rolled_back_at, rejected_at, mlflow_model_version, mlflow_synced"
)

_SET_TENANT: LiteralString = "SELECT set_config('app.tenant_id', %s, true)"

_UPSERT_INDEX: LiteralString = (
    "INSERT INTO index_versions (index_version, parent_version, embedding_model, chunker_version, "
    "status) VALUES (%s, %s, %s, %s, 'building') "
    "ON CONFLICT (index_version) DO UPDATE SET status = 'building', "
    "embedding_model = EXCLUDED.embedding_model, chunker_version = EXCLUDED.chunker_version "
    "WHERE index_versions.status IN ('building', 'rejected')"
)
_PARENT_VERSION: LiteralString = (
    "SELECT index_version FROM pipeline_policies WHERE policy_id = %s::uuid"
)
_INSERT_POLICY: LiteralString = (
    "INSERT INTO pipeline_policies (policy_id, policy_hash, ingest_run_id, index_version, "
    "parent_policy_id, change_kind, change_reason, embedder_id, embedder_version, embedder_dim, "
    "chunker_version, chunk_tokens, chunk_overlap, parent_tokens, redaction_rules_version, "
    "reranker_id, prompt_version, generator_chain_ref, doc_tenants, eval_run_id, status, "
    "declared_by, declared_at, verified_at, indexed_at, committed_at, rolled_back_at, "
    "rejected_at, mlflow_model_version, mlflow_synced) VALUES (%s::uuid, %s, %s, %s, %s::uuid, "
    "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "%s, %s, %s)"
)
_STAMP_INDEX: LiteralString = (
    "UPDATE index_versions SET policy_id = %s::uuid, policy_hash = %s WHERE index_version = %s"
)
_INSERT_DOC: LiteralString = (
    "INSERT INTO policy_documents (policy_id, tenant_id, doc_id, doc_version, doc_hash) "
    "VALUES (%s::uuid, %s, %s, %s, %s)"
)
_SELECT_DOCS: LiteralString = (
    "SELECT tenant_id, doc_id, doc_version, doc_hash FROM policy_documents "
    "WHERE policy_id = %s::uuid ORDER BY tenant_id, doc_id, doc_version"
)
_GET: LiteralString = "SELECT __COLS__ FROM pipeline_policies WHERE policy_id = %s::uuid".replace(
    "__COLS__", _COLS
)
_GET_VERSION: LiteralString = (
    "SELECT __COLS__ FROM pipeline_policies WHERE index_version = %s "
    "ORDER BY declared_at DESC, policy_id DESC LIMIT 1"
).replace("__COLS__", _COLS)
_UNSYNCED: LiteralString = (
    "SELECT __COLS__ FROM pipeline_policies WHERE NOT mlflow_synced "
    "ORDER BY committed_at NULLS LAST, declared_at, policy_id LIMIT %s"
).replace("__COLS__", _COLS)
_MOVE_TEMPLATE: LiteralString = (
    "UPDATE pipeline_policies SET status = %s, __TS__ = %s, "
    "eval_run_id = coalesce(%s, eval_run_id), "
    "mlflow_synced = CASE WHEN %s THEN false ELSE mlflow_synced END "
    "WHERE policy_id = %s::uuid AND status = %s RETURNING __COLS__"
).replace("__COLS__", _COLS)
_MOVE: dict[PolicyStatus, LiteralString] = {
    st: _MOVE_TEMPLATE.replace("__TS__", col) for st, col in _TS_COLUMN.items()
}
_AUDIT: LiteralString = "SELECT audit_append(%s, %s, %s, %s, %s, %s)"

_PREVIOUS: LiteralString = (
    "WITH RECURSIVE lineage AS ("
    " SELECT policy_id, parent_policy_id FROM pipeline_policies WHERE policy_id = %s::uuid"
    " UNION"
    " SELECT p.policy_id, p.parent_policy_id FROM pipeline_policies p"
    " JOIN lineage l ON p.policy_id = l.parent_policy_id) "
    "SELECT __COLS__ FROM pipeline_policies WHERE policy_id IN "
    "(SELECT policy_id FROM lineage) AND policy_id <> %s::uuid AND status = 'committed' "
    "AND committed_at IS NOT NULL AND committed_at <= %s "
    "ORDER BY committed_at DESC, declared_at DESC LIMIT 1"
).replace("__COLS__", _COLS)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


class SqlPolicyRepository:
    """`role` is applied with SET ROLE on every connection (tests and workers use `rag_app`)."""

    def __init__(self, dsn: str, *, role: str | None = "rag_app") -> None:
        self._dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
        self._role = role

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator[psycopg.AsyncConnection[dict[str, Any]]]:
        async with await psycopg.AsyncConnection.connect(
            self._dsn, row_factory=dict_row
        ) as conn:  # commits on success, rolls back on any exception
            if self._role:
                await conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(self._role)))
            yield conn

    @staticmethod
    async def _audit(conn: psycopg.AsyncConnection[dict[str, Any]], ev: AuditEvent) -> None:
        target = ev.detail.get("policy_id")
        await conn.execute(
            _AUDIT,
            (ev.at, ev.tenant_id, ev.actor, ev.action, target, Jsonb(ev.detail)),
        )

    @staticmethod
    async def _hydrate(
        conn: psycopg.AsyncConnection[dict[str, Any]], rows: list[dict[str, Any]]
    ) -> list[PipelinePolicy]:
        out: list[PipelinePolicy] = []
        for r in rows:
            docs: list[PolicyDocument] = []
            for tenant in r["doc_tenants"]:
                await conn.execute(_SET_TENANT, (tenant,))
                cur = await conn.execute(_SELECT_DOCS, (r["policy_id"],))
                docs += [PolicyDocument(**d) for d in await cur.fetchall()]
            data = {k: v for k, v in r.items() if k != "doc_tenants"}
            out.append(PipelinePolicy(**data, documents=tuple(docs)))
        return out

    async def create(
        self,
        policy: PipelinePolicy,
        *,
        audit_event: AuditEvent | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        by_tenant: dict[str, list[PolicyDocument]] = defaultdict(list)
        for d in policy.documents:
            by_tenant[d.tenant_id].append(d)
        try:
            async with self._tx() as conn:
                parent_version = None
                if policy.parent_policy_id is not None:
                    cur = await conn.execute(_PARENT_VERSION, (policy.parent_policy_id,))
                    row = await cur.fetchone()
                    parent_version = row["index_version"] if row else None
                cur = await conn.execute(
                    _UPSERT_INDEX,
                    (policy.index_version, parent_version, policy.embedder_id,
                     policy.chunker_version),
                )  # fmt: skip
                if cur.rowcount == 0:
                    raise Conflict("index version is not open for a new policy")
                await conn.execute(
                    _INSERT_POLICY,
                    (
                        policy.policy_id, policy.policy_hash, policy.ingest_run_id,
                        policy.index_version, policy.parent_policy_id,
                        policy.change_kind.value, policy.change_reason, policy.embedder_id,
                        policy.embedder_version, policy.embedder_dim, policy.chunker_version,
                        policy.chunk_tokens, policy.chunk_overlap, policy.parent_tokens,
                        policy.redaction_rules_version, policy.reranker_id,
                        policy.prompt_version, policy.generator_chain_ref, sorted(by_tenant),
                        policy.eval_run_id, policy.status.value, policy.declared_by,
                        policy.declared_at, policy.verified_at, policy.indexed_at,
                        policy.committed_at, policy.rolled_back_at, policy.rejected_at,
                        policy.mlflow_model_version, policy.mlflow_synced,
                    ),
                )  # fmt: skip
                for tenant in sorted(by_tenant):
                    await conn.execute(_SET_TENANT, (tenant,))
                    for d in by_tenant[tenant]:
                        await conn.execute(
                            _INSERT_DOC,
                            (policy.policy_id, tenant, d.doc_id, d.doc_version, d.doc_hash),
                        )
                await conn.execute(
                    _STAMP_INDEX, (policy.policy_id, policy.policy_hash, policy.index_version)
                )
                if audit_event is not None:
                    await self._audit(conn, audit_event)
        except UniqueViolation as e:
            raise Conflict("policy already declared for this index version") from e

    async def get(self, policy_id: str) -> PipelinePolicy | None:
        if not _is_uuid(policy_id):
            return None
        async with self._tx() as conn:
            cur = await conn.execute(_GET, (policy_id,))
            rows = await cur.fetchall()
            return (await self._hydrate(conn, rows))[0] if rows else None

    async def get_for_version(self, index_version: str) -> PipelinePolicy | None:
        async with self._tx() as conn:
            cur = await conn.execute(_GET_VERSION, (index_version,))
            rows = await cur.fetchall()
            return (await self._hydrate(conn, rows))[0] if rows else None

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
        if not _is_uuid(policy_id):
            raise NotFound("policy not found")
        async with self._tx() as conn:
            rows: list[dict[str, Any]] = []
            if new_status in ALLOWED[expected_status]:
                cur = await conn.execute(
                    _MOVE[new_status],
                    (new_status.value, at, eval_run_id, new_status in _RESYNC, policy_id,
                     expected_status.value),
                )  # fmt: skip
                rows = await cur.fetchall()
            if not rows:
                cur = await conn.execute(
                    "SELECT 1 FROM pipeline_policies WHERE policy_id = %s::uuid", (policy_id,)
                )
                if await cur.fetchone() is None:
                    raise NotFound("policy not found")
                raise IllegalTransition(
                    f"{expected_status.value} -> {new_status.value} not allowed"
                )
            cur = await conn.execute(
                "UPDATE index_versions SET status = %s, "
                "committed_at = CASE WHEN %s THEN %s ELSE committed_at END "
                "WHERE index_version = %s AND status = %s",
                (INDEX_STATUS[new_status], new_status is PolicyStatus.COMMITTED, at,
                 rows[0]["index_version"], expected_index_status),
            )  # fmt: skip
            if cur.rowcount == 0:  # raising rolls the policy update back with it
                raise IllegalTransition(
                    f"index status is not {expected_index_status}: {new_status.value} refused"
                )
            if audit_event is not None:
                await self._audit(conn, audit_event)
            return (await self._hydrate(conn, rows))[0]

    async def list_unsynced(self, limit: int) -> list[PipelinePolicy]:
        async with self._tx() as conn:
            cur = await conn.execute(_UNSYNCED, (limit,))
            return await self._hydrate(conn, await cur.fetchall())

    async def record_mlflow_version(self, policy_id: str, mlflow_model_version: str) -> None:
        if not _is_uuid(policy_id):
            return
        async with self._tx() as conn:
            await conn.execute(
                "UPDATE pipeline_policies SET mlflow_model_version = %s WHERE policy_id = %s::uuid",
                (mlflow_model_version, policy_id),
            )

    async def get_previous_committed(self, policy: PipelinePolicy) -> PipelinePolicy | None:
        if policy.committed_at is None or policy.parent_policy_id is None:
            return None
        async with self._tx() as conn:
            cur = await conn.execute(
                _PREVIOUS, (policy.policy_id, policy.policy_id, policy.committed_at)
            )
            rows = await cur.fetchall()
            return (await self._hydrate(conn, rows))[0] if rows else None

    async def mark_synced(self, policy_id: str, mlflow_model_version: str | None) -> None:
        if not _is_uuid(policy_id):
            return
        async with self._tx() as conn:
            await conn.execute(
                "UPDATE pipeline_policies SET mlflow_synced = true, mlflow_model_version = %s "
                "WHERE policy_id = %s::uuid",
                (mlflow_model_version, policy_id),
            )
