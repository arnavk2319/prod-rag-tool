"""Postgres side of index-version commit: gate measurements, gate_runs, alias, retention.

Every statement is a module constant with bound parameters (or composed with sql.Identifier).
Reads and the alias swap run as `rag_app` (tenant tables under RLS, one tenant per transaction).
Only partition removal needs the table owner, so retention uses `admin_dsn` (role=None); the
owner connection is used for nothing else here.
"""

import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import structlog
from psycopg import sql
from psycopg.types.json import Jsonb

from rag_platform.adapters._pg import tx
from rag_platform.services.index_versions import GateMeasures

log = structlog.get_logger(__name__)
Conn = psycopg.AsyncConnection[dict[str, Any]]

ALIAS_LOCK_KEY = "alias:live"
_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"
_LIVE = "SELECT index_version, previous_version FROM index_alias WHERE alias = 'live'"
_SET_ALIAS = (
    "INSERT INTO index_alias (alias, index_version, previous_version, updated_at) "
    "VALUES ('live', %s, %s, now()) ON CONFLICT (alias) DO UPDATE SET "
    "index_version = EXCLUDED.index_version, previous_version = EXCLUDED.previous_version, "
    "updated_at = now()"
)
_VERSION = (
    "SELECT index_version, parent_version, status, row_count FROM index_versions "
    "WHERE index_version = %s"
)
_PARTITIONS = (
    "SELECT c.relname AS relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
    "WHERE i.inhparent = %s::regclass "
    "AND pg_get_expr(c.relpartbound, c.oid) = format('FOR VALUES IN (%%L)', %s::text)"
)
_ROWS = "SELECT count(*) AS n FROM chunks WHERE index_version = %s AND tenant_id = %s"
_VECTORS = (
    "SELECT count(*) FILTER (WHERE vector_dims(embedding) <> %s) AS bad_dim, "
    "count(*) FILTER (WHERE NOT (vector_norm(embedding) < 'Infinity'::float8)) AS nonfinite, "
    "count(*) FILTER (WHERE vector_norm(embedding) = 0) AS zero "
    "FROM chunks WHERE index_version = %s AND tenant_id = %s"
)
_SELF_RETRIEVAL = (
    "WITH s AS (SELECT chunk_id, embedding FROM chunks WHERE index_version = %s "
    "AND tenant_id = %s ORDER BY chunk_id LIMIT %s) "
    "SELECT count(*) AS total, count(*) FILTER (WHERE t.chunk_id = s.chunk_id "
    "OR t.dist <= 0.000001) AS hits FROM s CROSS JOIN LATERAL ("
    "SELECT c.chunk_id, c.embedding <=> s.embedding AS dist FROM chunks c "
    "WHERE c.index_version = %s AND c.tenant_id = %s "
    "ORDER BY c.embedding <=> s.embedding, c.chunk_id LIMIT 1) t"
)
_ORPHANS = (
    "SELECT count(*) AS n FROM chunks c WHERE c.index_version = %s AND c.tenant_id = %s "
    "AND NOT EXISTS (SELECT 1 FROM parent_chunks p WHERE p.index_version = c.index_version "
    "AND p.tenant_id = c.tenant_id AND p.parent_id = c.parent_id)"
)
_PARENTS = (
    "SELECT (SELECT count(*) FROM parent_chunks WHERE index_version = %s AND tenant_id = %s) "
    "AS parents, (SELECT count(DISTINCT parent_id) FROM chunks WHERE index_version = %s "
    "AND tenant_id = %s) AS referenced"
)
_RLS_ACTIVE = (
    "SELECT row_security_active('chunks') AS chunks, "
    "row_security_active('parent_chunks') AS parents"
)
_UNLISTED = (
    "SELECT (SELECT count(*) FROM chunks WHERE index_version = %s AND NOT (tenant_id = ANY(%s))) "
    "AS chunks, (SELECT count(*) FROM parent_chunks WHERE index_version = %s "
    "AND NOT (tenant_id = ANY(%s))) AS parents"
)
_INSERT_GATE = (
    "INSERT INTO gate_runs (gate_run_id, kind, index_version, passed, metrics) "
    "VALUES (%s, 'index_data_gate', %s, %s, %s)"
)
_STAMP_GATE = "UPDATE index_versions SET gate_run_id = %s WHERE index_version = %s"
_HISTORY = (
    "SELECT index_version, status FROM index_versions "
    "WHERE status IN ('committed', 'rolled_back') "
    "ORDER BY committed_at DESC NULLS LAST, created_at DESC, index_version DESC"
)
_DEAD = "SELECT index_version FROM index_versions WHERE status IN ('rejected', 'failed')"
_PROTECTED = "SELECT index_version, previous_version FROM index_alias"


class SqlIndexVersionStore:
    def __init__(
        self, dsn: str, *, role: str | None = "rag_app", admin_dsn: str | None = None
    ) -> None:
        self._dsn = dsn
        self._role = role
        self._admin_dsn = admin_dsn or dsn

    # ---- gate ------------------------------------------------------------------------------

    async def measure(
        self, index_version: str, tenants: Iterable[str], *, dim: int, sample: int
    ) -> GateMeasures:
        m = GateMeasures()
        async with tx(self._dsn, role=self._role) as c:
            row = await (await c.execute(_VERSION, (index_version,))).fetchone()
        m.rows_recorded = row["row_count"] if row else None
        for t in sorted(set(tenants)):
            async with tx(self._dsn, role=self._role, tenant_id=t) as c:
                await c.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
                rows = await (await c.execute(_ROWS, (index_version, t))).fetchone()
                v = await (await c.execute(_VECTORS, (dim, index_version, t))).fetchone()
                r = await (
                    await c.execute(_SELF_RETRIEVAL, (index_version, t, sample, index_version, t))
                ).fetchone()
                o = await (await c.execute(_ORPHANS, (index_version, t))).fetchone()
                p = await (
                    await c.execute(_PARENTS, (index_version, t, index_version, t))
                ).fetchone()
            if not (rows and v and r and o and p):  # aggregates always return one row
                raise RuntimeError("gate measurement returned no row")
            m.rows_actual += rows["n"]
            m.bad_dim += v["bad_dim"]
            m.nonfinite += v["nonfinite"]
            m.zero_vectors += v["zero"]
            m.self_total += r["total"]
            m.self_hits += r["hits"]
            m.orphan_children += o["n"]
            m.parent_rows += p["parents"]
            m.parents_referenced += p["referenced"]
        # Rows of tenants outside the policy are invisible to the per-tenant RLS reads above, so
        # count them once over the whole partition on the owner connection (`admin_dsn`, falling
        # back to `dsn`). Fail closed: if that connection is itself subject to RLS the count
        # would read 0 and the gate would pass blind, so refuse to measure at all.
        listed = sorted(set(tenants))
        async with tx(self._admin_dsn, role=None) as c:
            rls = await (await c.execute(_RLS_ACTIVE)).fetchone()
            if not rls or rls["chunks"] or rls["parents"]:
                log.error("index.gate_unmeasurable", reason="owner_connection_subject_to_rls")
                raise RuntimeError(
                    "unlisted-tenant check needs an RLS-bypassing connection (admin_dsn)"
                )
            u = await (
                await c.execute(_UNLISTED, (index_version, listed, index_version, listed))
            ).fetchone()
        if not u:
            raise RuntimeError("gate measurement returned no row")
        m.unlisted_chunk_rows = u["chunks"]
        m.unlisted_parent_rows = u["parents"]
        return m

    async def record_gate_run(
        self, index_version: str, passed: bool, metrics: dict[str, Any]
    ) -> str:
        gate_run_id = f"gr-{uuid.uuid4().hex}"
        async with tx(self._dsn, role=self._role) as c:
            await c.execute(_INSERT_GATE, (gate_run_id, index_version, passed, Jsonb(metrics)))
            await c.execute(_STAMP_GATE, (gate_run_id, index_version))
        return gate_run_id

    # ---- alias -----------------------------------------------------------------------------

    @asynccontextmanager
    async def alias_tx(self) -> AsyncIterator[Conn]:
        """One rag_app transaction holding the alias lock; commits on exit, rolls back on error."""
        async with tx(self._dsn, role=self._role) as c:
            await c.execute(_LOCK, (ALIAS_LOCK_KEY,))
            yield c

    @staticmethod
    async def live(c: Conn) -> tuple[str, str | None] | None:
        row = await (await c.execute(_LIVE)).fetchone()
        return (row["index_version"], row["previous_version"]) if row else None

    @staticmethod
    async def version_row(c: Conn, index_version: str) -> dict[str, Any] | None:
        return await (await c.execute(_VERSION, (index_version,))).fetchone()

    @staticmethod
    async def has_partition(c: Conn, index_version: str) -> bool:
        found = await (await c.execute(_PARTITIONS, ("chunks", index_version))).fetchone()
        return found is not None

    @staticmethod
    async def set_alias(c: Conn, index_version: str, previous: str | None) -> None:
        await c.execute(_SET_ALIAS, (index_version, previous))

    async def read_live(self) -> str | None:
        async with tx(self._dsn, role=self._role) as c:
            row = await self.live(c)
        return row[0] if row else None

    # ---- retention -------------------------------------------------------------------------

    async def retain(self, keep: int) -> list[str]:
        """Remove chunk and parent partitions outside the keep window; returns versions removed.

        Rule: history = committed and rolled_back versions, newest first; the first `keep` stay,
        and the alias target and its previous version always stay. Rejected and failed versions
        are unusable, so their partitions go at once. Staged and building versions are left alone
        (they may be in flight). index_versions rows are kept (audit, parent_version FKs).
        Runs under the alias lock, so a concurrent commit or rollback cannot race it.
        Requires the owner DSN (`admin_dsn`): DROP needs table ownership, so with an app-role
        DSN it fails with a permission error (nothing is removed).
        """
        removed: list[str] = []
        async with tx(self._admin_dsn, role=None) as c:
            await c.execute(_LOCK, (ALIAS_LOCK_KEY,))
            history = [r["index_version"] for r in await (await c.execute(_HISTORY)).fetchall()]
            protected = set(history[:keep])
            for r in await (await c.execute(_PROTECTED)).fetchall():
                protected |= {v for v in (r["index_version"], r["previous_version"]) if v}
            doomed = [v for v in history if v not in protected]
            doomed += [r["index_version"] for r in await (await c.execute(_DEAD)).fetchall()]
            for v in doomed:
                if v in protected:
                    continue
                gone = False
                for parent in ("chunks", "parent_chunks"):
                    found = await (await c.execute(_PARTITIONS, (parent, v))).fetchall()
                    for p in found:
                        # partition name comes from pg_class, quoted as an identifier
                        # nosemgrep
                        await c.execute(
                            sql.SQL("DROP TABLE {}").format(sql.Identifier("public", p["relname"]))
                        )
                        gone = True
                if gone:
                    removed.append(v)
        return removed
