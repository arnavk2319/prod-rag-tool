"""Write side of the index: stage a version, copy forward, summarize what was written.

Chunks and parents are INSERT-only (the schema grants rag_app no UPDATE on either). Every
statement is a module constant with bound parameters and runs per tenant under RLS.
Partition DDL (`create_chunks_partition`) needs the table owner, so it uses `admin_dsn`
(role=None), not the `rag_app` connection. TODO(hardening): a SECURITY DEFINER wrapper would
let the worker avoid the owner connection (needs a migration, out of T-010 scope).
"""

from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager

from rag_platform.adapters._pg import tx
from rag_platform.domain.ports import Chunk, ParentChunk
from rag_platform.services.ingest.copy_forward import COPY_FORWARD_STATEMENTS
from rag_platform.services.policy_gate import ChunkSummary

_LOCK = "SELECT pg_advisory_lock(hashtextextended(%s, 0))"
_PARTITION = "SELECT create_chunks_partition(%s)"
_INSERT_PARENT = (
    "INSERT INTO parent_chunks (index_version, tenant_id, parent_id, doc_id, doc_version, ordinal, "
    "text, token_count, acl_labels, heading_path, doc_hash, span_start, span_end, effective_date, "
    "expires_date, chunker_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "%s, %s, %s) ON CONFLICT DO NOTHING"
)
_INSERT_CHUNK = (
    "INSERT INTO chunks (chunk_id, index_version, tenant_id, doc_id, doc_version, ordinal, text, "
    "span_start, span_end, acl_labels, effective_date, expires_date, embedding_model, "
    "chunker_version, embedding, parent_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "%s, %s, %s, %s::vector, %s) ON CONFLICT DO NOTHING"
)
_CLEAR_CHUNKS = "DELETE FROM chunks WHERE index_version = %s AND tenant_id = %s"
_CLEAR_PARENTS = "DELETE FROM parent_chunks WHERE index_version = %s AND tenant_id = %s"
_SUMMARY = (
    "SELECT embedding_model, chunker_version, vector_dims(embedding) AS dim, doc_id, "
    "count(*) AS n FROM chunks WHERE index_version = %s AND tenant_id = %s "
    "GROUP BY embedding_model, chunker_version, vector_dims(embedding), doc_id"
)
_ROW_COUNT = "UPDATE index_versions SET row_count = %s WHERE index_version = %s"
_LIVE = "SELECT index_version FROM index_alias WHERE alias = 'live'"
_DOC_CHUNKS = (
    "SELECT count(*) AS n FROM chunks WHERE index_version = %s AND tenant_id = %s AND doc_id = %s"
)


def vector_literal(v: Iterable[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


class SqlIndexWriter:
    def __init__(
        self, dsn: str, *, role: str | None = "rag_app", admin_dsn: str | None = None
    ) -> None:
        self._dsn = dsn
        self._role = role
        self._admin_dsn = admin_dsn or dsn

    @asynccontextmanager
    async def version_lock(self, index_version: str) -> AsyncIterator[None]:
        """Session advisory lock on a dedicated connection: a second worker on the same staged
        version waits, then sees the finished policy and returns. A dead worker drops the lock."""
        async with tx(self._admin_dsn, role=None) as c:
            await c.execute(_LOCK, (f"staged:{index_version}",))
            yield

    async def ensure_partition(self, index_version: str) -> None:
        async with tx(self._admin_dsn, role=None) as c:
            await c.execute(_PARTITION, (index_version,))

    async def live_version(self) -> str | None:
        async with tx(self._dsn, role=self._role) as c:
            row = await (await c.execute(_LIVE)).fetchone()
        return row["index_version"] if row else None

    async def clear_version(self, index_version: str, tenants: Iterable[str]) -> None:
        """Remove partial rows of a staged version before a retry (never touches other versions)."""
        for t in sorted(set(tenants)):
            async with tx(self._dsn, role=self._role, tenant_id=t) as c:
                await c.execute(_CLEAR_CHUNKS, (index_version, t))
                await c.execute(_CLEAR_PARENTS, (index_version, t))

    async def copy_forward(
        self, from_version: str, to_version: str, tenant_id: str, doc_ids: Sequence[str]
    ) -> None:
        """INSERT ... SELECT, parents first then children, in one transaction. No embedder."""
        if not doc_ids:
            return
        params = {
            "from_version": from_version,
            "to_version": to_version,
            "tenant_id": tenant_id,
            "doc_ids": list(doc_ids),
        }
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            for stmt in COPY_FORWARD_STATEMENTS:
                await c.execute(stmt, params)

    async def add_document(self, parents: Sequence[ParentChunk], chunks: Sequence[Chunk]) -> None:
        """One document, one transaction: parents then children (all one tenant, one version)."""
        if not parents:
            return
        by_id = {p.parent_id: p for p in parents}
        first = parents[0]
        async with (
            tx(self._dsn, role=self._role, tenant_id=first.tenant_id) as c,
            c.cursor() as cur,
        ):
            await cur.executemany(
                _INSERT_PARENT,
                [
                    (
                        p.index_version, p.tenant_id, p.parent_id, p.doc_id, p.doc_version,
                        p.ordinal, p.text, p.token_count, sorted(p.acl_labels),
                        " > ".join(p.heading_path), p.doc_hash, p.span[0], p.span[1],
                        p.effective_date, p.expires_date, p.chunker_version,
                    )
                    for p in parents
                ],
            )  # fmt: skip
            rows = []
            for i, ch in enumerate(chunks):
                p = by_id[ch.parent_id]
                rows.append(
                    (
                        ch.chunk_id, ch.index_version, ch.tenant_id, ch.doc_id, p.doc_version,
                        i, ch.text, ch.span[0], ch.span[1], sorted(ch.acl_labels),
                        p.effective_date, p.expires_date, ch.embedding_model,
                        ch.chunker_version, vector_literal(ch.embedding), ch.parent_id,
                    )
                )  # fmt: skip
            await cur.executemany(_INSERT_CHUNK, rows)

    async def summarize(self, index_version: str, tenants: Iterable[str]) -> ChunkSummary:
        pairs: set[tuple[str, str]] = set()
        dims: set[int] = set()
        docs: set[tuple[str, str]] = set()
        rows = 0
        for t in sorted(set(tenants)):
            async with tx(self._dsn, role=self._role, tenant_id=t) as c:
                found = await (await c.execute(_SUMMARY, (index_version, t))).fetchall()
            for r in found:
                pairs.add((r["embedding_model"], r["chunker_version"]))
                dims.add(r["dim"])
                docs.add((t, r["doc_id"]))
                rows += r["n"]
        async with tx(self._dsn, role=self._role) as c:
            await c.execute(_ROW_COUNT, (rows, index_version))
        dim = next(iter(dims)) if len(dims) == 1 else -1  # -1: none or mixed, fails verify
        return ChunkSummary(frozenset(pairs), rows, dim, frozenset(docs))

    async def count_doc_chunks(self, index_version: str, tenant_id: str, doc_id: str) -> int:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            row = await (
                await c.execute(_DOC_CHUNKS, (index_version, tenant_id, doc_id))
            ).fetchone()
        return int(row["n"]) if row else 0
