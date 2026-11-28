"""DocumentRepository over Postgres (`documents`, RLS by tenant).

`register` is one statement, `INSERT ... ON CONFLICT (tenant_id, doc_hash) DO UPDATE`, so two
concurrent uploads of the same bytes cannot create two rows. Outcomes match FakeDocumentRepository.
"""

from collections.abc import Sequence
from typing import Any

from psycopg.errors import UniqueViolation

from rag_platform.adapters._pg import tx
from rag_platform.domain.documents import DocRow, DocStatus, Document, Outcome

# The WHERE on DO UPDATE turns an active-row conflict into "no row returned" (a duplicate).
# xmax = 0 is true only for a freshly inserted row (created); an updated row is reactivated.
_REGISTER = (
    "INSERT INTO documents (doc_id, tenant_id, doc_hash, title, acl_labels, pii_class) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (tenant_id, doc_hash) DO UPDATE SET status = 'active' "
    "WHERE documents.status = 'retired' "
    "RETURNING doc_id, tenant_id, doc_hash, title, status, version, (xmax = 0) AS inserted"
)
_BY_HASH = (
    "SELECT doc_id, tenant_id, doc_hash, title, status, version FROM documents "
    "WHERE tenant_id = %s AND doc_hash = %s"
)
_INSERT = (
    "INSERT INTO documents (doc_id, tenant_id, doc_hash, title, status, version) "
    "VALUES (%s, %s, %s, %s, %s, %s)"
)
_SET_STATUS = (
    "UPDATE documents SET status = %s WHERE tenant_id = %s AND doc_id = %s "
    "RETURNING doc_id, tenant_id, doc_hash, title, status, version"
)
_ACTIVE = (
    "SELECT doc_id, tenant_id, doc_hash, title, status, version, acl_labels, effective_date, "
    "expires_date FROM documents WHERE tenant_id = %s AND status = 'active' "
    "AND doc_id = ANY(%s) ORDER BY doc_id"
)


def _doc(r: dict[str, Any]) -> Document:
    return Document(
        r["doc_id"], r["tenant_id"], r["doc_hash"], r["title"], r["status"], r["version"]
    )


class SqlDocumentRepository:
    def __init__(self, dsn: str, *, role: str | None = "rag_app") -> None:
        self._dsn = dsn
        self._role = role

    async def get_by_hash(self, tenant_id: str, doc_hash: str) -> Document | None:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            row = await (await c.execute(_BY_HASH, (tenant_id, doc_hash))).fetchone()
        return _doc(row) if row else None

    async def register(
        self,
        tenant_id: str,
        doc_hash: str,
        doc_id: str,
        title: str,
        *,
        acl_labels: Sequence[str] = (),
        pii_class: str = "none",
    ) -> tuple[Document, Outcome]:
        try:
            async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
                params = (doc_id, tenant_id, doc_hash, title, list(acl_labels), pii_class)
                row = await (await c.execute(_REGISTER, params)).fetchone()
                if row is not None:
                    return _doc(row), ("created" if row["inserted"] else "reactivated")
                row = await (await c.execute(_BY_HASH, (tenant_id, doc_hash))).fetchone()
                if row is None:  # deleted between the two statements: not a duplicate
                    raise RuntimeError("document vanished during register")
                return _doc(row), "duplicate"
        except UniqueViolation as e:
            raise ValueError("unique violation: (tenant_id, doc_id)") from e

    async def insert(self, doc: Document) -> Document:
        try:
            async with tx(self._dsn, role=self._role, tenant_id=doc.tenant_id) as c:
                await c.execute(
                    _INSERT,
                    (doc.doc_id, doc.tenant_id, doc.doc_hash, doc.title, doc.status, doc.version),
                )
        except UniqueViolation as e:
            raise ValueError("unique violation") from e
        return doc

    async def set_status(self, tenant_id: str, doc_id: str, status: DocStatus) -> Document:
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            row = await (await c.execute(_SET_STATUS, (status, tenant_id, doc_id))).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return _doc(row)

    async def list_active(self, tenant_id: str, doc_ids: Sequence[str]) -> list[DocRow]:
        """Active documents among `doc_ids` (retired or unknown ids are omitted)."""
        async with tx(self._dsn, role=self._role, tenant_id=tenant_id) as c:
            rows = await (await c.execute(_ACTIVE, (tenant_id, list(doc_ids)))).fetchall()
        return [
            DocRow(_doc(r), frozenset(r["acl_labels"]), r["effective_date"], r["expires_date"])
            for r in rows
        ]
