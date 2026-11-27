"""Document rows and the repository port (UNIQUE (tenant_id, doc_hash) in the schema)."""

from dataclasses import dataclass
from typing import Literal, Protocol

DocStatus = Literal["active", "retired"]
Outcome = Literal["created", "reactivated", "duplicate"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    tenant_id: str
    doc_hash: str
    title: str
    status: DocStatus = "active"
    version: int = 1


class DocumentRepository(Protocol):
    async def get_by_hash(self, tenant_id: str, doc_hash: str) -> Document | None: ...
    async def register(
        self, tenant_id: str, doc_hash: str, doc_id: str, title: str
    ) -> tuple[Document, Outcome]:
        """One atomic decision, like INSERT ... ON CONFLICT (tenant_id, doc_hash) DO UPDATE.

        No row -> insert (created). Retired row -> set active, same row (reactivated).
        Active row -> unchanged (duplicate). Never a second row per (tenant_id, doc_hash).
        """
        ...

    async def insert(self, doc: Document) -> Document: ...
    async def set_status(self, tenant_id: str, doc_id: str, status: DocStatus) -> Document: ...
