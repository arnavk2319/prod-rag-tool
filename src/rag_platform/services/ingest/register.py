"""Idempotent document registration keyed on (tenant_id, doc_hash)."""

import hashlib
from dataclasses import dataclass

from rag_platform.domain.documents import Document, DocumentRepository, Outcome


@dataclass(frozen=True)
class Registration:
    document: Document
    outcome: Outcome


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def register_document(
    repo: DocumentRepository, *, tenant_id: str, doc_hash: str, doc_id: str, title: str
) -> Registration:
    """Thin wrapper: the created/reactivated/duplicate decision is one atomic repo operation."""
    doc, outcome = await repo.register(tenant_id, doc_hash, doc_id, title)
    return Registration(doc, outcome)
