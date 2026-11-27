"""Turn a chunked document into index records; only children are embedded (ADR-0005)."""

from dataclasses import dataclass
from datetime import date

from rag_platform.domain.ports import Chunk, Embedder, ParentChunk
from rag_platform.services.ingest.chunker import ChunkedDocument


@dataclass(frozen=True)
class DocMeta:
    tenant_id: str
    doc_id: str
    doc_hash: str
    index_version: str
    acl_labels: frozenset[str]
    effective_date: date
    doc_version: int = 1
    expires_date: date | None = None


async def build_records(
    doc: ChunkedDocument, meta: DocMeta, embedder: Embedder
) -> tuple[list[ParentChunk], list[Chunk]]:
    """Parents are stored as text only. `Embedder.embed` receives child texts and nothing else."""
    parents = [
        ParentChunk(
            parent_id=p.parent_id,
            doc_id=meta.doc_id,
            tenant_id=meta.tenant_id,
            text=p.text,
            acl_labels=meta.acl_labels,
            index_version=meta.index_version,
            chunker_version=doc.chunker_version,
            ordinal=p.ordinal,
            token_count=p.tokens,
            doc_hash=meta.doc_hash,
            effective_date=meta.effective_date,
            doc_version=meta.doc_version,
            heading_path=p.heading_path,
            span=p.span,
            expires_date=meta.expires_date,
        )
        for p in doc.parents
    ]
    vectors = await embedder.embed([c.embed_text for c in doc.children]) if doc.children else []
    chunks = [
        Chunk(
            chunk_id=c.chunk_id,
            doc_id=meta.doc_id,
            tenant_id=meta.tenant_id,
            text=c.text,
            embedding=vec,
            acl_labels=meta.acl_labels,
            index_version=meta.index_version,
            embedding_model=embedder.model_id,
            chunker_version=doc.chunker_version,
            parent_id=c.parent_id,
            span=c.span,
        )
        for c, vec in zip(doc.children, vectors, strict=True)
    ]
    return parents, chunks
