"""Ports: the only things domain and services may depend on."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    tenant_id: str
    text: str
    embedding: tuple[float, ...]
    acl_labels: frozenset[str]
    index_version: str
    embedding_model: str
    chunker_version: str
    parent_id: str
    span: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class ParentChunk:
    """The passage a child belongs to. Returned to the generator; never embedded or searched."""

    parent_id: str
    doc_id: str
    tenant_id: str
    text: str
    acl_labels: frozenset[str]
    index_version: str
    chunker_version: str
    ordinal: int
    token_count: int
    doc_hash: str
    effective_date: date
    doc_version: int = 1
    heading_path: tuple[str, ...] = ()
    span: tuple[int, int] = (0, 0)
    expires_date: date | None = None


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ChildMatch:
    chunk_id: str
    score: float
    span: tuple[int, int]


@dataclass(frozen=True)
class RetrievedPassage:
    """Retrieval result: a parent's text plus the child chunks that matched inside it."""

    parent_id: str
    doc_id: str
    text: str
    token_count: int
    score: float  # best matching child
    children: tuple[ChildMatch, ...]
    truncated: bool = False


@dataclass(frozen=True)
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    usd: float
    model: str


@dataclass(frozen=True)
class AuditEvent:
    at: datetime
    actor: str
    action: str
    tenant_id: str
    detail: dict[str, Any] = field(default_factory=dict)


class Clock(Protocol):
    def now(self) -> datetime: ...


class ObjectStore(Protocol):
    async def put(self, tenant_id: str, data: bytes, mime: str) -> str: ...
    async def get(self, tenant_id: str, sha256: str) -> bytes | None: ...
    async def delete(self, tenant_id: str, sha256: str) -> bool: ...


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...
    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


class IndexRepository(Protocol):
    async def add_chunks(self, chunks: Sequence[Chunk]) -> int: ...
    async def add_parents(self, parents: Sequence[ParentChunk]) -> int: ...
    async def get_parents(
        self, *, tenant_id: str, index_version: str, parent_ids: Sequence[str]
    ) -> dict[str, ParentChunk]: ...
    async def search(
        self,
        embedding: Sequence[float],
        *,
        tenant_id: str,
        user_labels: frozenset[str],
        index_version: str,
        k: int,
    ) -> list[ScoredChunk]: ...
    async def delete_document(self, tenant_id: str, doc_id: str) -> int: ...


class ModelProvider(Protocol):
    async def complete(self, prompt: str, *, max_tokens: int) -> Completion: ...


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_s: int) -> None: ...
    async def delete(self, key: str) -> None: ...


class RateLimiter(Protocol):
    async def allow(self, key: str, limit: int, window_s: int) -> bool: ...


class Registry(Protocol):
    async def get_pointer(self, name: str) -> str | None: ...
    async def set_pointer(self, name: str, value: str) -> None: ...


class AuditLog(Protocol):
    async def record(self, event: AuditEvent) -> None: ...
    async def list_events(self, tenant_id: str) -> list[AuditEvent]: ...
