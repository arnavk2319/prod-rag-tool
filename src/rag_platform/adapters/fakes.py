"""Deterministic in-memory fakes for every port (no network, injected time)."""

import hashlib
import math
import threading
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from rag_platform.domain.documents import DocStatus, Document, Outcome
from rag_platform.domain.ports import (
    AuditEvent,
    Chunk,
    Completion,
    ParentChunk,
    ScoredChunk,
)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


class FakeObjectStore:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    async def put(self, tenant_id: str, data: bytes, mime: str) -> str:
        sha = hashlib.sha256(data).hexdigest()
        self._data[(tenant_id, sha)] = data
        return sha

    async def get(self, tenant_id: str, sha256: str) -> bytes | None:
        return self._data.get((tenant_id, sha256))

    async def delete(self, tenant_id: str, sha256: str) -> bool:
        return self._data.pop((tenant_id, sha256), None) is not None


class FakeEmbedder:
    dim = 8

    @property
    def model_id(self) -> str:
        return "fake-embed-v1"

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        out: list[tuple[float, ...]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            vec = [b - 127.5 for b in digest[: self.dim]]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append(tuple(v / norm for v in vec))
        return out


class FakeReranker:
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        words = set(query.lower().split())

        def overlap(c: ScoredChunk) -> int:
            return len(words & set(c.chunk.text.lower().split()))

        return sorted(candidates, key=lambda c: (-overlap(c), -c.score, c.chunk.chunk_id))


class FakeIndexRepository:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._parents: dict[tuple[str, str, str], ParentChunk] = {}

    async def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        for c in chunks:
            self._chunks[c.chunk_id] = c  # idempotent on chunk id
        return len(chunks)

    async def add_parents(self, parents: Sequence[ParentChunk]) -> int:
        for p in parents:
            self._parents[(p.index_version, p.tenant_id, p.parent_id)] = p  # idempotent
        return len(parents)

    async def get_parents(
        self, *, tenant_id: str, index_version: str, parent_ids: Sequence[str]
    ) -> dict[str, ParentChunk]:
        """Keyed like the SQL table (index_version, tenant_id, parent_id): no cross-tenant hit."""
        found = {pid: self._parents.get((index_version, tenant_id, pid)) for pid in parent_ids}
        return {pid: p for pid, p in found.items() if p is not None}

    async def search(
        self,
        embedding: Sequence[float],
        *,
        tenant_id: str,
        user_labels: frozenset[str],
        index_version: str,
        k: int,
    ) -> list[ScoredChunk]:
        hits = [
            ScoredChunk(c, sum(a * b for a, b in zip(embedding, c.embedding, strict=True)))
            for c in self._chunks.values()
            if c.tenant_id == tenant_id
            and c.index_version == index_version
            and c.acl_labels
            and c.acl_labels <= user_labels
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[:k]

    async def delete_document(self, tenant_id: str, doc_id: str) -> int:
        ids = [
            i for i, c in self._chunks.items() if c.tenant_id == tenant_id and c.doc_id == doc_id
        ]
        for i in ids:
            del self._chunks[i]
        for key in [
            k for k, p in self._parents.items() if k[1] == tenant_id and p.doc_id == doc_id
        ]:
            del self._parents[key]
        return len(ids)


class FakeModelProvider:
    async def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        text = f"fake answer: {hashlib.sha256(prompt.encode()).hexdigest()[:8]}"
        return Completion(
            text=text,
            tokens_in=len(prompt.split()),
            tokens_out=min(len(text.split()), max_tokens),
            usd=0.0,
            model="fake-llm-v1",
        )


class FakeCache:
    def __init__(self, clock: "FakeClock") -> None:
        self._clock = clock
        self._data: dict[str, tuple[str, datetime]] = {}

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None or item[1] <= self._clock.now():
            self._data.pop(key, None)
            return None
        return item[0]

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        self._data[key] = (value, self._clock.now() + timedelta(seconds=ttl_s))

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class FakeRateLimiter:
    def __init__(self, clock: "FakeClock") -> None:
        self._clock = clock
        self._hits: dict[str, list[datetime]] = {}

    async def allow(self, key: str, limit: int, window_s: int) -> bool:
        now = self._clock.now()
        cutoff = now - timedelta(seconds=window_s)
        hits = [t for t in self._hits.get(key, []) if t > cutoff]
        if len(hits) >= limit:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True


class FakeRegistry:
    def __init__(self) -> None:
        self._ptrs: dict[str, str] = {}

    async def get_pointer(self, name: str) -> str | None:
        return self._ptrs.get(name)

    async def set_pointer(self, name: str, value: str) -> None:
        self._ptrs[name] = value


class FakeAuditLog:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self._events.append(event)

    async def list_events(self, tenant_id: str) -> list[AuditEvent]:
        return [e for e in self._events if e.tenant_id == tenant_id]


class FakeDocumentRepository:
    """In-memory documents table; enforces UNIQUE (tenant_id, doc_hash) and tenant-scoped keys."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], Document] = {}  # (tenant_id, doc_id)
        self._lock = threading.Lock()

    async def register(
        self, tenant_id: str, doc_hash: str, doc_id: str, title: str
    ) -> tuple[Document, Outcome]:
        with self._lock:  # no await inside: atomic across tasks and threads
            for d in self._rows.values():
                if d.tenant_id == tenant_id and d.doc_hash == doc_hash:
                    if d.status == "retired":
                        d = replace(d, status="active")
                        self._rows[(tenant_id, d.doc_id)] = d
                        return d, "reactivated"
                    return d, "duplicate"
            if (tenant_id, doc_id) in self._rows:
                raise ValueError("unique violation: (tenant_id, doc_id)")
            doc = Document(doc_id, tenant_id, doc_hash, title)
            self._rows[(tenant_id, doc_id)] = doc
            return doc, "created"

    def _find(self, tenant_id: str, doc_hash: str) -> Document | None:
        for d in self._rows.values():
            if d.tenant_id == tenant_id and d.doc_hash == doc_hash:
                return d
        return None

    async def get_by_hash(self, tenant_id: str, doc_hash: str) -> Document | None:
        with self._lock:
            return self._find(tenant_id, doc_hash)

    async def insert(self, doc: Document) -> Document:
        with self._lock:
            if self._find(doc.tenant_id, doc.doc_hash) is not None:
                raise ValueError("unique violation: (tenant_id, doc_hash)")
            if (doc.tenant_id, doc.doc_id) in self._rows:
                raise ValueError("unique violation: (tenant_id, doc_id)")
            self._rows[(doc.tenant_id, doc.doc_id)] = doc
            return doc

    async def set_status(self, tenant_id: str, doc_id: str, status: DocStatus) -> Document:
        with self._lock:
            doc = replace(self._rows[(tenant_id, doc_id)], status=status)
            self._rows[(tenant_id, doc_id)] = doc
            return doc
