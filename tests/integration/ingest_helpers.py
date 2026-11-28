"""Shared wiring for the ingest integration tests (real Postgres, fake queue, hash embedder)."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import psycopg

from rag_platform.adapters.embedders import HashEmbedder
from rag_platform.adapters.fake_job_queue import FakeJobQueue
from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
from rag_platform.adapters.policy_fakes import FakeRegistryMirror
from rag_platform.adapters.sql_document_repository import SqlDocumentRepository
from rag_platform.adapters.sql_index_writer import SqlIndexWriter
from rag_platform.adapters.sql_ingest_jobs import SqlJobStore
from rag_platform.adapters.sql_object_store import PostgresObjectStore
from rag_platform.adapters.sql_policy_repository import SqlPolicyRepository
from rag_platform.services.ingest.pipeline import IngestPipeline, PipelineConfig
from rag_platform.services.ingest.submit import Submitted, submit_upload
from rag_platform.services.policy_gate import PolicyGate
from rag_platform.worker.ingest_worker import IngestWorker

TENANT = "t1"
LABELS = ("staff",)
_WORDS = [
    "river",
    "stone",
    "market",
    "garden",
    "window",
    "copper",
    "silver",
    "harbor",
    "engine",
    "forest",
    "paper",
    "candle",
]


def text_for(tag: str, words: int = 1500) -> bytes:
    seed = hashlib.sha256(tag.encode()).digest()
    picks = (_WORDS[seed[i % len(seed)] % len(_WORDS)] for i in range(words))
    return (
        " ".join(f"{w}{i // 40}" if i % 7 == 0 else w for i, w in enumerate(picks)) + f" {tag}."
    ).encode()


class SimulatedKill(BaseException):
    """Stands in for SIGKILL: not an Exception, so no handler in the code under test runs."""


class CountingEmbedder:
    dim = HashEmbedder.dim

    def __init__(self) -> None:
        self._inner = HashEmbedder()
        self.texts: list[str] = []
        self.calls = 0
        self.kill_on_call: int | None = None

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls += 1
        if self.kill_on_call == self.calls:
            raise SimulatedKill
        self.texts += list(texts)
        return await self._inner.embed(texts)


@dataclass
class Stack:
    dsn: str
    embedder: CountingEmbedder
    queue: FakeJobQueue
    store: PostgresObjectStore
    docs: SqlDocumentRepository
    jobs: SqlJobStore
    writer: SqlIndexWriter
    policies: SqlPolicyRepository
    pipeline: IngestPipeline
    worker: IngestWorker
    base: list[str | None] = field(default_factory=lambda: [None])

    async def upload(self, name: str, data: bytes, tenant: str = TENANT) -> Submitted:
        return await submit_upload(
            store=self.store, jobs=self.jobs, queue=self.queue, docs=self.docs,
            tenant_id=tenant, filename=name, data=data, acl_labels=LABELS, policies=self.policies,
            mime="text/plain",
        )  # fmt: skip

    async def run_all(self) -> int:
        n = 0
        while await self.worker.run_once():
            n += 1
        return n


def make_stack(
    dsn: str, embedder: CountingEmbedder | None = None, config: PipelineConfig | None = None
) -> Stack:
    emb = embedder or CountingEmbedder()
    policies = SqlPolicyRepository(dsn)
    gate = PolicyGate(policies, FakeAuditLog(), FakeClock(), FakeRegistryMirror())
    docs, jobs = SqlDocumentRepository(dsn), SqlJobStore(dsn)
    store, writer, queue = PostgresObjectStore(dsn), SqlIndexWriter(dsn), FakeJobQueue()
    base: list[str | None] = [None]

    async def base_version() -> str | None:
        return base[0]

    pipeline = IngestPipeline(
        store=store, docs=docs, jobs=jobs, writer=writer, gate=gate, policies=policies,
        embedder=emb, config=config or PipelineConfig(embed_batch_size=2),
        base_version=base_version,
    )  # fmt: skip
    worker = IngestWorker(queue, pipeline, consumer="w1", claim_idle_ms=0, block_ms=0)
    return Stack(dsn, emb, queue, store, docs, jobs, writer, policies, pipeline, worker, base)


def q(dsn: str, sql_text: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """Superuser read (bypasses RLS) for assertions."""
    with psycopg.connect(dsn) as c:
        return c.execute(sql_text, params).fetchall()  # type: ignore[arg-type]
