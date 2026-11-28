# ruff: noqa: E501
"""T-010 unit tests: RedisJobQueue over an in-test fake client, BatchedEmbedder, worker acks."""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from redis.exceptions import ResponseError

from rag_platform.adapters.fake_job_queue import FakeJobQueue
from rag_platform.adapters.redis_job_queue import RedisJobQueue
from rag_platform.services.ingest.jobs import IngestJob, job_id_for
from rag_platform.services.ingest.pipeline import BatchedEmbedder
from rag_platform.worker.ingest_worker import IngestWorker

SHA = "b" * 64


def _job(i: int = 0) -> IngestJob:
    sha = f"{i:064x}"
    return IngestJob(
        job_id=job_id_for("t1", sha), tenant_id="t1", raw_sha256=sha, filename="a.txt",
        acl_labels=("staff",),
    )  # fmt: skip


class FakeRedis:
    """Just the stream commands RedisJobQueue uses."""

    def __init__(self) -> None:
        self.msgs: list[tuple[str, dict[str, str]]] = []
        self.groups: set[str] = set()
        self.delivered = 0
        self.pending: dict[str, str] = {}
        self.acked: list[str] = []
        self.group_calls = 0
        self.read_kwargs: list[dict[str, Any]] = []
        self.claim_kwargs: list[dict[str, Any]] = []
        self.stale = False  # tests flip this: pending messages count as idle

    async def xgroup_create(
        self, stream: str, group: str, id: str = "$", mkstream: bool = False
    ) -> None:
        self.group_calls += 1
        if group in self.groups:
            raise ResponseError("BUSYGROUP Consumer Group name already exists")
        self.groups.add(group)

    async def xadd(self, stream: str, fields: dict[str, str]) -> str:
        mid = f"{len(self.msgs) + 1}-0"
        self.msgs.append((mid, fields))
        return mid

    async def xautoclaim(
        self, stream: str, group: str, consumer: str, **kw: Any
    ) -> tuple[str, list[Any], list[Any]]:
        self.claim_kwargs.append(kw)
        if self.stale:
            for mid, f in self.msgs:
                if mid in self.pending:
                    self.pending[mid] = consumer
                    return ("0-0", [(mid, f)], [])
        return ("0-0", [], [])

    async def xreadgroup(
        self, group: str, consumer: str, streams: dict[str, str], count: int, block: int | None
    ) -> Any:
        self.read_kwargs.append({"count": count, "block": block})
        if self.delivered >= len(self.msgs):
            return []
        mid, f = self.msgs[self.delivered]
        self.delivered += 1
        self.pending[mid] = consumer
        return [(next(iter(streams)), [(mid, f)])]

    async def xack(self, stream: str, group: str, mid: str) -> int:
        self.acked.append(mid)
        return 1 if self.pending.pop(mid, None) else 0


def _queue(r: FakeRedis) -> RedisJobQueue:
    return RedisJobQueue(r, stream="s", group="g")  # type: ignore[arg-type]


async def test_redis_publish_claim_ack_round_trip_and_group_created_once() -> None:
    r = FakeRedis()
    q = _queue(r)
    mid = await q.publish(_job(1))
    await q.publish(_job(2))
    first = await q.claim("w1")
    assert first is not None and first.message_id == mid and first.job == _job(1)
    assert r.group_calls == 1
    await q.ack(first.message_id)
    assert r.acked == [mid]
    second = await q.claim("w1")
    assert second is not None and second.job == _job(2)
    assert await q.claim("w1") is None


async def test_redis_busygroup_is_tolerated_other_errors_raise() -> None:
    r = FakeRedis()
    r.groups.add("g")  # another worker already created it
    await _queue(r).publish(_job())
    assert r.msgs

    class Broken(FakeRedis):
        async def xgroup_create(self, *a: Any, **k: Any) -> None:
            raise ResponseError("WRONGTYPE bad key")

    with pytest.raises(ResponseError):
        await _queue(Broken()).publish(_job())


async def test_redis_stale_pending_is_reclaimed_before_new_messages() -> None:
    r = FakeRedis()
    q = _queue(r)
    await q.publish(_job(1))
    await q.publish(_job(2))
    got = await q.claim("w1")
    assert got is not None
    r.stale = True  # w1 died
    again = await q.claim("w2", min_idle_ms=1234)
    assert again is not None and again.message_id == got.message_id
    assert r.claim_kwargs[-1]["min_idle_time"] == 1234


async def test_redis_poison_message_is_acked_and_skipped() -> None:
    r = FakeRedis()
    q = _queue(r)
    await q.publish(_job(1))
    r.msgs.insert(0, ("0-1", {"job": "{not json"}))
    r.msgs.insert(1, ("0-2", {"other": "x"}))  # missing field
    r.msgs.insert(2, ("0-3", {"job": '{"job_id": "x"}'}))  # fails validation
    got = await q.claim("w1")
    assert got is not None and got.job == _job(1)
    assert r.acked == ["0-1", "0-2", "0-3"]


async def test_redis_block_zero_means_no_block_and_positive_is_passed() -> None:
    r = FakeRedis()
    q = _queue(r)
    assert await q.claim("w", block_ms=0) is None
    assert await q.claim("w", block_ms=250) is None
    assert [k["block"] for k in r.read_kwargs] == [None, 250]


async def test_batched_embedder_boundaries_order_and_zero_size() -> None:
    class E:
        model_id = "m"

        def __init__(self) -> None:
            self.sizes: list[int] = []

        async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
            self.sizes.append(len(texts))
            return [(float(t),) for t in texts]

    for size, n, calls in [
        (2, 0, 0),
        (2, 1, 1),
        (2, 4, 2),
        (2, 5, 3),
        (1, 3, 3),
        (64, 5, 1),
        (0, 3, 3),
    ]:
        inner = E()
        b = BatchedEmbedder(inner, size)  # type: ignore[arg-type]
        out = await b.embed([str(i) for i in range(n)])
        assert out == [(float(i),) for i in range(n)]
        assert len(inner.sizes) == calls == b.calls
        assert b.model_id == "m"


class _Pipe:
    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc
        self.seen: list[str] = []

    async def process(self, job: IngestJob) -> str:
        self.seen.append(job.job_id)
        if self.exc:
            raise self.exc
        return "done"


async def test_worker_acks_only_after_process_returns() -> None:
    q = FakeJobQueue()
    await q.publish(_job(1))
    w = IngestWorker(q, _Pipe(), consumer="w", block_ms=0)  # type: ignore[arg-type]
    assert await w.run_once() is True and q.acked == ["1-0"]
    assert await w.run_once() is False


@pytest.mark.parametrize("exc", [RuntimeError("db down"), KeyboardInterrupt()])
async def test_worker_does_not_ack_when_process_raises(exc: BaseException) -> None:
    q = FakeJobQueue()
    await q.publish(_job(1))
    w = IngestWorker(q, _Pipe(exc), consumer="w", block_ms=0)  # type: ignore[arg-type]
    with pytest.raises(type(exc)):
        await w.run_once()
    assert q.acked == [] and q.pending_ids() == ["1-0"]


async def test_worker_run_survives_infra_errors_backs_off_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = FakeJobQueue()
    await q.publish(_job(1))
    stop = asyncio.Event()
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)
        if len(sleeps) == 2:
            stop.set()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    pipe = _Pipe(RuntimeError("down"))
    w = IngestWorker(q, pipe, consumer="w", claim_idle_ms=0, block_ms=0)  # type: ignore[arg-type]
    await w.run(stop)
    assert sleeps == [1.0, 2.0]  # exponential backoff
    assert len(pipe.seen) == 2 and q.acked == []  # same message retried, never acked
