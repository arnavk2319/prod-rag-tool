"""JobQueue over a Redis stream and consumer group (redis-py asyncio, already a pinned dep).

Delivery is at-least-once: a message stays in the group's pending list until `ack`. A worker
that dies mid-job leaves it pending; another `claim` takes it over with XAUTOCLAIM once it has
been idle for `min_idle_ms`. Exactly-once effect comes from idempotent job state in Postgres.
"""

from typing import Any

import structlog
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from rag_platform.services.ingest.jobs import IngestJob, QueuedJob

_log = structlog.get_logger(__name__)
_FIELD = "job"


class RedisJobQueue:
    def __init__(self, client: Redis, *, stream: str, group: str) -> None:
        self._r = client
        self._stream = stream
        self._group = group
        self._ready = False

    @classmethod
    def from_url(cls, url: str, *, stream: str, group: str) -> "RedisJobQueue":
        return cls(Redis.from_url(url, decode_responses=True), stream=stream, group=group)

    async def _ensure_group(self) -> None:
        if self._ready:
            return
        try:
            await self._r.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._ready = True

    async def publish(self, job: IngestJob) -> str:
        await self._ensure_group()
        return str(await self._r.xadd(self._stream, {_FIELD: job.model_dump_json()}))

    def _parse(self, message_id: str, fields: dict[str, Any]) -> QueuedJob | None:
        try:
            return QueuedJob(message_id, IngestJob.model_validate_json(fields[_FIELD]))
        except (KeyError, ValidationError):
            _log.warning("ingest_message_invalid", message_id=message_id)
            return None

    async def claim(
        self, consumer: str, *, block_ms: int = 0, min_idle_ms: int = 60_000
    ) -> QueuedJob | None:
        await self._ensure_group()
        while True:
            _next, claimed, *_ = await self._r.xautoclaim(
                self._stream, self._group, consumer, min_idle_time=min_idle_ms, start_id="0-0",
                count=1,
            )  # fmt: skip
            if claimed:
                mid, fields = claimed[0]
            else:
                got = await self._r.xreadgroup(
                    self._group, consumer, {self._stream: ">"}, count=1, block=block_ms or None
                )
                if not got:
                    return None
                mid, fields = got[0][1][0]
            parsed = self._parse(mid, fields)
            if parsed is not None:
                return parsed
            await self._r.xack(self._stream, self._group, mid)  # poison message: drop, never loop

    async def ack(self, message_id: str) -> None:
        await self._r.xack(self._stream, self._group, message_id)
