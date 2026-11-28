"""In-memory JobQueue with Redis-stream semantics: pending until acked, stale pending reclaimed."""

from dataclasses import dataclass, field

from rag_platform.services.ingest.jobs import IngestJob, QueuedJob


@dataclass
class FakeJobQueue:
    _msgs: list[tuple[str, IngestJob]] = field(default_factory=list)
    _delivered: int = 0  # index of the next never-delivered message
    _pending: dict[str, str] = field(default_factory=dict)  # message id -> consumer
    acked: list[str] = field(default_factory=list)

    async def publish(self, job: IngestJob) -> str:
        mid = f"{len(self._msgs) + 1}-0"
        self._msgs.append((mid, job))
        return mid

    async def claim(
        self, consumer: str, *, block_ms: int = 0, min_idle_ms: int = 60_000
    ) -> QueuedJob | None:
        # Tests have no clock: min_idle_ms == 0 means "a crashed consumer's pending is stale".
        if min_idle_ms == 0:
            for mid, job in self._msgs:
                if mid in self._pending:
                    self._pending[mid] = consumer
                    return QueuedJob(mid, job)
        if self._delivered < len(self._msgs):
            mid, job = self._msgs[self._delivered]
            self._delivered += 1
            self._pending[mid] = consumer
            return QueuedJob(mid, job)
        return None

    async def ack(self, message_id: str) -> None:
        if self._pending.pop(message_id, None) is not None:
            self.acked.append(message_id)

    def pending_ids(self) -> list[str]:
        return list(self._pending)
