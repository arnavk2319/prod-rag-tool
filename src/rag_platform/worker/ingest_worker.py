"""Ingest worker loop: claim, process, ack. Queue and pipeline are injected (fakes in tests).

Ack happens only after `process` returns a terminal job status. If the process dies before the
ack, the message stays pending, is reclaimed after `claim_idle_ms`, and the idempotent job state
makes the second run complete the job once (never a second document, version or chunk set).
"""

import asyncio

import structlog

from rag_platform.services.ingest.jobs import JobQueue
from rag_platform.services.ingest.pipeline import IngestPipeline

_log = structlog.get_logger(__name__)


class IngestWorker:
    def __init__(
        self,
        queue: JobQueue,
        pipeline: IngestPipeline,
        *,
        consumer: str,
        claim_idle_ms: int = 60_000,
        block_ms: int = 5_000,
    ) -> None:
        self._queue = queue
        self._pipeline = pipeline
        self._consumer = consumer
        self._idle = claim_idle_ms
        self._block = block_ms

    async def run_once(self) -> bool:
        """Handle at most one message; False when the queue had nothing."""
        item = await self._queue.claim(self._consumer, block_ms=self._block, min_idle_ms=self._idle)
        if item is None:
            return False
        await self._pipeline.process(item.job)
        await self._queue.ack(item.message_id)
        return True

    async def run(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            try:
                await self.run_once()
                backoff = 1.0
            except Exception:  # noqa: BLE001 - infra error: message stays pending, retry later
                _log.warning("ingest_worker_error", consumer=self._consumer)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
