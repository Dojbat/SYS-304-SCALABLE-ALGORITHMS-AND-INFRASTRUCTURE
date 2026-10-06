"""Non-blocking request logging to PostgreSQL.

Writing one row per request inline would add a database round trip to every
/predict call. Instead, request handlers drop rows on an in-memory queue and
a background task flushes them in batches with COPY, which is much cheaper
per row than individual INSERTs.

The queue is bounded: if Postgres is down or slow and the queue fills up,
new rows are dropped (and counted) rather than growing memory without limit
or slowing down predictions. Logging is best-effort; serving is not.
"""

import asyncio
import contextlib
import logging
from typing import Any

logger = logging.getLogger("disaster-tweet-api")

COLUMNS = (
    "id",
    "ts",
    "endpoint",
    "status_code",
    "latency_ms",
    "text",
    "prediction",
    "confidence",
    "cached",
    "model_version",
    "error",
)


class RequestLogger:
    def __init__(
        self,
        get_pool,
        flush_interval_ms: int = 1000,
        max_batch_size: int = 500,
        max_queue_size: int = 10_000,
    ) -> None:
        # get_pool is an async callable returning a connection pool (or None
        # when the database is unavailable), so a database that comes up
        # after the backend is picked up on the next flush.
        self._get_pool = get_pool
        self._flush_interval_s = flush_interval_ms / 1000
        self._max_batch_size = max_batch_size
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=max_queue_size)
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.written = 0

    def log(self, row: dict[str, Any]) -> None:
        record = tuple(row.get(column) for column in COLUMNS)
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.flush()

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval_s)
            await self.flush()

    async def flush(self) -> None:
        while not self._queue.empty():
            batch = []
            while len(batch) < self._max_batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            await self._write(batch)

    async def _write(self, batch: list[tuple]) -> None:
        try:
            pool = await self._get_pool()
            if pool is None:
                raise ConnectionError("database unavailable")
            async with pool.acquire() as conn:
                await conn.copy_records_to_table("requests", records=batch, columns=COLUMNS)
            self.written += len(batch)
        except Exception:
            self.dropped += len(batch)
            logger.warning("dropped %d request log rows", len(batch), exc_info=True)
