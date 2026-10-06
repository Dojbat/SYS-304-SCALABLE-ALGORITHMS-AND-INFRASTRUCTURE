"""Dynamic batching: group concurrent /predict calls into one model forward
pass instead of running each request through the model separately.

A single background task owns an asyncio.Queue. Each incoming request drops
its (text, Future) pair on the queue and awaits the future. The background
task pulls the first item, then keeps pulling more (up to max_batch_size)
until either the batch is full or max_wait_ms has passed since the first
item arrived, then runs one predict_batch() call for the whole batch and
resolves every future. The model call itself runs in the default executor
thread pool so it doesn't block the event loop while it runs.
"""

import asyncio
import contextlib
from collections.abc import Callable

PredictBatchFn = Callable[[list[str]], list[dict]]


class DynamicBatcher:
    def __init__(
        self,
        predict_batch_fn: PredictBatchFn,
        max_batch_size: int = 16,
        max_wait_ms: int = 20,
    ) -> None:
        self._predict_batch_fn = predict_batch_fn
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000
        self._queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def submit(self, text: str) -> dict:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((text, future))
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            deadline = loop.time() + self._max_wait_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except TimeoutError:
                    break

            texts = [text for text, _ in batch]
            try:
                results = await loop.run_in_executor(None, self._predict_batch_fn, texts)
                for (_, future), result in zip(batch, results, strict=True):
                    if not future.done():
                        future.set_result(result)
            except Exception as exc:  # noqa: BLE001 - fan the same failure out to every waiter
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
