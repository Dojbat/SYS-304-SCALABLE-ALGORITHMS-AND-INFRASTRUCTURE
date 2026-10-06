import asyncio

import pytest
from app.batcher import DynamicBatcher


def _make_predict_batch_fn(call_log: list[list[str]]):
    def predict_batch(texts: list[str]) -> list[dict]:
        call_log.append(list(texts))
        return [{"text": t, "prediction": 1, "label": "disaster", "confidence": 0.9} for t in texts]

    return predict_batch


async def test_single_request_is_served() -> None:
    batcher = DynamicBatcher(_make_predict_batch_fn([]), max_batch_size=8, max_wait_ms=10)
    batcher.start()
    try:
        result = await batcher.submit("wildfire spreading")
        assert result["text"] == "wildfire spreading"
        assert result["prediction"] == 1
    finally:
        await batcher.stop()


async def test_concurrent_requests_are_grouped_into_one_batch_call() -> None:
    call_log: list[list[str]] = []
    batcher = DynamicBatcher(_make_predict_batch_fn(call_log), max_batch_size=8, max_wait_ms=50)
    batcher.start()
    try:
        texts = [f"tweet {i}" for i in range(5)]
        results = await asyncio.gather(*(batcher.submit(t) for t in texts))
        assert [r["text"] for r in results] == texts
        # everything submitted inside the max_wait_ms window lands in one call
        assert len(call_log) == 1
        assert sorted(call_log[0]) == sorted(texts)
    finally:
        await batcher.stop()


async def test_batch_size_cap_splits_a_larger_burst() -> None:
    call_log: list[list[str]] = []
    batcher = DynamicBatcher(_make_predict_batch_fn(call_log), max_batch_size=2, max_wait_ms=50)
    batcher.start()
    try:
        texts = [f"tweet {i}" for i in range(5)]
        results = await asyncio.gather(*(batcher.submit(t) for t in texts))
        assert [r["text"] for r in results] == texts
        assert all(len(batch) <= 2 for batch in call_log)
        assert sum(len(batch) for batch in call_log) == 5
    finally:
        await batcher.stop()


async def test_predict_batch_exception_is_raised_to_every_waiter() -> None:
    def failing_predict_batch(texts: list[str]) -> list[dict]:
        raise RuntimeError("model exploded")

    batcher = DynamicBatcher(failing_predict_batch, max_batch_size=8, max_wait_ms=50)
    batcher.start()
    try:
        with pytest.raises(RuntimeError, match="model exploded"):
            await asyncio.gather(*(batcher.submit(f"tweet {i}") for i in range(3)))
    finally:
        await batcher.stop()
