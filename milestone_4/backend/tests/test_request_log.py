import contextlib

from app.request_log import COLUMNS, RequestLogger


class RecordingPool:
    def __init__(self, fail: bool = False) -> None:
        self.batches: list[list[tuple]] = []
        self.fail = fail

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self

    async def copy_records_to_table(self, table, records, columns):
        if self.fail:
            raise ConnectionError("db down")
        self.batches.append(list(records))


def _logger_for(pool, **kwargs) -> RequestLogger:
    async def get_pool():
        return pool

    return RequestLogger(get_pool, **kwargs)


async def test_rows_are_written_in_batches() -> None:
    pool = RecordingPool()
    request_logger = _logger_for(pool, max_batch_size=2)
    for i in range(5):
        request_logger.log({"id": i, "status_code": 200})

    await request_logger.flush()

    assert [len(batch) for batch in pool.batches] == [2, 2, 1]
    assert request_logger.written == 5
    # rows are ordered by COLUMNS, missing fields filled with None
    assert pool.batches[0][0] == tuple(
        {"id": 0, "status_code": 200}.get(column) for column in COLUMNS
    )


async def test_database_failure_drops_rows_without_raising() -> None:
    request_logger = _logger_for(RecordingPool(fail=True))
    request_logger.log({"id": 1})

    await request_logger.flush()

    assert request_logger.dropped == 1
    assert request_logger.written == 0


async def test_missing_pool_drops_rows() -> None:
    request_logger = _logger_for(None)
    request_logger.log({"id": 1})

    await request_logger.flush()

    assert request_logger.dropped == 1


async def test_full_queue_drops_new_rows_instead_of_blocking() -> None:
    request_logger = _logger_for(RecordingPool(), max_queue_size=2)
    for i in range(3):
        request_logger.log({"id": i})

    assert request_logger.dropped == 1
