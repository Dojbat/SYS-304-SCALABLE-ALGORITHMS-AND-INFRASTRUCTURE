import contextlib
from uuid import UUID

import app.main as main_module
import fakeredis
import pytest
from app.batcher import DynamicBatcher
from app.cache import PredictionCache
from app.request_log import COLUMNS, RequestLogger
from fastapi.testclient import TestClient


class FakeModelManager:
    def __init__(self) -> None:
        self.loaded = True
        self.version = "v1"
        self.calls: list[list[str]] = []
        self.pending_version: str | None = None

    def reload_if_changed(self) -> bool:
        if self.pending_version and self.pending_version != self.version:
            self.version = self.pending_version
            return True
        return False

    def predict_batch(self, texts: list[str]) -> list[dict]:
        self.calls.append(list(texts))
        return [
            {
                "text": t,
                "prediction": 1 if "fire" in t else 0,
                "label": "disaster" if "fire" in t else "not disaster",
                "confidence": 0.9,
                "model_version": self.version,
            }
            for t in texts
        ]


class FakePool:
    """Stands in for an asyncpg pool: records COPY batches and executes."""

    def __init__(self) -> None:
        self.copied: list[tuple] = []
        self.executed: list[tuple] = []

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self

    async def copy_records_to_table(self, table, records, columns):
        assert table == "requests"
        assert tuple(columns) == COLUMNS
        self.copied.extend(records)

    async def execute(self, query, *args):
        self.executed.append(args)

    async def fetchval(self, query):
        return 1


@pytest.fixture
def manager() -> FakeModelManager:
    return FakeModelManager()


@pytest.fixture
def pool() -> FakePool:
    return FakePool()


@pytest.fixture
def client(monkeypatch, manager, pool):
    async def get_pool():
        return pool

    monkeypatch.setattr(main_module, "model_manager", manager)
    monkeypatch.setattr(
        main_module, "batcher", DynamicBatcher(manager.predict_batch, max_wait_ms=5)
    )
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(main_module, "redis_client", fake_redis)
    monkeypatch.setattr(main_module, "cache", PredictionCache(fake_redis, ttl_seconds=60))
    monkeypatch.setattr(main_module, "get_db_pool", get_pool)
    monkeypatch.setattr(main_module, "request_logger", RequestLogger(get_pool))
    monkeypatch.setattr(main_module, "ADMIN_TOKEN", "secret")

    with TestClient(main_module.app) as c:
        yield c


def _logged_rows(client, pool: FakePool) -> list[dict]:
    client.portal.call(main_module.request_logger.flush)
    return [dict(zip(COLUMNS, record, strict=True)) for record in pool.copied]


def test_health_reports_every_dependency(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "v1",
        "redis_connected": True,
        "database_connected": True,
    }


def test_predict_returns_request_id_and_model_version(client):
    response = client.post("/predict", json={"text": "Wildfire spreading near the highway"})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["model_version"] == "v1"
    assert body["cached"] is False
    assert UUID(body["request_id"])
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_successful_prediction_is_logged(client, pool):
    body = client.post("/predict", json={"text": "Wildfire spreading near the highway"}).json()

    rows = _logged_rows(client, pool)
    assert len(rows) == 1
    row = rows[0]
    assert str(row["id"]) == body["request_id"]
    assert row["status_code"] == 200
    assert row["text"] == "Wildfire spreading near the highway"
    assert row["prediction"] == 1
    assert row["confidence"] == pytest.approx(0.9)
    assert row["cached"] is False
    assert row["model_version"] == "v1"
    assert row["latency_ms"] > 0


def test_validation_error_is_logged_as_422(client, pool):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

    rows = _logged_rows(client, pool)
    assert len(rows) == 1
    assert rows[0]["status_code"] == 422
    assert rows[0]["prediction"] is None


def test_other_endpoints_are_not_logged(client, pool):
    client.get("/health")
    assert _logged_rows(client, pool) == []


def test_repeated_request_is_served_from_cache(client, manager):
    text = "Wildfire spreading near the highway"
    first = client.post("/predict", json={"text": text}).json()
    second = client.post("/predict", json={"text": text}).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert sum(len(batch) for batch in manager.calls) == 1


def test_cache_does_not_serve_answers_from_a_previous_model(client, manager):
    text = "Wildfire spreading near the highway"
    client.post("/predict", json={"text": text})

    manager.pending_version = "v2"
    reload = client.post("/admin/reload", headers={"X-Admin-Token": "secret"})
    assert reload.json() == {"changed": True, "model_version": "v2"}

    after = client.post("/predict", json={"text": text}).json()
    assert after["cached"] is False
    assert after["model_version"] == "v2"


def test_reload_requires_admin_token(client):
    assert client.post("/admin/reload").status_code == 403
    assert client.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 403


def test_feedback_is_stored(client, pool):
    request_id = client.post("/predict", json={"text": "flood"}).json()["request_id"]
    response = client.post("/feedback", json={"request_id": request_id, "label": 1})
    assert response.status_code == 200
    assert (UUID(request_id), 1) in pool.executed


def test_feedback_rejects_labels_other_than_0_or_1(client):
    response = client.post(
        "/feedback", json={"request_id": "00000000-0000-0000-0000-000000000000", "label": 2}
    )
    assert response.status_code == 422


def test_feedback_returns_503_without_database(client, monkeypatch):
    async def no_pool():
        return None

    monkeypatch.setattr(main_module, "get_db_pool", no_pool)
    response = client.post(
        "/feedback", json={"request_id": "00000000-0000-0000-0000-000000000000", "label": 1}
    )
    assert response.status_code == 503
