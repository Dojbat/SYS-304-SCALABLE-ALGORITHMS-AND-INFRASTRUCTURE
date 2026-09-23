import app.main as main_module
import fakeredis
import pytest
from app.batcher import DynamicBatcher
from app.cache import PredictionCache
from fastapi.testclient import TestClient


def _fake_predict_batch(call_log: list[list[str]]):
    def predict_batch(texts: list[str]) -> list[dict]:
        call_log.append(list(texts))
        return [
            {
                "text": t,
                "prediction": 1 if "fire" in t else 0,
                "label": "disaster" if "fire" in t else "not disaster",
                "confidence": 0.9,
            }
            for t in texts
        ]

    return predict_batch


@pytest.fixture
def call_log() -> list[list[str]]:
    return []


@pytest.fixture
def client(monkeypatch, call_log):
    monkeypatch.setattr(main_module, "load_model", lambda: None)
    monkeypatch.setattr(
        main_module, "batcher", DynamicBatcher(_fake_predict_batch(call_log), max_wait_ms=5)
    )
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(main_module, "redis_client", fake_redis)
    monkeypatch.setattr(main_module, "cache", PredictionCache(fake_redis, ttl_seconds=60))

    with TestClient(main_module.app) as c:
        yield c


def test_health_returns_ok_with_model_and_redis_up(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "model_loaded": True, "redis_connected": True}


def test_predict_valid_input_returns_200_with_schema(client):
    response = client.post("/predict", json={"text": "Wildfire spreading near the highway"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Wildfire spreading near the highway"
    assert body["prediction"] == 1
    assert body["label"] == "disaster"
    assert body["cached"] is False


def test_repeated_request_is_served_from_cache(client, call_log):
    text = "Wildfire spreading near the highway"
    first = client.post("/predict", json={"text": text}).json()
    second = client.post("/predict", json={"text": text}).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["prediction"] == first["prediction"]
    # only the first request should have reached the model
    assert sum(len(batch) for batch in call_log) == 1


def test_predict_empty_text_returns_422(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
