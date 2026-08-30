import os

import pytest
from app.main import app
from app.model import MODEL_PATH
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"baseline_model.pkl not found at {MODEL_PATH}"
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok_and_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_input_returns_200_with_schema(client):
    response = client.post("/predict", json={"text": "Wildfire spreading fast near the highway"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Wildfire spreading fast near the highway"
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disaster", "not disaster")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_empty_text_returns_422(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422
