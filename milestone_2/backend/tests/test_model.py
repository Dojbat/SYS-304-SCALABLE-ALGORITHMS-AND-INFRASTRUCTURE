import os

import pytest
from app.features import FEATURE_COLS
from app.model import MODEL_PATH, load_model, predict_one

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"baseline_model.pkl not found at {MODEL_PATH}"
)


def test_model_loads_and_reports_expected_feature_names():
    model = load_model()
    assert list(model.feature_names_in_) == FEATURE_COLS


def test_predict_one_returns_valid_schema():
    result = predict_one("Massive earthquake hits the city, buildings collapsing everywhere!")
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("disaster", "not disaster")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_one_on_clearly_benign_text():
    result = predict_one("just had a great sandwich for lunch, so good")
    assert result["label"] in ("disaster", "not disaster")
