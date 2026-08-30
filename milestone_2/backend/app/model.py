import os
from functools import lru_cache
from pathlib import Path

import joblib

from app.features import build_features

LABELS = {0: "not disaster", 1: "disaster"}


def _resolve_model_path() -> str:
    """MODEL_PATH env var wins. Otherwise walk up from this file looking for
    baseline_model.pkl, checking both the directory itself (container layout:
    /app/baseline_model.pkl, sibling of /app/app) and a milestone_1/ subfolder
    (local checkout layout: repo_root/milestone_1/baseline_model.pkl, a sibling
    of milestone_2/).
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return env_path
    directory = Path(__file__).resolve().parent
    for _ in range(6):
        direct = directory / "baseline_model.pkl"
        if direct.exists():
            return str(direct)
        sibling = directory / "milestone_1" / "baseline_model.pkl"
        if sibling.exists():
            return str(sibling)
        directory = directory.parent
    return "/app/baseline_model.pkl"


MODEL_PATH = _resolve_model_path()


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def predict_one(text: str) -> dict:
    model = load_model()
    features = build_features([text])
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    confidence = float(proba[prediction])
    return {
        "text": text,
        "prediction": prediction,
        "label": LABELS[prediction],
        "confidence": confidence,
    }
