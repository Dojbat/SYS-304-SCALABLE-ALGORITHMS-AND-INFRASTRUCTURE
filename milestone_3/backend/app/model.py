"""Inference for the merged + ONNX + INT8 model produced by
milestone_3/export_onnx.py. Same predict_one/predict_batch/load_model shape
as milestone_2/backend/app/model.py, but backed by onnxruntime instead of
eager PyTorch, and built for batched calls (the dynamic batcher always calls
predict_batch, even for a batch of one).
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

LABELS = {0: "not disaster", 1: "disaster"}
MAX_LENGTH = 128


def _resolve_onnx_dir() -> str:
    """ONNX_MODEL_PATH env var wins. Otherwise walk up from this file looking
    for an onnx_model/ directory (container layout: /app/onnx_model, sibling
    of /app/app) or milestone_3/onnx_model (local checkout layout).
    """
    env_path = os.environ.get("ONNX_MODEL_PATH")
    if env_path:
        return env_path
    directory = Path(__file__).resolve().parent
    for _ in range(6):
        direct = directory / "onnx_model"
        if direct.is_dir():
            return str(direct)
        sibling = directory / "milestone_3" / "onnx_model"
        if sibling.is_dir():
            return str(sibling)
        directory = directory.parent
    return "/app/onnx_model"


ONNX_DIR = _resolve_onnx_dir()
MODEL_PATH = str(Path(ONNX_DIR) / "model_int8.onnx")


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR, normalization=True)
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return tokenizer, session


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_batch(texts: list[str]) -> list[dict]:
    tokenizer, session = load_model()
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="np",
    )
    logits = session.run(
        ["logits"],
        {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        },
    )[0]
    probs = _softmax(logits)
    predictions = probs.argmax(axis=-1)
    return [
        {
            "text": text,
            "prediction": int(pred),
            "label": LABELS[int(pred)],
            "confidence": float(probs[i, pred]),
        }
        for i, (text, pred) in enumerate(zip(texts, predictions, strict=True))
    ]


def predict_one(text: str) -> dict:
    return predict_batch([text])[0]
