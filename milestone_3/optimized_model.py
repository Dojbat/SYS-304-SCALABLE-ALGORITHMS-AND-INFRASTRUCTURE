"""Inference wrapper for the optimized (merged + ONNX + INT8) model.

Mirrors milestone_2/backend/app/model.py's interface (load_model / predict_one)
so the two can be swapped and benchmarked against each other.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

LABELS = {0: "not disaster", 1: "disaster"}
MAX_LENGTH = 128

ONNX_DIR = Path(__file__).resolve().parent / "onnx_model"
MODEL_PATH = ONNX_DIR / "model_int8.onnx"


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR), normalization=True)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
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
