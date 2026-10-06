"""Inference for the merged + ONNX + INT8 model, with hot reload.

Same batched predict shape as milestone_3/backend/app/model.py, but the
model is no longer fixed at import time: a ModelManager holds whichever
version registry.json currently points at, and swaps to a new one when the
retraining pipeline deploys it. The swap replaces a single attribute, so a
batch that's already running keeps using the model it started with.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import BertweetTokenizer

from app.registry import MODEL_FILE, ModelRef, ensure_seeded, resolve_current

logger = logging.getLogger("disaster-tweet-api")

LABELS = {0: "not disaster", 1: "disaster"}
MAX_LENGTH = 128


@dataclass(frozen=True)
class LoadedModel:
    version: str
    tokenizer: object
    session: ort.InferenceSession


def _load(ref: ModelRef) -> LoadedModel:
    # explicit class rather than AutoTokenizer, which resolves differently
    # across transformers versions in a folder holding config.json
    tokenizer = BertweetTokenizer.from_pretrained(str(ref.path), normalization=True)
    session = ort.InferenceSession(str(ref.path / MODEL_FILE), providers=["CPUExecutionProvider"])
    return LoadedModel(version=ref.version, tokenizer=tokenizer, session=session)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class ModelManager:
    def __init__(self, models_dir: Path, seed_dir: Path) -> None:
        self.models_dir = models_dir
        self.seed_dir = seed_dir
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._current is not None

    @property
    def version(self) -> str | None:
        return self._current.version if self._current else None

    def reload_if_changed(self) -> bool:
        """Load the registry's current version if it isn't the one being
        served. Blocking (reads ~136 MB from disk), so call it from a worker
        thread. Returns True if the served model changed."""
        with self._lock:
            ensure_seeded(self.models_dir, self.seed_dir)
            ref = resolve_current(self.models_dir, self.seed_dir)
            if self._current is not None and self._current.version == ref.version:
                return False
            new_model = _load(ref)
            previous = self.version
            self._current = new_model
        logger.info("serving model %s (was %s)", ref.version, previous)
        return True

    def predict_batch(self, texts: list[str]) -> list[dict]:
        model = self._current
        if model is None:
            raise RuntimeError("model not loaded")
        encoded = model.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="np",
        )
        logits = model.session.run(
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
                "model_version": model.version,
            }
            for i, (text, pred) in enumerate(zip(texts, predictions, strict=True))
        ]
