import os
from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL_NAME = "vinai/bertweet-base"
LABELS = {0: "not disaster", 1: "disaster"}


def _resolve_model_path() -> str:
    """MODEL_PATH env var wins. Otherwise walk up from this file looking for a
    lora_model/ directory (container layout: /app/lora_model, sibling of
    /app/app) or a training/bertweet_lora_final_fulldata/ subfolder (local
    checkout layout: repo_root/training/bertweet_lora_final_fulldata, a
    sibling of milestone_2/).
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return env_path
    directory = Path(__file__).resolve().parent
    for _ in range(6):
        direct = directory / "lora_model"
        if direct.is_dir():
            return str(direct)
        sibling = directory / "training" / "bertweet_lora_final_fulldata"
        if sibling.is_dir():
            return str(sibling)
        directory = directory.parent
    return "/app/lora_model"


MODEL_PATH = _resolve_model_path()


@lru_cache(maxsize=1)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, normalization=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base_model, MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model


def predict_one(text: str) -> dict:
    tokenizer, model = load_model()
    device = next(model.parameters()).device
    encoded = tokenizer([text], truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**encoded).logits, dim=-1)[0]
    prediction = int(probs.argmax())
    confidence = float(probs[prediction])
    return {
        "text": text,
        "prediction": prediction,
        "label": LABELS[prediction],
        "confidence": confidence,
    }
