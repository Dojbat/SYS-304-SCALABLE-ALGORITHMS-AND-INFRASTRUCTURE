"""PyTorch side of retraining: load a LoRA adapter on top of BERTweet,
fine-tune it, score it. Runs on CUDA when available (the serving stack stays
CPU-only; only training needs the GPU).
"""

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    BertweetTokenizer,
    get_linear_schedule_with_warmup,
)

BASE_MODEL_NAME = "vinai/bertweet-base"
MAX_LENGTH = 128

# Copied from the source adapter rather than written with save_pretrained,
# which rewrites BERTweet's vocab.txt/bpe.codes into a format its own loader
# can't read back (see milestone_3/export_onnx.py).
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "bpe.codes",
    "added_tokens.json",
]


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    seed: int = 42


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_tokenizer(model_dir: Path) -> BertweetTokenizer:
    """Always the BERTweet tokenizer class, never AutoTokenizer: in a folder
    that also holds the exported config.json (model_type "roberta"), newer
    transformers versions resolve AutoTokenizer to a generic tokenizer with a
    different vocabulary (66,050 ids vs. BERTweet's 64,001), which feeds the
    model out-of-range token ids."""
    return BertweetTokenizer.from_pretrained(str(model_dir), normalization=True)


def load(adapter_dir: Path, device: torch.device, trainable: bool = False):
    tokenizer = load_tokenizer(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)
    model.to(device)
    return tokenizer, model


@torch.no_grad()
def predict_proba(
    tokenizer, model, texts: list[str], device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """P(disaster) for each text."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[i : i + batch_size],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**encoded).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fine_tune(
    tokenizer, model, texts: list[str], labels: list[int], device: torch.device, config: TrainConfig
) -> list[float]:
    """Continue training the adapter in place. Returns mean loss per epoch."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    examples = list(zip(texts, labels, strict=True))

    def collate(batch):
        batch_texts, batch_labels = zip(*batch, strict=True)
        encoded = tokenizer(
            list(batch_texts),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(batch_labels)
        return encoded

    loader = DataLoader(examples, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    total_steps = len(loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps
    )

    model.train()
    epoch_losses = []
    for _ in range(config.epochs):
        losses = []
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        epoch_losses.append(sum(losses) / len(losses))
    model.eval()
    return epoch_losses


def save_adapter(model, source_adapter_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    for name in TOKENIZER_FILES:
        shutil.copy(source_adapter_dir / name, out_dir / name)
