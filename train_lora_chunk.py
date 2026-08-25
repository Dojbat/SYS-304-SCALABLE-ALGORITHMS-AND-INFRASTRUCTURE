"""
Resumable LoRA fine-tuning for the disaster-tweets classifier.

Designed to be invoked repeatedly (each call bounded to MAX_WALL_SECONDS) since this
CPU-only environment appears to kill background processes somewhere around 20-25 minutes.
Each call: resumes from the last checkpoint if one exists, trains until the time budget
is hit or all EPOCHS are done, evaluates + checkpoints at epoch boundaries, and always
saves a resumable in-progress checkpoint before exiting.

Run this repeatedly:  python train_lora_chunk.py
It prints COMPLETE when done, or PARTIAL (call it again) otherwise.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

MAX_WALL_SECONDS = 480   # ~8 min of work per invocation, safely under the observed ~20-25 min cap
BATCH_SIZE = 8
MAX_LENGTH = 64
EPOCHS = 3
LR = 2e-4

PROGRESS_PATH = "roberta_lora_progress.json"
INPROGRESS_DIR = "roberta_lora_inprogress"
OPT_STATE_PATH = "roberta_lora_inprogress_optim.pt"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    train = pd.read_csv("train.csv")
    y = train["target"]
    train_idx, val_idx = train_test_split(train.index, test_size=0.2, random_state=42, stratify=y)
    return train, train_idx, val_idx


def evaluate(model, tok, texts, batch_size=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            out = model(**enc)
            preds.extend(out.logits.argmax(dim=-1).tolist())
    model.train()
    return preds


def main():
    t_start = time.time()
    train, train_idx, val_idx = load_data()
    train_texts = train.loc[train_idx, "text"].tolist()
    train_labels = train.loc[train_idx, "target"].tolist()
    val_texts = train.loc[val_idx, "text"].tolist()
    val_labels = train.loc[val_idx, "target"].tolist()

    n_train = len(train_texts)
    steps_per_epoch = n_train // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS

    tok = AutoTokenizer.from_pretrained("roberta-base")

    if os.path.exists(PROGRESS_PATH):
        progress = json.load(open(PROGRESS_PATH))
        log(f"resuming from progress: epoch={progress['epoch']} step_in_epoch={progress['step_in_epoch']} "
            f"global_step={progress['global_step']}")
        base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        model = PeftModel.from_pretrained(base_model, INPROGRESS_DIR, is_trainable=True)
    else:
        log("starting fresh")
        progress = {"epoch": 1, "step_in_epoch": 0, "global_step": 0, "epoch_results": []}
        base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    if os.path.exists(OPT_STATE_PATH):
        state = torch.load(OPT_STATE_PATH)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    model.train()
    epoch = progress["epoch"]
    step_in_epoch = progress["step_in_epoch"]
    global_step = progress["global_step"]
    epoch_results = progress["epoch_results"]

    def save_checkpoint(epoch_, step_in_epoch_, global_step_):
        model.save_pretrained(INPROGRESS_DIR)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, OPT_STATE_PATH)
        json.dump(
            {"epoch": epoch_, "step_in_epoch": step_in_epoch_, "global_step": global_step_,
             "epoch_results": epoch_results},
            open(PROGRESS_PATH, "w"),
        )

    while epoch <= EPOCHS:
        perm = np.random.RandomState(epoch).permutation(n_train)
        running_loss = 0.0
        n_this_call = 0
        for s in range(step_in_epoch, steps_per_epoch):
            idx = perm[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            batch_texts = [train_texts[i] for i in idx]
            batch_labels = torch.tensor([train_labels[i] for i in idx])

            enc = tok(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            out = model(**enc, labels=batch_labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            n_this_call += 1
            global_step += 1

            if n_this_call % 50 == 0:
                log(f"epoch {epoch} step {s + 1}/{steps_per_epoch} "
                    f"avg_loss(this call)={running_loss / n_this_call:.4f}")

            if time.time() - t_start > MAX_WALL_SECONDS:
                save_checkpoint(epoch, s + 1, global_step)
                log(f"time budget hit — checkpointed at epoch {epoch} step {s + 1}/{steps_per_epoch}. "
                    f"PARTIAL — run again to continue.")
                return

        # finished this epoch
        val_preds = evaluate(model, tok, val_texts)
        val_f1 = f1_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        log(f"epoch {epoch} COMPLETE — val accuracy {val_acc:.3f}  val f1 {val_f1:.3f}")

        ckpt_path = f"roberta_lora_epoch{epoch}"
        model.save_pretrained(ckpt_path)
        epoch_results.append({"epoch": epoch, "val_f1": val_f1, "val_acc": val_acc, "checkpoint": ckpt_path})

        epoch += 1
        step_in_epoch = 0
        save_checkpoint(epoch, step_in_epoch, global_step)

        if time.time() - t_start > MAX_WALL_SECONDS:
            log(f"time budget hit after finishing an epoch. PARTIAL — run again to continue.")
            return

    log("ALL EPOCHS COMPLETE")
    print("epoch_results:", json.dumps(epoch_results, indent=2))
    print("STATUS: COMPLETE")


if __name__ == "__main__":
    main()
