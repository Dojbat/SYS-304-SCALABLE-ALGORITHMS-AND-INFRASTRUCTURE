"""One retraining cycle: fetch labelled data, fine-tune, evaluate, and deploy
the candidate if it beats the current model.

    fetch  -> labelled live data from Postgres + original train.csv
    train  -> continue fine-tuning the current model's LoRA adapter (GPU)
    eval   -> current vs. candidate on held-out live data and the guard set
    export -> merge + ONNX + INT8 in the exporter container
    verify -> INT8 model scored again, so a bad quantization can't ship
    deploy -> move into models/vN, point registry.json at it, tell the API

Every cycle, whatever its outcome, is recorded in the retraining_runs table.
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import data
import httpx
import modeling
import numpy as np
import psycopg
from evaluate import DeployPolicy, Metrics, int8_ok, score, should_deploy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "milestone_4" / "backend"))
from app.registry import MODEL_FILE, read_registry, write_registry  # noqa: E402

ORIGINAL_ADAPTER = REPO_ROOT / "training" / "bertweet_lora_final_fulldata"
STAGING = "_staging"

logger = logging.getLogger("retraining")


@dataclass
class PipelineConfig:
    database_url: str
    backend_url: str
    admin_token: str
    models_dir: Path = REPO_ROOT / "milestone_4" / "models"
    train_csv: Path = REPO_ROOT / "train.csv"
    device: str = "auto"
    train: modeling.TrainConfig = field(default_factory=modeling.TrainConfig)
    policy: DeployPolicy = field(default_factory=DeployPolicy)
    budget: data.DataBudget = field(default_factory=data.DataBudget)


def adapter_path(models_dir: Path, version_info: dict) -> Path:
    adapter_dir = version_info.get("adapter_dir")
    return models_dir / adapter_dir if adapter_dir else ORIGINAL_ADAPTER


def next_version(registry: dict) -> str:
    numbers = [int(v[1:]) for v in registry["versions"] if v[1:].isdigit()]
    return f"v{max(numbers, default=0) + 1}"


def run_exporter(adapter_container_path: str, out_container_path: str) -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            "--profile",
            "tools",
            "run",
            "--rm",
            "exporter",
            "--adapter",
            adapter_container_path,
            "--out",
            out_container_path,
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def int8_probs(model_dir: Path, texts: list[str]) -> np.ndarray:
    """P(disaster) from the exported INT8 model, run exactly as the API runs it."""
    import onnxruntime as ort

    tokenizer = modeling.load_tokenizer(model_dir)
    session = ort.InferenceSession(str(model_dir / MODEL_FILE), providers=["CPUExecutionProvider"])
    out = []
    for i in range(0, len(texts), 64):
        encoded = tokenizer(
            texts[i : i + 64],
            truncation=True,
            max_length=modeling.MAX_LENGTH,
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
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        out.append((exp / exp.sum(axis=-1, keepdims=True))[:, 1])
    return np.concatenate(out)


def notify_backend(config: PipelineConfig) -> str:
    try:
        response = httpx.post(
            f"{config.backend_url}/admin/reload",
            headers={"X-Admin-Token": config.admin_token},
            timeout=60,
        )
        response.raise_for_status()
        return f"API now serving {response.json()['model_version']}"
    except httpx.HTTPError as exc:
        # not fatal: the API also polls registry.json
        return f"reload call failed ({exc}); API will pick it up on its next registry poll"


class RunRecorder:
    """Writes the retraining_runs row as the cycle progresses."""

    def __init__(self, conn: psycopg.Connection, trigger: str, started_at: datetime) -> None:
        self.conn = conn
        self.run_id = conn.execute(
            "INSERT INTO retraining_runs (started_at, trigger, status) "
            "VALUES (%s, %s, 'running') RETURNING id",
            (started_at, trigger),
        ).fetchone()[0]
        conn.commit()

    def update(self, **fields) -> None:
        columns = ", ".join(f"{name} = %s" for name in fields)
        self.conn.execute(
            f"UPDATE retraining_runs SET {columns} WHERE id = %s",
            (*fields.values(), self.run_id),
        )
        self.conn.commit()


def previous_cutoff(conn: psycopg.Connection) -> datetime | None:
    row = conn.execute(
        "SELECT max(data_cutoff) FROM retraining_runs WHERE data_cutoff IS NOT NULL"
    ).fetchone()
    return row[0] if row else None


def model_cutoff(conn: psycopg.Connection, version_info: dict) -> datetime | None:
    """Data cutoff of the run that produced this model (None for the seed
    model): everything before it is already in the model's weights."""
    run_id = version_info.get("run_id")
    if run_id is None:
        return None
    row = conn.execute(
        "SELECT data_cutoff FROM retraining_runs WHERE id = %s", (run_id,)
    ).fetchone()
    return row[0] if row else None


def run_once(config: PipelineConfig, trigger: str) -> str:
    """Runs one cycle and returns its final status."""
    with psycopg.connect(config.database_url) as conn:
        started_at = datetime.now(UTC)
        recorder = RunRecorder(conn, trigger, started_at)
        try:
            status, message = _run(config, conn, recorder, started_at)
        except Exception as exc:
            logger.exception("retraining run %d failed", recorder.run_id)
            status, message = "failed", f"{type(exc).__name__}: {exc}"
        recorder.update(status=status, message=message, finished_at=datetime.now(UTC))
        logger.info("run %d %s: %s", recorder.run_id, status, message)
        return status


def _run(
    config: PipelineConfig, conn: psycopg.Connection, recorder: RunRecorder, cutoff: datetime
) -> tuple[str, str]:
    registry = read_registry(config.models_dir)
    if registry is None:
        return "failed", f"no registry in {config.models_dir}; start the stack first"
    base_version = registry["current"]
    base_adapter = adapter_path(config.models_dir, registry["versions"][base_version])
    recorder.update(base_version=base_version)

    # ---- fetch (every query bounded by the budget) ----
    budget = config.budget
    since = data.window_start(
        cutoff, model_cutoff(conn, registry["versions"][base_version]), budget
    )
    cycle = data.build(
        recent_train=data.fetch_recent_train(conn, since, cutoff, budget.max_recent),
        recent_eval=data.fetch_recent_eval(conn, since, cutoff, budget.max_eval),
        older_train=data.fetch_older_train(conn, since, budget.max_older_live),
        original=data.load_original(config.train_csv),
        budget=budget,
        seed=recorder.run_id,
    )
    train_set = cycle.train
    logger.info("data since %s: %s", since.isoformat(timespec="seconds"), cycle.stats)
    recorder.update(
        data_cutoff=cutoff,
        n_new_labeled=data.count_new_labels(conn, previous_cutoff(conn)),
        n_train=len(train_set),
        n_eval=len(cycle.live_eval),
    )
    if len(cycle.live_eval) < config.policy.min_eval_examples or not cycle.live_train:
        return "skipped", (
            f"not enough labelled live data ({len(cycle.live_train)} train, "
            f"{len(cycle.live_eval)} eval)"
        )

    # ---- train ----
    device = modeling.pick_device(config.device)
    tokenizer, model = modeling.load(base_adapter, device, trainable=True)
    eval_texts = [e.text for e in cycle.live_eval]
    eval_labels = [e.label for e in cycle.live_eval]
    guard_texts = [e.text for e in cycle.guard]
    guard_labels = [e.label for e in cycle.guard]

    base_live = score(modeling.predict_proba(tokenizer, model, eval_texts, device), eval_labels)
    base_guard = score(modeling.predict_proba(tokenizer, model, guard_texts, device), guard_labels)

    t0 = time.perf_counter()
    losses = modeling.fine_tune(
        tokenizer,
        model,
        [e.text for e in train_set],
        [e.label for e in train_set],
        device,
        config.train,
    )
    train_s = time.perf_counter() - t0

    # ---- evaluate ----
    cand_live = score(modeling.predict_proba(tokenizer, model, eval_texts, device), eval_labels)
    cand_guard = score(modeling.predict_proba(tokenizer, model, guard_texts, device), guard_labels)
    recorder.update(
        base_f1=base_live.f1,
        candidate_f1=cand_live.f1,
        base_guard_acc=base_guard.accuracy,
        candidate_guard_acc=cand_guard.accuracy,
    )
    logger.info(
        "trained on %d examples in %.1fs on %s (loss %s); live F1 %.3f -> %.3f, "
        "guard acc %.3f -> %.3f",
        len(train_set),
        train_s,
        device,
        [round(x, 3) for x in losses],
        base_live.f1,
        cand_live.f1,
        base_guard.accuracy,
        cand_guard.accuracy,
    )
    deploy, reason = should_deploy(base_live, cand_live, base_guard, cand_guard, config.policy)
    if not deploy:
        return "rejected", reason

    # ---- export + verify ----
    version = next_version(registry)
    staging = config.models_dir / STAGING / version
    shutil.rmtree(staging, ignore_errors=True)
    modeling.save_adapter(model, base_adapter, staging / "adapter")
    del model
    run_exporter(f"/models/{STAGING}/{version}/adapter", f"/models/{STAGING}/{version}")

    int8_live = score(int8_probs(staging, eval_texts), eval_labels)
    recorder.update(candidate_int8_f1=int8_live.f1)
    ok, int8_reason = int8_ok(cand_live, int8_live, config.policy)
    if not ok:
        shutil.rmtree(staging, ignore_errors=True)
        return "rejected", int8_reason

    # ---- deploy ----
    final_dir = config.models_dir / version
    os.replace(staging, final_dir)
    registry = read_registry(config.models_dir)
    registry["versions"][version] = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": "retrain",
        "parent": base_version,
        "run_id": recorder.run_id,
        "adapter_dir": f"{version}/adapter",
        "metrics": _metrics_json(base_live, cand_live, base_guard, cand_guard, int8_live),
    }
    registry["current"] = version
    write_registry(config.models_dir, registry)
    recorder.update(candidate_version=version)

    return "deployed", f"{reason}; {int8_reason}; {notify_backend(config)}"


def _metrics_json(base_live, cand_live, base_guard, cand_guard, int8_live) -> dict:
    def m(x: Metrics) -> dict:
        return {"f1": round(x.f1, 4), "accuracy": round(x.accuracy, 4), "n": x.n}

    return {
        "base_live": m(base_live),
        "candidate_live": m(cand_live),
        "candidate_live_int8": m(int8_live),
        "base_guard": m(base_guard),
        "candidate_guard": m(cand_guard),
    }
