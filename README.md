# Disaster Tweets — Text Classification

SYS-304 Scalable Algorithms and Infrastructure — **Milestone 1: Problem Scoping & The Prototype**,
**Milestone 2: Deploying the Model Service**

## Problem

Twitter/X is often faster than official channels at surfacing real disasters as they happen, but
disaster-flavored language is also just... how people talk ("this traffic is a disaster", "my heart
is on fire"). The task is a binary text classification problem: given a tweet, predict whether it
describes a **real disaster** (`target=1`) or not (`target=0`). A system that gets this right could
help emergency services and news organizations monitor social media for real events instead of
noise.

## Dataset

[Kaggle — NLP Getting Started: Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

- `train.csv` — 7,613 labeled tweets: `id`, `keyword`, `location`, `text`, `target`
- `test.csv` — 3,263 unlabeled tweets to predict on
- `sample_submission.csv` — expected submission format

Chosen because it's a clean, well-understood, publicly available text classification problem — a
good target for building system architecture around before scaling up, per the assignment brief.

## Milestone 1 scope

Per the assignment, this milestone is a naive proof-of-concept, not a tuned final model:

1. **EDA** (`milestone_1/main.ipynb`, sections 1–12): class balance, missing values, text length, surface
   features (hashtags/mentions/URLs), the `keyword` column's per-class signal, feature correlations.
2. **Baseline model** (sections 13–17): TF-IDF over `text` + a handful of engineered numeric
   features (length, hashtag/mention/URL counts), fed into an untuned `RandomForestClassifier`. No
   hyperparameter search — a working proof-of-concept that maps tweet text to a prediction.
3. **Saved weights**: `baseline_model.pkl` — the fitted `sklearn` `Pipeline` (TF-IDF vectorizer +
   scaler + classifier) as one pickled unit, reloadable for inference without refitting.

The notebook keeps going well past this (a stronger TF-IDF model, zero-shot LLM prompting with
Qwen, and RoBERTa fine-tuning with LoRA) as an exploration of what later milestones in this course
are presumably about — scaling the naive baseline up. That work is included for context but isn't
the Milestone 1 deliverable; the sections above are.

## Repository contents

- `milestone_1/main.ipynb` — full data pipeline, training, and evaluation (baseline through the
  later exploration sections)
- `milestone_1/baseline_model.pkl` — the saved Milestone 1 baseline model
- `milestone_2/` — the FastAPI backend and web frontend that serve the baseline model (see below)

Raw competition data (`train.csv`, `test.csv`, `sample_submission.csv`) isn't bundled in this repo —
download it from the [competition data page](https://www.kaggle.com/competitions/nlp-getting-started/data)
and place it alongside `main.ipynb` in `milestone_1/` before running.

## Running it

```
pip install pandas numpy scikit-learn matplotlib joblib
jupyter notebook milestone_1/main.ipynb
```

Run top to bottom, or skip straight to loading `baseline_model.pkl` with `joblib.load(...)` to
reproduce predictions without retraining (sections 13–17 only need `pandas`/`numpy`/`scikit-learn`
— the later sections need `torch`, `transformers`, and `peft` as well, and a lot more time/compute).

## Milestone 2: Deploying the Model Service

The served model is `training/bertweet_lora_final_fulldata` — a LoRA fine-tune of
`vinai/bertweet-base` (accuracy **0.85**, disaster-class F1 **0.83** on held-out validation),
notably stronger than the Milestone 1 baseline (`milestone_1/baseline_model.pkl`, accuracy 0.770,
F1 0.692, kept in the repo for reference but no longer served). It's wrapped in a FastAPI backend
(`milestone_2/backend/`) with a small web frontend (`milestone_2/frontend/`), both containerized
and wired together with `docker compose`.

### Architecture

```
 browser ──▶ frontend (nginx:alpine, :3000)
                 │  static index.html/app.js
                 │  proxy_pass /api/ ──▶ backend:8000
                 ▼
             backend (FastAPI + uvicorn, :8000)
                 │  POST /predict  → tokenize → BERTweet + LoRA adapter → label + confidence
                 │  GET  /health   → liveness + model-loaded check
```

The browser never talks to the backend directly — nginx proxies `/api/*` to the backend container,
so there's no CORS configuration and no hardcoded ports in the frontend JS.

### Quickstart

```
./deploy.sh
```

Builds both images, starts the stack, and waits for the backend health check. Then open
<http://localhost:3000>, paste a tweet, and click Classify. Watch requests land in the backend with:

```
docker compose logs -f backend
```

Tear down with `docker compose down`.

### API

`POST /predict`
```json
{ "text": "Massive wildfire forces evacuation of the entire town" }
```
→
```json
{ "text": "...", "prediction": 1, "label": "disaster", "confidence": 0.99 }
```

`GET /health` → `{ "status": "ok", "model_loaded": true }`

### Tests

- `milestone_2/backend/tests/` — unit tests for the feature-engineering pipeline and the model
  wrapper, plus integration tests for `/predict` and `/health` (from `milestone_2/backend/`, run
  `PYTHONPATH=. pytest tests`).
- `milestone_2/tests/test_ui_e2e.py` — a Playwright test that drives the real UI against a running
  `docker compose` stack (`pytest milestone_2/tests/test_ui_e2e.py -m e2e` from the repo root).

CI (`.github/workflows/ci.yml`) runs `ruff` + the unit/integration suite on every push, then a
second job that builds both images, brings up the stack, smoke-tests `/api/predict` through nginx,
and runs the Playwright test against it.

### Notes

- The served model is a PEFT LoRA adapter (`training/bertweet_lora_final_fulldata/`) on top of the
  `vinai/bertweet-base` checkpoint, loaded via `transformers` + `peft` in
  `milestone_2/backend/app/model.py`. The base checkpoint's weights are baked into the Docker image
  at build time (see the backend `Dockerfile`) so the container doesn't need network access at
  runtime.
- `milestone_2/backend/app/features.py` is the Milestone 1 baseline's feature-engineering pipeline
  (`text` + 7 engineered numeric columns for `baseline_model.pkl`). It's no longer used by the
  serving path but is kept, with its tests, as a record of the Milestone 1 deliverable.
- The `bertweet_lora_final_fulldata` checkpoint originally saved its tokenizer without the
  fast-tokenizer `tokenizer.json` file, and its `vocab.txt`/`bpe.codes` were rewritten by
  `save_pretrained` into a format its own "custom" backend can't parse — reloading it standalone
  silently degraded to character-level tokenization and made the model collapse to always
  predicting "not disaster". Fixed by restoring the canonical `tokenizer.json`/`vocab.txt`/
  `bpe.codes` from `vinai/bertweet-base` into the checkpoint directory (the tokenizer itself was
  never fine-tuned, so this is lossless).
