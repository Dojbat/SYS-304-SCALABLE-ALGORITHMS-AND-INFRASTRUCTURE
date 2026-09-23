# Disaster Tweets — Text Classification

SYS-304 Scalable Algorithms and Infrastructure — **Milestone 1: Problem Scoping & The Prototype**,
**Milestone 2: Deploying the Model Service**, **Milestone 3: Model & Infrastructure Optimization**

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

## Milestone 2: Deploying the Model Service (naive baseline)

The served model is `training/bertweet_lora_final_fulldata` — a LoRA fine-tune of
`vinai/bertweet-base` (accuracy **0.85**, disaster-class F1 **0.83** on held-out validation),
notably stronger than the Milestone 1 baseline (`milestone_1/baseline_model.pkl`, accuracy 0.770,
F1 0.692, kept in the repo for reference but no longer served). It's wrapped in a FastAPI backend
(`milestone_2/backend/`) with a small web frontend (`milestone_2/frontend/`).

`docker-compose.yml` no longer builds this backend — Milestone 3 (below) replaced it as the served
stack. `milestone_2/backend/` and its tests are kept as the "naive" reference implementation that
Milestone 3's model and system optimizations are benchmarked against; run it standalone with
`cd milestone_2/backend && PYTHONPATH=. pytest tests`, or serve it with
`docker build -f milestone_2/backend/Dockerfile .`.

### API (unchanged in Milestone 3 — see below)

`POST /predict`
```json
{ "text": "Massive wildfire forces evacuation of the entire town" }
```
→
```json
{ "text": "...", "prediction": 1, "label": "disaster", "confidence": 0.99 }
```

`GET /health` → `{ "status": "ok", "model_loaded": true }`

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

## Milestone 3: Model & Infrastructure Optimization (Weeks 5–6)

`milestone_3/` is the deployed, optimized system — it's what `docker-compose.yml` builds now. Two
layers of optimization, benchmarked separately (`milestone_3/README.md`) and together (the
`docker compose` stack below):

**Week 5 — model-level** (`milestone_3/export_onnx.py`, full writeup in `milestone_3/README.md`):
merge the LoRA adapter into the base weights, export to ONNX, dynamically quantize to INT8 — a ~7x
inference speedup and ~1.7x memory reduction over the naive FP32 PyTorch+PEFT model from Milestone
2, at a 3.5-point accuracy cost.

**Week 6 — system-level** (`milestone_3/backend/app/`):
- **Dynamic batching** (`app/batcher.py`) — concurrent `/predict` calls arriving within a short
  window (default 20ms, or until 16 requests stack up) are grouped into one `predict_batch()` call
  instead of one model forward pass per request, so throughput under concurrent load scales with
  batch efficiency instead of one-request-at-a-time latency.
- **Exact-match caching** (`app/cache.py`, Redis) — disaster-tweet traffic is bursty and repetitive
  (a viral tweet gets quoted/retweeted verbatim, the same keyword-bearing phrases recur), so a
  cache on the exact request text skips the model entirely for repeat requests.

### System-level benchmark: naive (Milestone 2) vs. optimized (Milestone 3), under concurrent load

Week 5's benchmark (`milestone_3/benchmark.py`) measures the model in-process, one request at a
time — it never shows batching or caching doing anything, because nothing is concurrent. What
actually exercises Week 6's optimizations is concurrent HTTP load, which is what
`milestone_3/backend/loadtest.py` drives: 40 concurrent `POST /predict` requests fired at each
backend's real HTTP endpoint.

| scenario | naive (Milestone 2, no batching/cache) | optimized (Milestone 3) | ratio |
|---|---:|---:|---:|
| 40 concurrent, all unique text (cache miss, batching active) | 12.3 req/s, 2906 ms mean latency | 131.7 req/s, 229 ms mean latency | **10.7x** throughput |
| 40 concurrent, 5 unique texts, cache pre-warmed | *(not applicable — no cache)* | 294.6 req/s, 84 ms mean latency | **23.9x** vs. naive |
| container memory under the load above | 1.19 GiB | 316 MiB backend + 6.8 MiB Redis | **3.7x** smaller |

Reproduce: bring up `docker compose`, then
`python milestone_3/backend/loadtest.py --url http://localhost:8000 --concurrency 40` (unique
traffic) and `... --concurrency 40 --unique 5 --warm-cache` (repeat traffic); for the naive
comparison, `docker build -f milestone_2/backend/Dockerfile -t naive-backend .`, run it on a
different port, and point `--url` at that instead.

The naive backend's collapse under concurrency (12.3 req/s — barely above its ~12 req/s *sequential*
throughput from the Week 5 benchmark) is the actual bottleneck Week 6 targets: FastAPI's default
threadpool accepts concurrent connections, but each one still runs a full, un-batched CPU-bound
model call, so 40 concurrent requests mostly just queue up behind each other (169 OS threads
spawned under load, per `docker stats`, contributing to its 1.19 GiB footprint). Dynamic batching
turns concurrent arrivals into one forward pass instead of N serialized ones; Redis caching skips
the model altogether for repeat text. Neither shows up in a single-request benchmark — both only
matter once traffic is concurrent and repetitive, which is the realistic shape of tweet-monitoring
traffic.

### Architecture

```
 browser ──▶ frontend (nginx:alpine, :3000)
                 │  static index.html/app.js
                 │  proxy_pass /api/ ──▶ backend:8000
                 ▼
             backend (FastAPI + uvicorn, :8000)
                 │  POST /predict → cache lookup (Redis) ──▶ hit: return cached result
                 │                        │ miss
                 │                        ▼
                 │              dynamic batcher (asyncio queue, ≤16 requests / ≤20ms window)
                 │                        │
                 │                        ▼
                 │              BERTweet+LoRA, merged + ONNX + INT8 (onnxruntime, CPU)
                 │                        │
                 │                        ▼
                 │              cache write (Redis) → response
                 │  GET  /health  → model-loaded + Redis-connected check
                 ▼
             redis (redis:7-alpine, exact-match prediction cache, TTL 1h)
```

The browser still only ever talks to nginx — same frontend, same `/api/predict` contract (plus a
`cached: bool` field) as Milestone 2, so no frontend changes were needed.

### Quickstart

```
./deploy.sh
```

Builds all three images (frontend, backend, redis is pulled not built), starts the stack, and
waits for the backend health check — which now also checks Redis connectivity. Then open
<http://localhost:3000>, paste a tweet, and click Classify. Watch requests land in the backend
with:

```
docker compose logs -f backend
```

Tear down with `docker compose down`. Building the backend image regenerates the ONNX/INT8 model
from `training/bertweet_lora_final_fulldata` at build time (`milestone_3/backend/Dockerfile`'s
builder stage runs `export_onnx.py`), so it needs the same network access as Milestone 2's build
did to fetch `vinai/bertweet-base` — expect the first build to take a few minutes longer.

### API

Same as Milestone 2's, plus a `cached` field:

`POST /predict`
```json
{ "text": "Massive wildfire forces evacuation of the entire town" }
```
→
```json
{ "text": "...", "prediction": 1, "label": "disaster", "confidence": 0.99, "cached": false }
```

`GET /health` → `{ "status": "ok", "model_loaded": true, "redis_connected": true }` (`status` is
`"degraded"` if Redis is unreachable — the backend keeps serving predictions without caching
rather than failing outright).

### Tests

- `milestone_3/backend/tests/` — unit tests for the batcher (grouping, batch-size caps, error
  fan-out) and cache (key derivation, hit/miss) using `fakeredis`, an API integration suite with a
  mocked model and `fakeredis`, and a model-loading suite that exercises the real ONNX model when
  `milestone_3/onnx_model/` exists locally (skipped otherwise — it's a build artifact, not checked
  into git). Run with `cd milestone_3/backend && PYTHONPATH=. pytest tests`.
- `milestone_2/tests/test_ui_e2e.py` still drives the same UI against whatever's running on
  `:3000`/`:8000` — unaffected by which backend is behind it.

CI (`.github/workflows/ci.yml`) runs both backends' unit/integration suites (Milestone 3's without
the real ONNX model — the model-dependent tests skip), then a second job that builds the full
`docker compose` stack (frontend + Milestone 3 backend + Redis), smoke-tests `/api/predict` through
nginx, and runs the Playwright test against it.

### Notes

- `milestone_3/backend/app/model.py` only ever loads `onnxruntime` + the tokenizer — no
  `torch`/`peft`/`optimum` at runtime. Those are needed solely to *produce* the ONNX file
  (`export_onnx.py`), so the Dockerfile is a multi-stage build: a builder stage with the full
  ML toolchain runs the export, and the runtime stage copies out only the resulting
  `onnx_model/` directory (~136 MB) plus a slim `onnxruntime`/`transformers`/`fastapi`/`redis`
  dependency set.
- Uvicorn still runs a single worker. Multiple worker processes were considered and rejected for
  this service: the dynamic batcher's whole benefit comes from seeing all concurrent requests in
  one process so it can group them into one forward pass — splitting requests across N worker
  processes would each load a separate copy of the model (N× the memory) and only batch within its
  own slice of traffic, shrinking the batches instead of growing them. `uvicorn[standard]`'s
  default threadpool already lets FastAPI accept new connections while a batch is running in the
  executor.
- Redis failures (connection refused, timeout) are caught around both the cache read and cache
  write in `app/main.py`'s `/predict` handler — a Redis outage degrades to "every request hits the
  model," not a 500.
