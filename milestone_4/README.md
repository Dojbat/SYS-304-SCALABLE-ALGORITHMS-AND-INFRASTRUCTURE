# Milestone 4: Observability, Monitoring and Automated Retraining (Weeks 7–8)

The Milestone 3 service (BERTweet + LoRA, exported to INT8 ONNX, with dynamic
batching and a Redis cache) is extended so it can be watched while it runs and
can improve itself when live traffic drifts away from its training data.

The design document (domain, problem, design decisions, architecture diagram,
scaling bottlenecks) is submitted separately as a PDF.

```
                        ┌──────────────┐  /api  ┌─────────────────────────────────┐
 browser ──────────────▶│ frontend     │───────▶│ backend (FastAPI)               │
 workload generator ───────────────────────────▶│  /predict  /feedback  /health   │
                        └──────────────┘        │  batcher · cache · request log  │
                                                └───┬──────────┬─────────┬────────┘
                                         cache      │          │ COPY     │ reads
                                        ┌───────────▼──┐  ┌────▼─────┐  ┌▼──────────────────┐
                                        │ Redis        │  │ Postgres │  │ models/ (shared)  │
                                        └──────────────┘  │ requests │  │ registry.json     │
                                                          │ feedback │  │ v1/ v2/ ...       │
 dashboard (Streamlit) ──────────── reads ───────────────▶│ runs     │  └▲──────────────────┘
                                                          └────▲─────┘   │ writes vN, flips
                                                               │         │ "current"
                                         retraining (host, GPU)┴─────────┘
                                         trigger → fetch → fine-tune → eval
                                         → export (exporter container) → verify → deploy
```

## What's here

| Path | What it does |
|---|---|
| `backend/` | The Milestone 3 API plus request logging to Postgres, `POST /feedback`, `POST /admin/reload` and hot model reload from the shared registry |
| `db/schema.sql` | `requests`, `feedback` and `retraining_runs` tables |
| `dashboard/` | Streamlit dashboard: latency, throughput, error rate, confidence, data drift, model and retraining history |
| `dashboard/build_reference.py` | Builds `reference_stats.json`, the training-data profile drift is measured against |
| `workload/generate.py` | Workload generator: normal, drifted and invalid traffic, plus delayed ground-truth labels |
| `workload/drift_texts.py` | Synthetic drifted tweets (Spanish / Indonesian / Tagalog, and English slang) with known labels |
| `retraining/retrain.py` | Trigger loop (`watch`), one-off cycle (`run`), and `status` |
| `retraining/pipeline.py` | One cycle: fetch → fine-tune on GPU → evaluate → export → verify INT8 → deploy |
| `retraining/export_onnx.py` | LoRA merge → ONNX → INT8, run in the `exporter` container |
| `models/` | The shared model registry (gitignored, created at runtime) |
| `lb/nginx.conf` | Load balancer on :8000 in front of the backend replicas |
| `lb/frontend.conf` | Frontend nginx config under compose: `/api` goes through the load balancer |

## Scaling, step 1 (built in)

* **Backend replicas:** `BACKEND_REPLICAS` (default 2) behind an nginx load balancer on :8000.
  Replicas are stateless (cache, logs, labels and models are shared), and `X-Served-By` in each
  response names the replica. Restart replicas one at a time to avoid a brief 502.
* **Bounded storage:** `retrain.py maintenance` (hourly inside `watch`) deletes requests older
  than `--retention-days` (30) that never got a label. Labelled rows are kept; training reads
  them through the `labeled_examples` view.
* **Fixed training budget per cycle:**
  * a recent window since the current model's data cutoff (mistakes first, then least
    confident);
  * capped older-live and `train.csv` replay;
  * class balancing;
  * a total cap (`--max-train`, default 20k) that never cuts replay below 500.

  Cost per cycle stays constant as data grows.


## Running it

Kaggle's `train.csv` and `test.csv` must be in the repo root (the workload
generator samples them and retraining replays `train.csv`).

```bash
# 1. the stack: API, frontend, Redis, Postgres, dashboard
docker compose up --build -d --wait
docker compose --profile tools build exporter   # once; used by retraining

#    API        http://localhost:8000/health
#    frontend   http://localhost:3000
#    dashboard  http://localhost:8501

# 2. the retraining watcher, on the host so it can use the GPU
pip install -r milestone_4/retraining/requirements.txt   # CUDA torch first, see the file
python milestone_4/retraining/retrain.py watch --poll 15 --min-new-labels 300 --cooldown 120

# 3. traffic: first normal, then drifted
python milestone_4/workload/generate.py --duration 120 --rate 20
python milestone_4/workload/generate.py --duration 240 --rate 20 --drift-share 0.9
```

Watch the dashboard while step 3 runs: the drift tab lights up as soon as
drifted traffic starts, labelled accuracy falls, the watcher fires an
`accuracy_drop` retrain, and the served model switches from `v1` to `v2`
without a restart.

Other commands:

```bash
python milestone_4/retraining/retrain.py status              # what the trigger sees right now
python milestone_4/retraining/retrain.py run --trigger manual  # one cycle, no trigger needed
python milestone_4/retraining/retrain.py maintenance --dry-run # retention report
BACKEND_REPLICAS=3 docker compose up -d --wait               # scale the API
python milestone_4/workload/generate.py --duration 60 --error-share 0.3   # error-rate spike
```

### Rolling back

Edit `milestone_4/models/registry.json` and set `"current"` back to an older
version. The API picks the change up within `MODEL_POLL_SECONDS` (10 s), or
immediately with `curl -X POST -H "X-Admin-Token: change-me" localhost:8000/admin/reload`.

## Tests

```bash
cd milestone_4/backend && PYTHONPATH=. pytest tests   # API, logging, registry, batcher
cd milestone_4 && pytest tests                        # drift, workload, triggers, data split, deploy rule
```

## Demo run

20 requests/s with 2% invalid requests, watcher thresholds as in step 2 above. Full analysis is in
the design document.

| Phase | Traffic | Served | Labelled accuracy | Mean confidence | p50 / p95 |
|---|---|---|---|---|---|
| 1 | normal | v1 | 0.859 | 0.910 | 75 / 94 ms |
| 2 | 90% drifted | v1 | 0.731 | 0.916 | 74 / 100 ms |
| 2b | 90% drifted | v1 → v2 | 0.716 → 0.985 | 0.913 → 0.971 | 75 / 91 ms |
| 3 | normal | v2 | 0.854 | 0.898 | 74 / 90 ms |

- **Drift detected:** out-of-vocabulary rate 12% → 46%; PSI on every text feature > 0.25 (major).
- **Confidence missed it:** mean confidence stayed at 0.91–0.92 while accuracy fell 13 points,
  because the model is confidently wrong about non-English tweets. That's why the pipeline also
  triggers on labelled accuracy.
- **Retraining runs:**
  1. *Rejected:* fired before enough drifted labels existed; F1 gain +0.003 < 0.01.
  2. *Failed:* the INT8 check caught a tokenizer mismatch (since fixed), so nothing shipped.
  3. *Deployed v2:* held-out live F1 0.714 → 0.964 (INT8 0.941), guard accuracy 0.888 → 0.882,
     76 s from trigger to serving.
- **Logging:** all 12,599 requests were logged; none dropped.
