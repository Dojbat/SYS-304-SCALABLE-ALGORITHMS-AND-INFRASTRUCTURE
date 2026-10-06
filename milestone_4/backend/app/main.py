import asyncio
import contextlib
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, Header, HTTPException, Request

from app.batcher import DynamicBatcher
from app.cache import PredictionCache
from app.model import ModelManager
from app.request_log import RequestLogger
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("disaster-tweet-api")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "16"))
BATCH_MAX_WAIT_MS = int(os.environ.get("BATCH_MAX_WAIT_MS", "20"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
SEED_MODEL_DIR = Path(os.environ.get("SEED_MODEL_DIR", "/app/seed_model"))
MODEL_POLL_SECONDS = float(os.environ.get("MODEL_POLL_SECONDS", "10"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
DB_RETRY_SECONDS = 5.0
REPLICA = socket.gethostname()  # the container id under docker compose

model_manager = ModelManager(MODELS_DIR, SEED_MODEL_DIR)
batcher = DynamicBatcher(
    model_manager.predict_batch, max_batch_size=BATCH_MAX_SIZE, max_wait_ms=BATCH_MAX_WAIT_MS
)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
cache = PredictionCache(redis_client, ttl_seconds=CACHE_TTL_SECONDS)

_db_pool: asyncpg.Pool | None = None
_db_last_attempt = 0.0


async def get_db_pool() -> asyncpg.Pool | None:
    """The pool is created lazily and retried at most every few seconds, so
    the API starts (and keeps serving) even while Postgres is down."""
    global _db_pool, _db_last_attempt
    if _db_pool is not None or not DATABASE_URL:
        return _db_pool
    if time.monotonic() - _db_last_attempt < DB_RETRY_SECONDS:
        return None
    _db_last_attempt = time.monotonic()
    try:
        _db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    except Exception:
        logger.warning("database unavailable, request logging paused", exc_info=True)
    return _db_pool


request_logger = RequestLogger(get_db_pool)


async def _watch_registry() -> None:
    """Fallback to POST /admin/reload: pick up a newly deployed model even if
    the retraining pipeline couldn't reach the API to tell it."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(MODEL_POLL_SECONDS)
        try:
            await loop.run_in_executor(None, model_manager.reload_if_changed)
        except Exception:
            logger.warning(
                "model reload failed, still serving %s", model_manager.version, exc_info=True
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_running_loop().run_in_executor(None, model_manager.reload_if_changed)
    batcher.start()
    request_logger.start()
    watcher = asyncio.create_task(_watch_registry())
    yield
    watcher.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watcher
    await batcher.stop()
    await request_logger.stop()
    await redis_client.aclose()
    if _db_pool is not None:
        await _db_pool.close()


app = FastAPI(title="Disaster Tweet Classifier (monitored)", lifespan=lifespan)


@app.middleware("http")
async def log_predict_requests(request: Request, call_next):
    """Log every /predict call, including the ones that never reach the
    handler (422 validation errors) or crash in it (500s), so the dashboard
    can compute an error rate. The handler fills request.state.log_fields
    with the prediction when there is one."""
    if request.url.path != "/predict":
        return await call_next(request)

    request_id = uuid4()
    request.state.request_id = request_id
    request.state.log_fields = {}
    ts = datetime.now(UTC)
    t0 = time.perf_counter()
    status_code, error = 500, None
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        request_logger.log(
            {
                "id": request_id,
                "ts": ts,
                "endpoint": request.url.path,
                "status_code": status_code,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "error": error,
                **request.state.log_fields,
            }
        )
    response.headers["X-Request-ID"] = str(request_id)
    response.headers["X-Served-By"] = REPLICA
    return response


async def _redis_connected() -> bool:
    try:
        return bool(await redis_client.ping())
    except Exception:
        return False


async def _database_connected() -> bool:
    pool = await get_db_pool()
    if pool is None:
        return False
    try:
        await pool.fetchval("SELECT 1")
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    redis_connected = await _redis_connected()
    database_connected = await _database_connected()
    if not model_manager.loaded:
        status = "error"
    elif redis_connected and database_connected:
        status = "ok"
    else:
        status = "degraded"
    return HealthResponse(
        status=status,
        model_loaded=model_manager.loaded,
        model_version=model_manager.version,
        redis_connected=redis_connected,
        database_connected=database_connected,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    model_version = model_manager.version
    result = None
    cached = False
    try:
        result = await cache.get(body.text, model_version)
        cached = result is not None
    except Exception:
        logger.warning("cache read failed, falling back to inference", exc_info=True)

    if result is None:
        result = await batcher.submit(body.text)
        try:
            await cache.set(body.text, result["model_version"], result)
        except Exception:
            logger.warning("cache write failed", exc_info=True)

    request.state.log_fields = {
        "text": result["text"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "cached": cached,
        "model_version": result["model_version"],
    }
    return PredictResponse(**result, cached=cached, request_id=request.state.request_id)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Record the ground-truth label for an earlier /predict call. In
    production this would come from human review; here the workload
    generator supplies it."""
    pool = await get_db_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    await pool.execute(
        """
        INSERT INTO feedback (request_id, true_label) VALUES ($1, $2)
        ON CONFLICT (request_id) DO UPDATE
            SET true_label = EXCLUDED.true_label, received_at = now()
        """,
        body.request_id,
        body.label,
    )
    return FeedbackResponse(request_id=body.request_id, label=body.label)


@app.post("/admin/reload", response_model=ReloadResponse)
async def reload_model(x_admin_token: str = Header(default="")) -> ReloadResponse:
    """Called by the retraining pipeline right after it deploys a new model,
    so the switch happens immediately instead of on the next registry poll."""
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")
    loop = asyncio.get_running_loop()
    changed = await loop.run_in_executor(None, model_manager.reload_if_changed)
    return ReloadResponse(changed=changed, model_version=model_manager.version)
