import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI

from app import model as model_module
from app.batcher import DynamicBatcher
from app.cache import PredictionCache
from app.model import load_model, predict_batch
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("disaster-tweet-api")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "16"))
BATCH_MAX_WAIT_MS = int(os.environ.get("BATCH_MAX_WAIT_MS", "20"))

batcher = DynamicBatcher(
    predict_batch, max_batch_size=BATCH_MAX_SIZE, max_wait_ms=BATCH_MAX_WAIT_MS
)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
cache = PredictionCache(redis_client, ttl_seconds=CACHE_TTL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("model loaded from %s", model_module.ONNX_DIR)
    batcher.start()
    yield
    await batcher.stop()
    await redis_client.aclose()


app = FastAPI(title="Disaster Tweet Classifier (optimized)", lifespan=lifespan)


async def _redis_connected() -> bool:
    try:
        return bool(await redis_client.ping())
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        load_model()
    except Exception:
        return HealthResponse(status="error", model_loaded=False, redis_connected=False)

    redis_connected = await _redis_connected()
    status = "ok" if redis_connected else "degraded"
    return HealthResponse(status=status, model_loaded=True, redis_connected=redis_connected)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    cached = None
    try:
        cached = await cache.get(request.text)
    except Exception:
        logger.warning("cache read failed, falling back to inference", exc_info=True)

    if cached is not None:
        logger.info("predict text=%r label=%s cached=True", request.text, cached["label"])
        return PredictResponse(**cached, cached=True)

    result = await batcher.submit(request.text)

    try:
        await cache.set(request.text, result)
    except Exception:
        logger.warning("cache write failed", exc_info=True)

    logger.info(
        "predict text=%r label=%s confidence=%.3f cached=False",
        result["text"],
        result["label"],
        result["confidence"],
    )
    return PredictResponse(**result, cached=False)
