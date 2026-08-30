import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import model as model_module
from app.model import load_model, predict_one
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("disaster-tweet-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("model loaded from %s", model_module.MODEL_PATH)
    yield


app = FastAPI(title="Disaster Tweet Classifier", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True)
    except Exception:
        return HealthResponse(status="error", model_loaded=False)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    result = predict_one(request.text)
    logger.info(
        "predict text=%r label=%s confidence=%.3f",
        result["text"],
        result["label"],
        result["confidence"],
    )
    return PredictResponse(**result)
