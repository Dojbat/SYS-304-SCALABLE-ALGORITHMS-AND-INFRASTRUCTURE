from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    text: str
    prediction: int
    label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
