from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    request_id: UUID
    text: str
    prediction: int
    label: str
    confidence: float
    cached: bool
    model_version: str


class FeedbackRequest(BaseModel):
    request_id: UUID
    label: Literal[0, 1]


class FeedbackResponse(BaseModel):
    request_id: UUID
    label: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    redis_connected: bool
    database_connected: bool


class ReloadResponse(BaseModel):
    changed: bool
    model_version: str | None
