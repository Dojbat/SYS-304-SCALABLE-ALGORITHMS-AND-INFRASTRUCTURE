"""Exact-match prediction cache backed by Redis.

Same idea as milestone_3/backend/app/cache.py, with one change: the model
version is part of the key. When retraining deploys a new model, entries
cached by the old one simply stop matching (and expire on their TTL)
instead of returning the old model's answers.
"""

import hashlib
import json
from typing import Any

CACHE_KEY_PREFIX = "predict:"


def cache_key(text: str, model_version: str) -> str:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}{model_version}:{digest}"


class PredictionCache:
    def __init__(self, redis_client: Any, ttl_seconds: int) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    async def get(self, text: str, model_version: str) -> dict | None:
        raw = await self._redis.get(cache_key(text, model_version))
        return json.loads(raw) if raw is not None else None

    async def set(self, text: str, model_version: str, result: dict) -> None:
        await self._redis.set(
            cache_key(text, model_version), json.dumps(result), ex=self._ttl_seconds
        )
