"""Exact-match prediction cache backed by Redis.

Disaster-tweet traffic is bursty and repetitive by nature — a viral tweet
gets retweeted/quoted verbatim, and the same handful of keyword-bearing
phrases ("earthquake", "wildfire evacuation", ...) recur across a feed. An
exact-match cache on the request text avoids re-running the model for
duplicate requests, which is common enough in this domain to be worth the
Redis round trip.
"""

import hashlib
import json
from typing import Any

CACHE_KEY_PREFIX = "predict:"


def cache_key(text: str) -> str:
    return CACHE_KEY_PREFIX + hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class PredictionCache:
    def __init__(self, redis_client: Any, ttl_seconds: int) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    async def get(self, text: str) -> dict | None:
        raw = await self._redis.get(cache_key(text))
        return json.loads(raw) if raw is not None else None

    async def set(self, text: str, result: dict) -> None:
        await self._redis.set(cache_key(text), json.dumps(result), ex=self._ttl_seconds)
