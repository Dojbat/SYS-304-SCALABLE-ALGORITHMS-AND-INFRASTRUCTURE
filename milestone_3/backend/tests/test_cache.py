import fakeredis
from app.cache import PredictionCache, cache_key


def test_cache_key_is_deterministic_and_distinguishes_text() -> None:
    assert cache_key("hello") == cache_key("hello")
    assert cache_key("hello") != cache_key("world")


def test_cache_key_ignores_surrounding_whitespace() -> None:
    assert cache_key("hello") == cache_key("  hello  ")


async def test_miss_then_set_then_hit() -> None:
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache = PredictionCache(redis_client, ttl_seconds=60)

    assert await cache.get("wildfire spreading") is None

    result = {
        "text": "wildfire spreading",
        "prediction": 1,
        "label": "disaster",
        "confidence": 0.98,
    }
    await cache.set("wildfire spreading", result)

    assert await cache.get("wildfire spreading") == result


async def test_different_text_is_a_separate_cache_entry() -> None:
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache = PredictionCache(redis_client, ttl_seconds=60)

    await cache.set("a", {"text": "a", "prediction": 0, "label": "not disaster", "confidence": 0.6})

    assert await cache.get("b") is None
