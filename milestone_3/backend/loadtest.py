"""System-level load test — fires a burst of concurrent /predict requests at
a running backend and reports throughput, latency, and cache-hit rate.

This is what actually exercises dynamic batching and Redis caching: the
Week 5 benchmark (milestone_3/benchmark.py) calls the model in-process, one
request at a time, so it never has more than one request in flight. This
script drives the real HTTP API concurrently, against either backend
(the naive Milestone 2 one or the optimized Milestone 3 one), so the same
script produces the naive-vs-optimized comparison this milestone's report
is built on.

Run against a backend exposed on localhost (docker compose or a bare
`docker run`), e.g.:

    python milestone_3/backend/loadtest.py --url http://localhost:8000 --concurrency 40
    python milestone_3/backend/loadtest.py --url http://localhost:8000 \
        --concurrency 40 --unique 5 --warm-cache
"""

import argparse
import asyncio
import time

import httpx


async def _one(client: httpx.AsyncClient, url: str, text: str) -> tuple[float, dict]:
    t0 = time.perf_counter()
    response = await client.post(f"{url}/predict", json={"text": text})
    response.raise_for_status()
    return time.perf_counter() - t0, response.json()


async def run(url: str, concurrency: int, unique: int, warm_cache: bool) -> None:
    unique_texts = [f"load test disaster tweet variant {i} building on fire" for i in range(unique)]
    texts = [unique_texts[i % unique] for i in range(concurrency)]

    async with httpx.AsyncClient(timeout=60) as client:
        if warm_cache:
            for text in unique_texts:
                await _one(client, url, text)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(_one(client, url, text) for text in texts))
        total_s = time.perf_counter() - t0

    latencies_ms = sorted(r[0] * 1000 for r in results)
    cache_hits = sum(1 for _, body in results if body.get("cached"))
    n = len(results)

    print(f"url:              {url}")
    print(f"concurrency:      {concurrency} requests ({unique} unique texts)")
    print(f"warm_cache:       {warm_cache}")
    print(f"total wall time:  {total_s * 1000:.1f} ms")
    print(f"throughput:       {n / total_s:.1f} req/s")
    print(f"mean latency:     {sum(latencies_ms) / n:.1f} ms")
    print(f"p50 latency:      {latencies_ms[n // 2]:.1f} ms")
    print(f"p95 latency:      {latencies_ms[int(n * 0.95)]:.1f} ms")
    print(f"cache hits:       {cache_hits}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument(
        "--unique",
        type=int,
        default=None,
        help="number of distinct texts to cycle through (default: all unique, i.e. == concurrency)",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="send each unique text once before the burst, so the burst is all cache hits",
    )
    args = parser.parse_args()
    unique = args.unique if args.unique is not None else args.concurrency
    asyncio.run(run(args.url, args.concurrency, unique, args.warm_cache))


if __name__ == "__main__":
    main()
