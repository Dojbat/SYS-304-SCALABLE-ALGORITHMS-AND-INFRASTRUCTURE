"""Workload generator: sends a stream of /predict requests through the
system, and later sends ground-truth labels for some of them to /feedback.

Traffic mix (fractions of requests, set by flags):
- normal: real tweets. Labelled ones from train.csv, unlabelled ones from
  test.csv, so not every request ever gets a label.
- drift:  synthetic tweets from drift_texts.py (newer slang, emoji, and
  figurative use of disaster words), all with known labels.
- errors: invalid requests (empty text, missing field, text over the 1000
  character limit, malformed JSON) that the API should reject with 422.

Labels are sent to /feedback for a share of the labelled requests, after a
random delay, to mimic labels arriving from human review some time later.

Requests are sent open-loop at a fixed rate (they don't wait for earlier
responses), with a cap on how many can be in flight at once.

Examples (from the repo root, with the stack running):
    python milestone_4/workload/generate.py --duration 120 --rate 20
    python milestone_4/workload/generate.py --duration 300 --rate 20 --drift-share 0.8
    python milestone_4/workload/generate.py --duration 60 --rate 10 --error-share 0.2
"""

import argparse
import asyncio
import csv
import random
import time
from collections import Counter
from pathlib import Path

import httpx
from drift_texts import generate as generate_drift

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_tweets(path: Path, labelled: bool) -> list[tuple[str, int | None]]:
    with open(path, encoding="utf-8", newline="") as f:
        return [
            (row["text"], int(row["target"]) if labelled else None) for row in csv.DictReader(f)
        ]


class Workload:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.train = load_tweets(REPO_ROOT / "train.csv", labelled=True)
        self.test = load_tweets(REPO_ROOT / "test.csv", labelled=False)
        self.stats: Counter[str] = Counter()
        self.latencies_ms: list[float] = []
        self.confidences: list[float] = []
        self.feedback_tasks: list[asyncio.Task] = []

    def next_request(self) -> tuple[str, dict | bytes, int | None]:
        """Returns (kind, payload, true_label)."""
        roll = self.rng.random()
        if roll < self.args.error_share:
            return "error", self._invalid_payload(), None
        if roll < self.args.error_share + self.args.drift_share:
            text, label = generate_drift(self.rng)
            return "drift", {"text": text}, label
        pool = self.train if self.rng.random() < self.args.labelled_share else self.test
        text, label = self.rng.choice(pool)
        return "normal", {"text": text}, label

    def _invalid_payload(self) -> dict | bytes:
        return self.rng.choice(
            [
                {"text": ""},
                {"tweet": "field has the wrong name"},
                {"text": "x" * 1200},
                b"{not json",
            ]
        )

    async def send_one(self, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> None:
        kind, payload, label = self.next_request()
        async with sem:
            t0 = time.perf_counter()
            try:
                if isinstance(payload, bytes):
                    response = await client.post(
                        "/predict", content=payload, headers={"Content-Type": "application/json"}
                    )
                else:
                    response = await client.post("/predict", json=payload)
            except httpx.HTTPError:
                self.stats["transport_error"] += 1
                return
            self.latencies_ms.append((time.perf_counter() - t0) * 1000)

        self.stats[f"{kind}_{response.status_code}"] += 1
        if response.status_code != 200:
            return
        body = response.json()
        self.confidences.append(body["confidence"])
        if body.get("cached"):
            self.stats["cached"] += 1
        if label is not None and self.rng.random() < self.args.feedback_share:
            delay = self.rng.uniform(0, self.args.max_feedback_delay)
            self.feedback_tasks.append(
                asyncio.create_task(self.send_feedback(client, body["request_id"], label, delay))
            )

    async def send_feedback(
        self, client: httpx.AsyncClient, request_id: str, label: int, delay: float
    ) -> None:
        await asyncio.sleep(delay)
        try:
            response = await client.post(
                "/feedback", json={"request_id": request_id, "label": label}
            )
            self.stats[f"feedback_{response.status_code}"] += 1
        except httpx.HTTPError:
            self.stats["feedback_transport_error"] += 1

    async def run(self) -> None:
        args = self.args
        sem = asyncio.Semaphore(args.concurrency)
        interval = 1 / args.rate
        total = args.count if args.count else int(args.duration * args.rate)
        async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
            tasks = []
            start = time.perf_counter()
            for i in range(total):
                # schedule against the start time so slow iterations don't
                # accumulate drift in the send rate
                delay = start + i * interval - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(self.send_one(client, sem)))
                if args.progress_every and (i + 1) % args.progress_every == 0:
                    self.print_progress(i + 1, total, time.perf_counter() - start)
            await asyncio.gather(*tasks)
            await asyncio.gather(*self.feedback_tasks)
            self.print_summary(time.perf_counter() - start)

    def print_progress(self, sent: int, total: int, elapsed: float) -> None:
        recent = self.confidences[-200:]
        mean_conf = sum(recent) / len(recent) if recent else float("nan")
        print(f"[{elapsed:6.1f}s] sent {sent}/{total}  recent mean confidence {mean_conf:.3f}")

    def print_summary(self, elapsed: float) -> None:
        latencies = sorted(self.latencies_ms)
        n = len(latencies)
        print("\n--- workload summary ---")
        print(f"wall time:        {elapsed:.1f} s")
        print(f"responses:        {n} ({n / elapsed:.1f} req/s)")
        if n:
            print(f"latency p50/p95:  {latencies[n // 2]:.1f} / {latencies[int(n * 0.95)]:.1f} ms")
        if self.confidences:
            print(f"mean confidence:  {sum(self.confidences) / len(self.confidences):.3f}")
        for key, value in sorted(self.stats.items()):
            print(f"{key + ':':<18}{value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rate", type=float, default=10, help="requests per second")
    parser.add_argument("--duration", type=float, default=60, help="seconds (ignored with --count)")
    parser.add_argument("--count", type=int, default=0, help="total requests to send")
    parser.add_argument("--concurrency", type=int, default=32, help="max requests in flight")
    parser.add_argument("--drift-share", type=float, default=0.0)
    parser.add_argument("--error-share", type=float, default=0.02)
    parser.add_argument(
        "--labelled-share",
        type=float,
        default=0.7,
        help="share of normal traffic drawn from train.csv (labelled) vs test.csv",
    )
    parser.add_argument(
        "--feedback-share",
        type=float,
        default=0.8,
        help="share of labelled requests whose label is sent to /feedback",
    )
    parser.add_argument("--max-feedback-delay", type=float, default=5.0, help="seconds")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.drift_share + args.error_share > 1:
        parser.error("--drift-share + --error-share must be <= 1")
    asyncio.run(Workload(args).run())


if __name__ == "__main__":
    main()
