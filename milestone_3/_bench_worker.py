"""Runs one model variant's benchmark in its own process and writes JSON results.

Run in a subprocess (by benchmark.py) rather than in-process alongside the
other variant so each variant's memory measurement reflects only that
variant's footprint, not both models loaded together.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rss_mb() -> float:
    gc.collect()
    return psutil.Process().memory_info().rss / 1e6


def load_predict_fn(variant: str):
    if variant == "naive":
        sys.path.insert(0, str(REPO_ROOT / "milestone_2" / "backend"))
        import os

        os.environ.setdefault(
            "MODEL_PATH", str(REPO_ROOT / "training" / "bertweet_lora_final_fulldata")
        )
        from app.model import load_model, predict_one

        return load_model, predict_one
    elif variant == "optimized":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from optimized_model import load_model, predict_one

        return load_model, predict_one
    raise ValueError(f"unknown variant {variant!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["naive", "optimized"])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(REPO_ROOT / "train.csv").sample(n=args.num_samples, random_state=args.seed)
    texts = df["text"].tolist()
    labels = df["target"].tolist()

    baseline_rss = _rss_mb()

    load_model, predict_one = load_predict_fn(args.variant)

    load_start = time.perf_counter()
    load_model()
    load_time_s = time.perf_counter() - load_start
    post_load_rss = _rss_mb()

    # warm-up (excluded from timing — first calls pay for lazy init / caches)
    for text in texts[:5]:
        predict_one(text)

    latencies_s = []
    predictions = []
    infer_start = time.perf_counter()
    for text in texts:
        t0 = time.perf_counter()
        result = predict_one(text)
        latencies_s.append(time.perf_counter() - t0)
        predictions.append(result["prediction"])
    total_infer_s = time.perf_counter() - infer_start
    peak_rss = _rss_mb()

    latencies_ms = sorted(x * 1000 for x in latencies_s)
    n = len(latencies_ms)
    correct = sum(p == y for p, y in zip(predictions, labels, strict=True))

    results = {
        "variant": args.variant,
        "num_samples": n,
        "load_time_s": load_time_s,
        "rss_before_load_mb": baseline_rss,
        "rss_after_load_mb": post_load_rss,
        "model_footprint_mb": post_load_rss - baseline_rss,
        "rss_peak_mb": peak_rss,
        "throughput_req_per_s": n / total_infer_s,
        "latency_mean_ms": sum(latencies_ms) / n,
        "latency_p50_ms": latencies_ms[n // 2],
        "latency_p95_ms": latencies_ms[int(n * 0.95)],
        "latency_max_ms": latencies_ms[-1],
        "accuracy_vs_label": correct / n,
        "predictions": predictions,
        "texts": texts,
        "labels": labels,
    }

    Path(args.out).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
