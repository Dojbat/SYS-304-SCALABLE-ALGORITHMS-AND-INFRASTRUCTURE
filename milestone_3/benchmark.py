"""Week 5 benchmarking script — naive (PyTorch+PEFT, FP32) vs optimized
(LoRA merged, ONNX, INT8 dynamic quantization).

Each variant is benchmarked in its own subprocess (see _bench_worker.py) so
memory measurements aren't contaminated by the other variant's weights also
sitting in the process. Sends the same batch of tweets to both, one request
at a time (matching how /predict is actually called), and reports:

- load time and resident memory footprint after loading
- per-request latency (mean / p50 / p95 / max) and throughput
- accuracy against train.csv labels, and prediction agreement between the
  two variants (train.csv was used to fine-tune the served model, so
  accuracy here is a memorization check, not a held-out estimate — the
  agreement rate is the more meaningful number for judging whether
  quantization changed behavior)
- on-disk size of the served model artifacts

Run from the repo root:
    python milestone_3/benchmark.py [--num-samples N]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

NAIVE_MODEL_FILES = [
    REPO_ROOT / "training" / "bertweet_lora_final_fulldata" / "adapter_model.safetensors",
]
OPTIMIZED_MODEL_FILES = [
    Path(__file__).resolve().parent / "onnx_model" / "model_int8.onnx",
]


def run_worker(variant: str, num_samples: int, seed: int) -> dict:
    out_path = RESULTS_DIR / f"_{variant}.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "_bench_worker.py"),
            variant,
            "--num-samples",
            str(num_samples),
            "--seed",
            str(seed),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return json.loads(out_path.read_text())


def print_table(naive: dict, optimized: dict) -> None:
    rows = [
        ("load time (s)", naive["load_time_s"], optimized["load_time_s"]),
        (
            "model footprint (MB, RSS delta)",
            naive["model_footprint_mb"],
            optimized["model_footprint_mb"],
        ),
        ("peak RSS (MB)", naive["rss_peak_mb"], optimized["rss_peak_mb"]),
        ("mean latency (ms)", naive["latency_mean_ms"], optimized["latency_mean_ms"]),
        ("p50 latency (ms)", naive["latency_p50_ms"], optimized["latency_p50_ms"]),
        ("p95 latency (ms)", naive["latency_p95_ms"], optimized["latency_p95_ms"]),
        ("throughput (req/s)", naive["throughput_req_per_s"], optimized["throughput_req_per_s"]),
        ("accuracy vs label", naive["accuracy_vs_label"], optimized["accuracy_vs_label"]),
    ]
    header = (
        f"{'metric':<34}{'naive (FP32 PyTorch+PEFT)':>28}"
        f"{'optimized (ONNX INT8)':>25}{'speedup/ratio':>16}"
    )
    print(header)
    print("-" * len(header))
    for name, n_val, o_val in rows:
        if "latency" in name:
            ratio = n_val / o_val if o_val else float("nan")
            ratio_str = f"{ratio:.2f}x faster"
        elif "throughput" in name:
            ratio = o_val / n_val if n_val else float("nan")
            ratio_str = f"{ratio:.2f}x"
        elif "footprint" in name or "RSS" in name:
            ratio = n_val / o_val if o_val else float("nan")
            ratio_str = f"{ratio:.2f}x smaller"
        else:
            ratio_str = ""
        print(f"{name:<34}{n_val:>28.3f}{o_val:>25.3f}{ratio_str:>16}")

    agree = sum(
        p == q for p, q in zip(naive["predictions"], optimized["predictions"], strict=True)
    ) / len(naive["predictions"])
    print(f"\nprediction agreement (naive vs optimized): {agree:.1%}")

    naive_disk = sum(f.stat().st_size for f in NAIVE_MODEL_FILES if f.exists()) / 1e6
    optimized_disk = sum(f.stat().st_size for f in OPTIMIZED_MODEL_FILES if f.exists()) / 1e6
    print(
        f"served-artifact disk size: naive LoRA adapter {naive_disk:.1f} MB "
        f"(+ shared vinai/bertweet-base FP32 checkpoint, ~540 MB) | "
        f"optimized merged+INT8 ONNX {optimized_disk:.1f} MB (self-contained)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"benchmarking naive model on {args.num_samples} requests...")
    naive = run_worker("naive", args.num_samples, args.seed)
    print(f"benchmarking optimized model on {args.num_samples} requests...")
    optimized = run_worker("optimized", args.num_samples, args.seed)

    print()
    print_table(naive, optimized)

    summary_path = RESULTS_DIR / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "naive": {
                    k: v for k, v in naive.items() if k not in ("predictions", "texts", "labels")
                },
                "optimized": {
                    k: v
                    for k, v in optimized.items()
                    if k not in ("predictions", "texts", "labels")
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
