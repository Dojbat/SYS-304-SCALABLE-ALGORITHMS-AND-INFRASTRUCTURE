"""Render the benchmark charts for REPORT.md from the saved result files.

Reads:
    milestone_3/results/loadtest_results.json            (concurrent HTTP load test)
    milestone_3/results/benchmark_summary_2026-09-23.json (in-process model benchmark)
Writes PNGs to milestone_3/report/figures/.

Run from the repo root:
    python milestone_3/report/make_charts.py
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIG_DIR = HERE / "figures"

NAIVE = "#8a8984"  # neutral grey: the baseline
OPTIMIZED = "#2a78d6"  # categorical slot 1
REDIS = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _median_and_range(runs: list[dict], key: str) -> tuple[float, float, float]:
    values = [r[key] for r in runs]
    return statistics.median(values), min(values), max(values)


def _grouped_bars(ax, groups, series, fmt="{:.1f}"):
    """series: list of (label, color, [(value, lo, hi) | (value,)] per group)."""
    n = len(series)
    width = 0.8 / n
    for i, (label, color, points) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        values = [p[0] for p in points]
        bars = ax.bar(xs, values, width * 0.92, color=color, label=label, zorder=2)
        if all(len(p) == 3 for p in points):
            lower = [p[0] - p[1] for p in points]
            upper = [p[2] - p[0] for p in points]
            ax.errorbar(
                xs,
                values,
                yerr=[lower, upper],
                fmt="none",
                ecolor=INK_2,
                elinewidth=1,
                capsize=3,
                zorder=3,
            )
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                fmt.format(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4 if len(points[0]) == 1 else 12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def concurrent_load(load: dict) -> None:
    runs = load["runs"]
    naive_rps = _median_and_range(runs["naive_unique"], "throughput_rps")
    unique_rps = _median_and_range(runs["optimized_unique"], "throughput_rps")
    cached_rps = _median_and_range(runs["optimized_cached"], "throughput_rps")
    naive_ms = _median_and_range(runs["naive_unique"], "mean_ms")
    unique_ms = _median_and_range(runs["optimized_unique"], "mean_ms")
    cached_ms = _median_and_range(runs["optimized_cached"], "mean_ms")
    naive_p95 = _median_and_range(runs["naive_unique"], "p95_ms")
    unique_p95 = _median_and_range(runs["optimized_unique"], "p95_ms")
    cached_p95 = _median_and_range(runs["optimized_cached"], "p95_ms")

    labels = ["Naive\n(Milestone 2)", "Optimized\nunique texts", "Optimized\ncache warm"]
    colors = [NAIVE, OPTIMIZED, OPTIMIZED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for ax, points, title, ylabel, fmt in [
        (
            ax1,
            [naive_rps, unique_rps, cached_rps],
            "Throughput (higher is better)",
            "requests / second",
            "{:.1f}",
        ),
        (
            ax2,
            [naive_ms, unique_ms, cached_ms],
            "Mean latency (lower is better)",
            "milliseconds",
            "{:,.0f}",
        ),
    ]:
        values = [p[0] for p in points]
        bars = ax.bar(range(3), values, 0.6, color=colors, zorder=2)
        ax.errorbar(
            range(3),
            values,
            yerr=[[p[0] - p[1] for p in points], [p[2] - p[0] for p in points]],
            fmt="none",
            ecolor=INK_2,
            elinewidth=1,
            capsize=3,
            zorder=3,
        )
        for bar, p in zip(bars, points, strict=True):
            ax.annotate(
                fmt.format(p[0]),
                (bar.get_x() + bar.get_width() / 2, p[2]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
        ax.set_xticks(range(3), labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(p[2] for p in points) * 1.15)

    fig.suptitle(
        "40 concurrent POST /predict requests — median of 3 runs, whiskers = min–max",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK_2,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "concurrent_load.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    points = [naive_p95, unique_p95, cached_p95]
    values = [p[0] for p in points]
    bars = ax.bar(range(3), values, 0.6, color=colors, zorder=2)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:,.0f}",
            (bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.set_xticks(range(3), labels)
    ax.set_title("p95 latency, 40 concurrent (lower is better)")
    ax.set_ylabel("milliseconds")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "concurrent_p95.png")
    plt.close(fig)


def memory(load: dict, bench: dict) -> None:
    mem = load["memory_under_load_mib"]
    naive_lo, naive_hi = mem["naive_backend"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

    ax1.barh([1], [naive_lo], 0.5, color=NAIVE, zorder=2)
    ax1.errorbar(
        [naive_lo],
        [1],
        xerr=[[0], [naive_hi - naive_lo]],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax1.barh([0], [mem["optimized_backend"]], 0.5, color=OPTIMIZED, label="backend", zorder=2)
    ax1.barh(
        [0],
        [mem["optimized_redis"]],
        0.5,
        left=[mem["optimized_backend"] + 4],
        color=REDIS,
        label="Redis",
        zorder=2,
    )
    ax1.annotate(
        f"{naive_lo:,.0f}–{naive_hi:,.0f} MiB",
        (naive_hi, 1),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )
    total = mem["optimized_backend"] + mem["optimized_redis"]
    ax1.annotate(
        f"{mem['optimized_backend']:,.0f} + {mem['optimized_redis']:,.0f} MiB",
        (total + 4, 0),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )
    ax1.set_yticks([0, 1], ["Optimized", "Naive"])
    ax1.set_xlim(0, naive_hi * 1.45)
    ax1.set_xlabel("MiB")
    ax1.set_title("Container memory under load")
    ax1.grid(axis="y", visible=False)
    ax1.legend(loc="lower right", fontsize=9)

    n, o = bench["naive"], bench["optimized"]
    _grouped_bars(
        ax2,
        ["Model memory\n(RSS increase)", "Peak RSS"],
        [
            ("Naive FP32", NAIVE, [(n["model_footprint_mb"],), (n["rss_peak_mb"],)]),
            ("Optimized INT8", OPTIMIZED, [(o["model_footprint_mb"],), (o["rss_peak_mb"],)]),
        ],
        fmt="{:,.0f}",
    )
    ax2.set_ylabel("MB")
    ax2.set_ylim(0, n["rss_peak_mb"] * 1.35)
    ax2.set_title("In-process model memory")
    ax2.legend(loc="upper right", fontsize=9, ncols=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "memory.png")
    plt.close(fig)


def model_latency(bench: dict) -> None:
    n, o = bench["naive"], bench["optimized"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    _grouped_bars(
        ax1,
        ["mean", "p50", "p95"],
        [
            (
                "Naive FP32 PyTorch+PEFT",
                NAIVE,
                [(n["latency_mean_ms"],), (n["latency_p50_ms"],), (n["latency_p95_ms"],)],
            ),
            (
                "Optimized ONNX INT8",
                OPTIMIZED,
                [(o["latency_mean_ms"],), (o["latency_p50_ms"],), (o["latency_p95_ms"],)],
            ),
        ],
    )
    ax1.set_ylabel("milliseconds")
    ax1.set_ylim(0, n["latency_p95_ms"] * 1.2)
    ax1.set_title("Single-request latency (lower is better)")
    ax1.legend(loc="upper left", fontsize=9)

    bars = ax2.bar(
        [0, 1],
        [n["throughput_req_per_s"], o["throughput_req_per_s"]],
        0.6,
        color=[NAIVE, OPTIMIZED],
        zorder=2,
    )
    for bar in bars:
        ax2.annotate(
            f"{bar.get_height():.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax2.set_xticks([0, 1], ["Naive FP32", "Optimized INT8"])
    ax2.set_ylabel("requests / second")
    ax2.set_ylim(0, o["throughput_req_per_s"] * 1.2)
    ax2.set_title("Sequential throughput")
    ax2.grid(axis="x", visible=False)

    fig.suptitle(
        "In-process benchmark, 200 tweets sent one at a time (benchmark.py)",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK_2,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_latency.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    load = json.loads((RESULTS / "loadtest_results.json").read_text())
    bench = json.loads((RESULTS / "benchmark_summary_2026-09-23.json").read_text())
    concurrent_load(load)
    memory(load, bench)
    model_latency(bench)
    print(f"wrote charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
