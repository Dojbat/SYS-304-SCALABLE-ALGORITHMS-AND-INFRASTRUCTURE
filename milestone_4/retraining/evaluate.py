"""Metrics and the deploy / don't-deploy decision."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    f1: float
    accuracy: float
    n: int


@dataclass(frozen=True)
class DeployPolicy:
    # the candidate must beat the current model on held-out live data by this much
    min_f1_gain: float = 0.01
    # and may lose at most this much accuracy on the original-data guard set
    max_guard_drop: float = 0.02
    # the INT8 export may lose at most this much F1 vs. the PyTorch candidate
    max_int8_f1_drop: float = 0.03
    # below this many held-out live examples the comparison isn't trusted
    min_eval_examples: int = 40


def score(probs: np.ndarray, labels: list[int]) -> Metrics:
    """F1 on the "disaster" class, plus accuracy."""
    y = np.asarray(labels)
    pred = (probs >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    denom = int(pred.sum() + y.sum())
    f1 = 2 * tp / denom if denom else 0.0
    accuracy = float((pred == y).mean()) if len(y) else 0.0
    return Metrics(f1=f1, accuracy=accuracy, n=len(y))


def should_deploy(
    base_live: Metrics,
    candidate_live: Metrics,
    base_guard: Metrics,
    candidate_guard: Metrics,
    policy: DeployPolicy,
) -> tuple[bool, str]:
    if candidate_live.n < policy.min_eval_examples:
        return False, f"only {candidate_live.n} held-out live examples"
    gain = candidate_live.f1 - base_live.f1
    if gain < policy.min_f1_gain:
        return False, f"live F1 gain {gain:+.3f} < {policy.min_f1_gain}"
    guard_drop = base_guard.accuracy - candidate_guard.accuracy
    if guard_drop > policy.max_guard_drop:
        return False, f"guard accuracy dropped {guard_drop:.3f} > {policy.max_guard_drop}"
    return True, f"live F1 {base_live.f1:.3f} -> {candidate_live.f1:.3f}"


def int8_ok(candidate_live: Metrics, int8_live: Metrics, policy: DeployPolicy) -> tuple[bool, str]:
    """Guards against the per-tensor quantization collapse found in
    Milestone 3: an export that quantizes badly is caught here, before it
    reaches production."""
    drop = candidate_live.f1 - int8_live.f1
    if drop > policy.max_int8_f1_drop:
        return False, f"INT8 export lost {drop:.3f} F1 vs. PyTorch"
    return True, f"INT8 F1 {int8_live.f1:.3f}"
