"""When to retrain.

Three triggers, checked in order:

- confidence_drop: mean confidence of the current model's recent
  predictions falls below a threshold.
- accuracy_drop: accuracy on recent predictions whose ground-truth label
  has come back through /feedback falls below a threshold. Confidence alone
  misses the most dangerous kind of drift, where the model is *confidently*
  wrong (it labels Spanish earthquake reports "not disaster" with ~0.9
  confidence), so labelled accuracy is checked too.
- schedule: the model (or the last retraining attempt) is older than a
  maximum age.

None of them fire until enough new labelled data has arrived to learn from,
or within a cooldown after the previous attempt, so a rejected candidate
doesn't cause a retraining loop.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerConfig:
    max_age_s: float = 24 * 3600
    confidence_threshold: float = 0.85
    confidence_window: int = 300
    accuracy_threshold: float = 0.78
    accuracy_window: int = 400
    min_new_labels: int = 200
    cooldown_s: float = 300


@dataclass(frozen=True)
class SystemState:
    # seconds since the current model was deployed or retraining last ran,
    # whichever is more recent
    since_last_activity_s: float
    # seconds since retraining last ran, None if it never has
    since_last_run_s: float | None
    new_labels: int
    # None until the window is full
    recent_confidence: float | None
    recent_accuracy: float | None


def decide(state: SystemState, config: TriggerConfig) -> tuple[str | None, str]:
    """Returns (trigger name or None, human-readable reason)."""
    if state.since_last_run_s is not None and state.since_last_run_s < config.cooldown_s:
        return None, f"cooldown ({state.since_last_run_s:.0f}s / {config.cooldown_s:.0f}s)"
    if state.new_labels < config.min_new_labels:
        return None, f"waiting for labels ({state.new_labels} / {config.min_new_labels} new)"
    if (
        state.recent_confidence is not None
        and state.recent_confidence < config.confidence_threshold
    ):
        return "confidence_drop", (
            f"mean confidence {state.recent_confidence:.3f} < {config.confidence_threshold}"
        )
    if state.recent_accuracy is not None and state.recent_accuracy < config.accuracy_threshold:
        return "accuracy_drop", (
            f"labelled accuracy {state.recent_accuracy:.3f} < {config.accuracy_threshold}"
        )
    if state.since_last_activity_s >= config.max_age_s:
        return (
            "schedule",
            f"model age {state.since_last_activity_s:.0f}s >= {config.max_age_s:.0f}s",
        )
    return None, "healthy"
