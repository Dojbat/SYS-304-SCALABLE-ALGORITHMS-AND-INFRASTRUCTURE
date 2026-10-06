from trigger import SystemState, TriggerConfig, decide

CONFIG = TriggerConfig(
    max_age_s=3600,
    confidence_threshold=0.85,
    accuracy_threshold=0.80,
    min_new_labels=100,
    cooldown_s=60,
)


def _state(**overrides) -> SystemState:
    values = {
        "since_last_activity_s": 10,
        "since_last_run_s": None,
        "new_labels": 500,
        "recent_confidence": 0.92,
        "recent_accuracy": 0.86,
    }
    values.update(overrides)
    return SystemState(**values)


def test_healthy_system_does_not_trigger():
    assert decide(_state(), CONFIG)[0] is None


def test_confidence_drop_triggers():
    assert decide(_state(recent_confidence=0.80), CONFIG)[0] == "confidence_drop"


def test_accuracy_drop_triggers_even_when_confidence_looks_fine():
    # the model is confidently wrong: the case confidence alone misses
    assert decide(_state(recent_accuracy=0.70), CONFIG)[0] == "accuracy_drop"


def test_schedule_triggers_after_max_age():
    assert decide(_state(since_last_activity_s=4000), CONFIG)[0] == "schedule"


def test_nothing_triggers_without_enough_new_labels():
    trigger, reason = decide(_state(new_labels=50, recent_accuracy=0.5), CONFIG)
    assert trigger is None
    assert "waiting for labels" in reason


def test_cooldown_blocks_a_retrigger_right_after_a_run():
    trigger, reason = decide(_state(since_last_run_s=30, recent_accuracy=0.5), CONFIG)
    assert trigger is None
    assert "cooldown" in reason


def test_windows_that_are_not_full_yet_are_ignored():
    state = _state(recent_confidence=None, recent_accuracy=None)
    assert decide(state, CONFIG)[0] is None
