from datetime import UTC, datetime, timedelta

import data
from data import DataBudget, Example

BUDGET = DataBudget(
    max_train=1000,
    max_recent=400,
    max_older_live=200,
    max_original=300,
    min_original=50,
    max_eval=100,
    max_class_share=0.7,
)


def _original(n: int = 2000) -> list[Example]:
    return [Example(f"original tweet {i}", i % 2) for i in range(n)]


def _live(prefix: str, n: int, label=lambda i: i % 2) -> list[Example]:
    return [Example(f"{prefix} {i}", label(i)) for i in range(n)]


def test_eval_split_is_deterministic_and_roughly_one_in_five():
    texts = [f"live tweet {i}" for i in range(2000)]
    first = [data.is_eval(t) for t in texts]
    assert first == [data.is_eval(t) for t in texts]
    assert 0.15 < sum(first) / len(first) < 0.25


def test_window_starts_at_the_models_cutoff_but_never_too_far_back():
    now = datetime(2026, 10, 6, tzinfo=UTC)
    budget = DataBudget(recent_days=7)
    yesterday = now - timedelta(days=1)
    assert data.window_start(now, yesterday, budget) == yesterday
    assert data.window_start(now, now - timedelta(days=30), budget) == now - timedelta(days=7)
    assert data.window_start(now, None, budget) == now - timedelta(days=7)


def test_every_source_is_capped_by_the_budget():
    cycle = data.build(
        recent_train=_live("recent", 5000),
        recent_eval=_live("eval", 5000),
        older_train=_live("older", 5000),
        original=_original(),
        budget=BUDGET,
    )
    assert cycle.stats["recent_used"] <= BUDGET.max_recent
    assert cycle.stats["older_used"] <= BUDGET.max_older_live
    assert len(cycle.replay) <= BUDGET.max_original
    assert len(cycle.live_eval) == BUDGET.max_eval
    assert len(cycle.train) <= BUDGET.max_train


def test_total_cap_trims_live_data_but_keeps_the_replay_floor():
    budget = DataBudget(
        max_train=500, max_recent=400, max_older_live=200, max_original=300, min_original=50
    )
    cycle = data.build(_live("recent", 400), [], _live("older", 200), _original(), budget)
    assert len(cycle.train) == 500
    assert cycle.stats["original_used"] == 50  # floor protects against forgetting
    assert cycle.stats["recent_used"] == 400  # highest priority, kept in full
    assert cycle.stats["older_used"] == 50  # trimmed to fit


def test_held_out_data_never_reaches_training():
    original = _original()
    guard_text = data.build([], [], [], original, BUDGET).guard[0].text
    eval_examples = _live("shared", 20)
    cycle = data.build(
        recent_train=eval_examples + [Example(guard_text, 1)] + _live("recent", 100),
        recent_eval=eval_examples,
        older_train=eval_examples,
        original=original,
        budget=BUDGET,
    )
    train_texts = {e.text for e in cycle.train}
    assert not train_texts & {e.text for e in cycle.live_eval}
    assert not train_texts & {e.text for e in cycle.guard}
    assert len(cycle.guard) == data.GUARD_SIZE


def test_duplicate_texts_are_used_once():
    duplicated = [Example("same tweet", 1)] * 50
    cycle = data.build(duplicated, [], duplicated, _original(), BUDGET)
    assert sum(e.text == "same tweet" for e in cycle.train) == 1


def test_replay_matches_live_size_within_limits():
    small = data.build(_live("recent", 10), [], [], _original(), BUDGET)
    assert len(small.replay) == BUDGET.min_original
    large = data.build(_live("recent", 400), [], _live("older", 200), _original(), BUDGET)
    assert len(large.replay) == BUDGET.max_original


def test_class_balance_trims_the_majority_from_the_lowest_priority_end():
    examples = _live("pos", 90, label=lambda i: 1) + _live("neg", 10, label=lambda i: 0)
    balanced = data.balance(examples, max_share=0.7)
    positives = sum(e.label for e in balanced)
    assert positives / len(balanced) <= 0.7 + 1e-9
    assert sum(1 for e in balanced if e.label == 0) == 10  # minority untouched
    assert balanced[0] == examples[0]  # highest-priority examples kept


def test_balance_leaves_a_balanced_or_single_class_set_alone():
    mixed = _live("x", 10)
    assert data.balance(mixed, 0.7) == mixed
    only_positive = _live("p", 5, label=lambda i: 1)
    assert data.balance(only_positive, 0.7) == only_positive
