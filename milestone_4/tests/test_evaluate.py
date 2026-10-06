import numpy as np
from evaluate import DeployPolicy, Metrics, int8_ok, score, should_deploy

POLICY = DeployPolicy(min_f1_gain=0.01, max_guard_drop=0.02, min_eval_examples=40)


def test_score_computes_f1_and_accuracy():
    metrics = score(np.array([0.9, 0.8, 0.2, 0.6]), [1, 0, 0, 1])
    # predictions 1,1,0,1 vs labels 1,0,0,1: tp=2, fp=1, fn=0
    assert metrics.accuracy == 0.75
    assert metrics.f1 == 0.8
    assert metrics.n == 4


def test_better_candidate_is_deployed():
    ok, _ = should_deploy(
        Metrics(0.70, 0.74, 100),
        Metrics(0.95, 0.96, 100),
        Metrics(0.85, 0.856, 500),
        Metrics(0.84, 0.845, 500),
        POLICY,
    )
    assert ok


def test_candidate_without_enough_gain_is_rejected():
    ok, reason = should_deploy(
        Metrics(0.90, 0.9, 100),
        Metrics(0.905, 0.9, 100),
        Metrics(0.85, 0.85, 500),
        Metrics(0.85, 0.85, 500),
        POLICY,
    )
    assert not ok
    assert "gain" in reason


def test_candidate_that_forgets_the_original_task_is_rejected():
    ok, reason = should_deploy(
        Metrics(0.70, 0.74, 100),
        Metrics(0.95, 0.96, 100),
        Metrics(0.85, 0.86, 500),
        Metrics(0.80, 0.80, 500),
        POLICY,
    )
    assert not ok
    assert "guard" in reason


def test_too_small_eval_set_is_not_trusted():
    ok, _ = should_deploy(
        Metrics(0.5, 0.5, 10),
        Metrics(1.0, 1.0, 10),
        Metrics(0.85, 0.85, 500),
        Metrics(0.85, 0.85, 500),
        POLICY,
    )
    assert not ok


def test_broken_int8_export_is_caught():
    assert int8_ok(Metrics(0.95, 0.95, 100), Metrics(0.94, 0.94, 100), POLICY)[0]
    assert not int8_ok(Metrics(0.95, 0.95, 100), Metrics(0.40, 0.60, 100), POLICY)[0]
