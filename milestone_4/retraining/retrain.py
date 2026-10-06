"""Automated retraining: watch the live system and retrain when a trigger
fires, or run one cycle by hand.

Runs on the host (not in docker) so fine-tuning can use the GPU. Needs the
stack up (`docker compose up`) for Postgres, the API and the shared models
folder.

    python milestone_4/retraining/retrain.py status
    python milestone_4/retraining/retrain.py watch --poll 30
    python milestone_4/retraining/retrain.py run --trigger manual
    python milestone_4/retraining/retrain.py maintenance --dry-run   # retention report

Demo-friendly thresholds (retrain soon after drifted traffic starts):
    python milestone_4/retraining/retrain.py watch --poll 15 --min-new-labels 300 \
        --cooldown 120
"""

import argparse
import logging
import os
import time
from datetime import UTC, datetime

import psycopg
from data import DataBudget, count_new_labels
from pipeline import PipelineConfig, previous_cutoff, read_registry, run_once
from trigger import SystemState, TriggerConfig, decide

logger = logging.getLogger("retraining")

MAINTENANCE_EVERY_S = 3600


def maintenance(database_url: str, retention_days: int, dry_run: bool = False) -> None:
    """Delete requests older than the retention period that never got a
    label. Labelled rows are kept for retraining. See apply_retention() in
    db/schema.sql."""
    with psycopg.connect(database_url) as conn:
        deleted = conn.execute(
            "SELECT apply_retention(%s, %s)", (retention_days, dry_run)
        ).fetchone()[0]
        rows, labelled = conn.execute(
            "SELECT (SELECT count(*) FROM requests), (SELECT count(*) FROM labeled_examples)"
        ).fetchone()
    verb = "would delete" if dry_run else "deleted"
    logger.info(
        "maintenance: %s %d unlabelled requests older than %d days; "
        "%d request rows, %d labelled examples",
        verb,
        deleted,
        retention_days,
        rows,
        labelled,
    )


def gather_state(
    conn: psycopg.Connection, config: PipelineConfig, trigger_config: TriggerConfig
) -> tuple[SystemState, dict]:
    registry = read_registry(config.models_dir)
    if registry is None:
        raise RuntimeError(f"no registry in {config.models_dir}; start the stack first")
    version = registry["current"]
    now = datetime.now(UTC)

    deployed_at = datetime.fromisoformat(registry["versions"][version]["created_at"])
    last_run = conn.execute("SELECT max(started_at) FROM retraining_runs").fetchone()[0]
    since_last_run = (now - last_run).total_seconds() if last_run else None
    last_activity = max(deployed_at, last_run) if last_run else deployed_at

    new_labels = count_new_labels(conn, previous_cutoff(conn))

    conf_mean, conf_n = conn.execute(
        """
        SELECT avg(confidence), count(*) FROM (
            SELECT confidence FROM requests
            WHERE model_version = %s AND status_code = 200 AND NOT cached
            ORDER BY ts DESC LIMIT %s
        ) recent
        """,
        (version, trigger_config.confidence_window),
    ).fetchone()
    acc_mean, acc_n = conn.execute(
        """
        SELECT avg((prediction = true_label)::int), count(*) FROM (
            SELECT r.prediction, f.true_label FROM requests r
            JOIN feedback f ON f.request_id = r.id
            WHERE r.model_version = %s
            ORDER BY r.ts DESC LIMIT %s
        ) recent
        """,
        (version, trigger_config.accuracy_window),
    ).fetchone()

    state = SystemState(
        since_last_activity_s=(now - last_activity).total_seconds(),
        since_last_run_s=since_last_run,
        new_labels=new_labels,
        recent_confidence=(
            float(conf_mean) if conf_n >= trigger_config.confidence_window else None
        ),
        recent_accuracy=float(acc_mean) if acc_n >= trigger_config.accuracy_window else None,
    )
    info = {
        "version": version,
        "conf_n": conf_n,
        "acc_n": acc_n,
        "conf": conf_mean,
        "acc": acc_mean,
    }
    return state, info


def _fmt(value) -> str:
    return "  -  " if value is None else f"{float(value):.3f}"


def status(config: PipelineConfig, trigger_config: TriggerConfig) -> tuple[str | None, str]:
    with psycopg.connect(config.database_url) as conn:
        state, info = gather_state(conn, config, trigger_config)
    trigger, reason = decide(state, trigger_config)
    logger.info(
        "model %s | confidence %s (%d/%d) | labelled acc %s (%d/%d) | new labels %d | "
        "age %.0fs -> %s",
        info["version"],
        _fmt(info["conf"]),
        info["conf_n"],
        trigger_config.confidence_window,
        _fmt(info["acc"]),
        info["acc_n"],
        trigger_config.accuracy_window,
        state.new_labels,
        state.since_last_activity_s,
        f"TRIGGER {trigger}: {reason}" if trigger else reason,
    )
    return trigger, reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["status", "watch", "run", "maintenance"])
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/tweets"),
    )
    parser.add_argument(
        "--backend-url", default=os.environ.get("BACKEND_URL", "http://localhost:8000")
    )
    parser.add_argument("--admin-token", default=os.environ.get("ADMIN_TOKEN", "change-me"))
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--trigger", default="manual", help="label for a `run`")
    parser.add_argument("--poll", type=float, default=30, help="seconds between checks")

    defaults = TriggerConfig()
    parser.add_argument(
        "--max-age",
        type=float,
        default=defaults.max_age_s,
        help="seconds before a scheduled retrain",
    )
    parser.add_argument("--confidence-threshold", type=float, default=defaults.confidence_threshold)
    parser.add_argument("--confidence-window", type=int, default=defaults.confidence_window)
    parser.add_argument("--accuracy-threshold", type=float, default=defaults.accuracy_threshold)
    parser.add_argument("--accuracy-window", type=int, default=defaults.accuracy_window)
    parser.add_argument("--min-new-labels", type=int, default=defaults.min_new_labels)
    parser.add_argument("--cooldown", type=float, default=defaults.cooldown_s)

    budget = DataBudget()
    parser.add_argument(
        "--max-train", type=int, default=budget.max_train, help="max training examples per cycle"
    )
    parser.add_argument("--max-recent", type=int, default=budget.max_recent)
    parser.add_argument("--max-eval", type=int, default=budget.max_eval)
    parser.add_argument("--recent-days", type=float, default=budget.recent_days)

    parser.add_argument(
        "--retention-days", type=int, default=30, help="delete unlabelled requests older than this"
    )
    parser.add_argument("--dry-run", action="store_true", help="maintenance: report only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = PipelineConfig(
        database_url=args.database_url,
        backend_url=args.backend_url,
        admin_token=args.admin_token,
        device=args.device,
        budget=DataBudget(
            max_train=args.max_train,
            max_recent=args.max_recent,
            max_eval=args.max_eval,
            recent_days=args.recent_days,
        ),
    )
    trigger_config = TriggerConfig(
        max_age_s=args.max_age,
        confidence_threshold=args.confidence_threshold,
        confidence_window=args.confidence_window,
        accuracy_threshold=args.accuracy_threshold,
        accuracy_window=args.accuracy_window,
        min_new_labels=args.min_new_labels,
        cooldown_s=args.cooldown,
    )

    if args.command == "status":
        status(config, trigger_config)
    elif args.command == "run":
        run_once(config, args.trigger)
    elif args.command == "maintenance":
        maintenance(args.database_url, args.retention_days, args.dry_run)
    else:
        last_maintenance = 0.0
        while True:
            try:
                if time.monotonic() - last_maintenance >= MAINTENANCE_EVERY_S:
                    maintenance(args.database_url, args.retention_days)
                    last_maintenance = time.monotonic()
                trigger, _ = status(config, trigger_config)
                if trigger:
                    run_once(config, trigger)
            except (psycopg.OperationalError, RuntimeError) as exc:
                logger.warning("check failed: %s", exc)
            time.sleep(args.poll)


if __name__ == "__main__":
    main()
