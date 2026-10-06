"""Training and evaluation data for one retraining cycle, on a fixed budget.

Every cycle continues training the *current* model, which has already
learned everything before its own data cutoff. So a cycle only needs to
teach what is new, plus enough older data to stop the model forgetting it.
That lets each cycle read a bounded amount of data however much has piled up
in the database. Each query has a LIMIT:

- recent window: labelled examples since the current model's data cutoff
  (at most `recent_days` back). If there are more than the budget allows,
  the ones the current model got wrong or was least sure about come first;
  those teach the most.
- older live replay: a random sample of labelled examples from before the
  window.
- original replay: a sample of train.csv, as large as the live training data
  (within limits), so the model keeps the original task.
- guard set: a fixed 500-tweet sample of train.csv, never trained on, to
  check that a new model hasn't forgotten the original task.
- evaluation: the most recent held-out live examples, capped.

Live examples are split train/eval by a hash of the text, computed the same
way in Python and in SQL, so a tweet is always on the same side of the
split and no model ever trains on a held-out tweet, in any cycle.
"""

import csv
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

EVAL_BUCKETS = 5  # 1 in 5 live tweets is held out for evaluation
GUARD_SIZE = 500
GUARD_SEED = 1234

# Must match is_eval() below: first byte of SHA-256 of the UTF-8 text, mod 5.
# mod() rather than %, which psycopg would read as a parameter placeholder
EVAL_BUCKET_SQL = f"mod(get_byte(sha256(convert_to(text, 'UTF8')), 0), {EVAL_BUCKETS})"


@dataclass(frozen=True)
class DataBudget:
    max_train: int = 20_000
    max_recent: int = 10_000
    max_older_live: int = 5_000
    max_original: int = 5_000
    min_original: int = 500
    max_eval: int = 2_000
    recent_days: float = 7.0
    # the majority class may be at most this share of the training set
    max_class_share: float = 0.7


@dataclass(frozen=True)
class Example:
    text: str
    label: int


@dataclass
class CycleData:
    live_train: list[Example]
    live_eval: list[Example]
    replay: list[Example]
    guard: list[Example]
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def train(self) -> list[Example]:
        return self.live_train + self.replay


def is_eval(text: str) -> bool:
    return hashlib.sha256(text.encode("utf-8")).digest()[0] % EVAL_BUCKETS == 0


def window_start(cutoff: datetime, model_cutoff: datetime | None, budget: DataBudget) -> datetime:
    """The recent window starts where the current model's training data
    ended, but never more than `recent_days` back."""
    earliest = cutoff - timedelta(days=budget.recent_days)
    return max(model_cutoff, earliest) if model_cutoff else earliest


# ----------------------------------------------------------------- queries


def fetch_recent_eval(conn, since: datetime, cutoff: datetime, limit: int) -> list[Example]:
    rows = conn.execute(
        f"""
        SELECT text, true_label FROM labeled_examples
        WHERE received_at > %s AND received_at <= %s AND {EVAL_BUCKET_SQL} = 0
        ORDER BY received_at DESC
        LIMIT %s
        """,
        (since, cutoff, limit),
    ).fetchall()
    return [Example(text, int(label)) for text, label in rows]


def fetch_recent_train(conn, since: datetime, cutoff: datetime, limit: int) -> list[Example]:
    rows = conn.execute(
        f"""
        SELECT text, true_label FROM labeled_examples
        WHERE received_at > %s AND received_at <= %s AND {EVAL_BUCKET_SQL} <> 0
        ORDER BY (prediction IS DISTINCT FROM true_label) DESC,  -- mistakes first
                 confidence ASC NULLS LAST,                      -- then least sure
                 received_at DESC
        LIMIT %s
        """,
        (since, cutoff, limit),
    ).fetchall()
    return [Example(text, int(label)) for text, label in rows]


def fetch_older_train(conn, since: datetime, limit: int) -> list[Example]:
    # ORDER BY random() scans the older labelled rows; labels are the scarce
    # part of the data, so this stays small long after `requests` is huge.
    rows = conn.execute(
        f"""
        SELECT text, true_label FROM labeled_examples
        WHERE received_at <= %s AND {EVAL_BUCKET_SQL} <> 0
        ORDER BY random()
        LIMIT %s
        """,
        (since, limit),
    ).fetchall()
    return [Example(text, int(label)) for text, label in rows]


def count_new_labels(conn, since: datetime | None) -> int:
    if since is None:
        return conn.execute("SELECT count(*) FROM labeled_examples").fetchone()[0]
    return conn.execute(
        "SELECT count(*) FROM labeled_examples WHERE received_at > %s", (since,)
    ).fetchone()[0]


def load_original(train_csv: Path) -> list[Example]:
    with open(train_csv, encoding="utf-8", newline="") as f:
        return [Example(row["text"], int(row["target"])) for row in csv.DictReader(f)]


# --------------------------------------------------------------- selection


def _dedupe(examples: list[Example], exclude: set[str]) -> list[Example]:
    seen = set(exclude)
    out = []
    for example in examples:
        if example.text not in seen:
            seen.add(example.text)
            out.append(example)
    return out


def balance(examples: list[Example], max_share: float) -> list[Example]:
    """Drop majority-class examples from the end of the list (lowest
    priority) until that class is at most `max_share` of the total."""
    positives = sum(e.label for e in examples)
    negatives = len(examples) - positives
    majority = 1 if positives > negatives else 0
    minority_count = min(positives, negatives)
    if minority_count == 0:
        return examples
    allowed = int(minority_count * max_share / (1 - max_share))
    excess = max(positives, negatives) - allowed
    if excess <= 0:
        return examples
    out = []
    for example in reversed(examples):
        if excess > 0 and example.label == majority:
            excess -= 1
            continue
        out.append(example)
    out.reverse()
    return out


def build(
    recent_train: list[Example],
    recent_eval: list[Example],
    older_train: list[Example],
    original: list[Example],
    budget: DataBudget,
    seed: int = 0,
) -> CycleData:
    """Assemble one cycle's data from already-bounded query results. Training
    priority, highest first: recent live (mistakes first), older live, then
    original replay. Class balancing and the total cap cut from the lowest
    priority, except that original replay keeps at least `min_original`."""
    rng = random.Random(seed)
    shuffled = original[:]
    random.Random(GUARD_SEED).shuffle(shuffled)
    guard = shuffled[:GUARD_SIZE]
    guard_texts = {e.text for e in guard}

    live_eval = _dedupe(recent_eval, exclude=set())[: budget.max_eval]
    held_out = guard_texts | {e.text for e in live_eval}

    recent = _dedupe(recent_train, held_out)[: budget.max_recent]
    older = _dedupe(older_train, held_out | {e.text for e in recent})[: budget.max_older_live]
    live = recent + older

    pool = [e for e in shuffled[GUARD_SIZE:] if e.text not in held_out]
    original_size = min(len(pool), budget.max_original, max(budget.min_original, len(live)))
    replay = rng.sample(pool, original_size)

    recent_ids, older_ids = set(map(id, recent)), set(map(id, older))
    balanced = balance(live + replay, budget.max_class_share)
    final_live = [e for e in balanced if id(e) in recent_ids or id(e) in older_ids]
    final_replay = [e for e in balanced if id(e) not in recent_ids and id(e) not in older_ids]
    if len(final_live) + len(final_replay) > budget.max_train:
        # the total cap trims live data from the low-priority end, but never
        # takes original replay below its floor: that is what stops the model
        # forgetting the original task
        replay_keep = min(
            len(final_replay),
            max(budget.min_original, budget.max_train - len(final_live)),
            budget.max_train,
        )
        final_replay = final_replay[:replay_keep]
        final_live = final_live[: budget.max_train - replay_keep]

    stats = {
        "recent_fetched": len(recent_train),
        "older_fetched": len(older_train),
        "eval_fetched": len(recent_eval),
        "recent_used": sum(1 for e in final_live if id(e) in recent_ids),
        "older_used": sum(1 for e in final_live if id(e) in older_ids),
        "original_used": len(final_replay),
        "eval_used": len(live_eval),
    }
    return CycleData(final_live, live_eval, final_replay, guard, stats)
