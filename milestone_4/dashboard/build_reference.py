"""Build the training-data profile the drift dashboard compares against.

train.csv isn't in the repo (Kaggle data), and the dashboard container
shouldn't need it at runtime, so the profile is computed once here and the
small JSON it produces is committed.

Run from the repo root:
    python milestone_4/dashboard/build_reference.py
"""

import csv
import json
from pathlib import Path

from drift import build_reference

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = Path(__file__).resolve().parent / "reference_stats.json"


def main() -> None:
    with open(REPO_ROOT / "train.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    texts = [row["text"] for row in rows]
    labels = [int(row["target"]) for row in rows]

    reference = build_reference(texts, labels)
    OUT_PATH.write_text(json.dumps(reference, indent=1), encoding="utf-8")
    print(
        f"wrote {OUT_PATH}: {reference['num_texts']} tweets, "
        f"positive rate {reference['positive_rate']:.3f}, "
        f"vocab {len(reference['vocab'])}, OOV rate {reference['oov_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
