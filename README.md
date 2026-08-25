# Disaster Tweets — Text Classification

**Milestone 1: Problem Scoping & The Prototype** — SYS-304 Scalable Algorithms and Infrastructure

## Problem

Twitter/X is often faster than official channels at surfacing real disasters as they happen, but
disaster-flavored language is also just... how people talk ("this traffic is a disaster", "my heart
is on fire"). The task is a binary text classification problem: given a tweet, predict whether it
describes a **real disaster** (`target=1`) or not (`target=0`). A system that gets this right could
help emergency services and news organizations monitor social media for real events instead of
noise.

## Dataset

[Kaggle — NLP Getting Started: Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

- `train.csv` — 7,613 labeled tweets: `id`, `keyword`, `location`, `text`, `target`
- `test.csv` — 3,263 unlabeled tweets to predict on
- `sample_submission.csv` — expected submission format

Chosen because it's a clean, well-understood, publicly available text classification problem — a
good target for building system architecture around before scaling up, per the assignment brief.

## Milestone 1 scope

Per the assignment, this milestone is a naive proof-of-concept, not a tuned final model:

1. **EDA** (`main.ipynb`, sections 1–12): class balance, missing values, text length, surface
   features (hashtags/mentions/URLs), the `keyword` column's per-class signal, feature correlations.
2. **Baseline model** (sections 13–17): TF-IDF over `text` + a handful of engineered numeric
   features (length, hashtag/mention/URL counts), fed into an untuned `RandomForestClassifier`. No
   hyperparameter search — a working proof-of-concept that maps tweet text to a prediction.
3. **Saved weights**: `baseline_model.pkl` — the fitted `sklearn` `Pipeline` (TF-IDF vectorizer +
   scaler + classifier) as one pickled unit, reloadable for inference without refitting.

The notebook keeps going well past this (a stronger TF-IDF model, zero-shot LLM prompting with
Qwen, and RoBERTa fine-tuning with LoRA) as an exploration of what later milestones in this course
are presumably about — scaling the naive baseline up. That work is included for context but isn't
the Milestone 1 deliverable; the sections above are.

## Repository contents

- `main.ipynb` — full data pipeline, training, and evaluation (baseline through the later
  exploration sections)
- `baseline_model.pkl` — the saved Milestone 1 baseline model

Raw competition data (`train.csv`, `test.csv`, `sample_submission.csv`) isn't bundled in this repo —
download it from the [competition data page](https://www.kaggle.com/competitions/nlp-getting-started/data)
and place it alongside `main.ipynb` before running.

## Running it

```
pip install pandas numpy scikit-learn matplotlib joblib
jupyter notebook main.ipynb
```

Run top to bottom, or skip straight to loading `baseline_model.pkl` with `joblib.load(...)` to
reproduce predictions without retraining (sections 13–17 only need `pandas`/`numpy`/`scikit-learn`
— the later sections need `torch`, `transformers`, and `peft` as well, and a lot more time/compute).
