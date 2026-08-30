"""Feature engineering for the disaster-tweet baseline model.

Ported verbatim from main.ipynb (inference-time cells, see baseline_model.pkl
training section) so predictions at serve time match the columns the model
was fit on. FEATURE_COLS defines both the required columns and their order —
the ColumnTransformer inside baseline_model.pkl was fit against this exact
order.
"""

import re

import pandas as pd

NUMERIC_FEATURES = [
    "char_len",
    "word_len",
    "n_hashtags",
    "n_mentions",
    "n_urls",
    "n_exclaim",
    "has_url",
]

FEATURE_COLS = ["text"] + NUMERIC_FEATURES

_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"http\S+")


def count_pattern(s: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(s))


def build_features(texts: list[str]) -> pd.DataFrame:
    """Build the 8-column DataFrame baseline_model.pkl expects from raw tweet text."""
    df = pd.DataFrame({"text": texts})
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().str.len()
    df["n_hashtags"] = df["text"].apply(lambda s: count_pattern(s, _HASHTAG_RE))
    df["n_mentions"] = df["text"].apply(lambda s: count_pattern(s, _MENTION_RE))
    df["n_urls"] = df["text"].apply(lambda s: count_pattern(s, _URL_RE))
    df["n_exclaim"] = df["text"].apply(lambda s: s.count("!"))
    df["has_url"] = (df["n_urls"] > 0).astype(int)
    return df[FEATURE_COLS]
