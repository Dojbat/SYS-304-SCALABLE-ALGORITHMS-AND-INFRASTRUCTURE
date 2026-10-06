"""Data drift: compare live request text against the training data.

The model only sees tweet text, so drift is measured on features of the
text itself. Each live window is compared to a reference profile computed
once from train.csv (build_reference.py -> reference_stats.json):

- Numeric features (length, word count, hashtags, mentions, URLs, share of
  uppercase / non-ASCII characters) are bucketed with the training data's
  own quantile edges and compared with the Population Stability Index.
- Vocabulary: the share of live words that fall outside the training
  vocabulary (out-of-vocabulary rate). New slang, new event names or a new
  language all push this up even when the numeric features look normal.
- Prediction mix: the share of live tweets predicted "disaster", against
  the training set's positive rate.

PSI rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
"""

import math
import re
from collections import Counter

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
WORD_RE = re.compile(r"[a-z][a-z']+")

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
VOCAB_SIZE = 5000
NUM_BINS = 10
EPS = 1e-4

NUMERIC_FEATURES = {
    "char_length": "Characters",
    "word_count": "Words",
    "hashtag_count": "Hashtags",
    "mention_count": "@mentions",
    "url_count": "URLs",
    "uppercase_ratio": "Uppercase share",
    "non_ascii_ratio": "Non-ASCII share (emoji, other scripts)",
}


def text_features(text: str) -> dict[str, float]:
    letters = [c for c in text if c.isalpha()]
    return {
        "char_length": len(text),
        "word_count": len(text.split()),
        "hashtag_count": len(HASHTAG_RE.findall(text)),
        "mention_count": len(MENTION_RE.findall(text)),
        "url_count": len(URL_RE.findall(text)),
        "uppercase_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "non_ascii_ratio": sum(ord(c) > 127 for c in text) / len(text) if text else 0.0,
    }


def words(text: str) -> list[str]:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return WORD_RE.findall(text.lower())


def bin_edges(values: list[float], num_bins: int = NUM_BINS) -> list[float]:
    """Interior quantile edges, deduplicated. Count features like
    url_count have most of their mass at 0, so many quantiles coincide."""
    ordered = sorted(values)
    n = len(ordered)
    edges = {ordered[min(n - 1, int(n * q / num_bins))] for q in range(1, num_bins)}
    return sorted(edges)


def histogram(values: list[float], edges: list[float]) -> list[float]:
    """Share of values in each bucket. Bucket i holds values v with
    edges[i-1] < v <= edges[i]; the last bucket is everything above."""
    counts = [0] * (len(edges) + 1)
    for v in values:
        i = 0
        while i < len(edges) and v > edges[i]:
            i += 1
        counts[i] += 1
    total = len(values) or 1
    return [c / total for c in counts]


def psi(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(e, EPS), max(a, EPS)
        total += (a - e) * math.log(a / e)
    return total


def psi_level(value: float) -> str:
    if value >= PSI_MAJOR:
        return "major"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def build_reference(texts: list[str], labels: list[int]) -> dict:
    features = [text_features(t) for t in texts]
    numeric = {}
    for name in NUMERIC_FEATURES:
        values = [f[name] for f in features]
        edges = bin_edges(values)
        numeric[name] = {
            "edges": edges,
            "hist": histogram(values, edges),
            "mean": sum(values) / len(values),
        }

    word_counts = Counter(w for t in texts for w in words(t))
    vocab = [w for w, _ in word_counts.most_common(VOCAB_SIZE)]
    vocab_set = set(vocab)
    total_words = sum(word_counts.values())
    in_vocab = sum(c for w, c in word_counts.items() if w in vocab_set)

    return {
        "num_texts": len(texts),
        "positive_rate": sum(labels) / len(labels),
        "numeric": numeric,
        "vocab": vocab,
        "oov_rate": 1 - in_vocab / total_words,
    }


def compare(reference: dict, texts: list[str], predictions: list[int]) -> dict:
    """Drift report for one window of live traffic against the reference."""
    features = [text_features(t) for t in texts]
    numeric = {}
    for name, ref in reference["numeric"].items():
        values = [f[name] for f in features]
        live_hist = histogram(values, ref["edges"])
        value = psi(ref["hist"], live_hist)
        numeric[name] = {
            "psi": value,
            "level": psi_level(value),
            "reference_mean": ref["mean"],
            "live_mean": sum(values) / len(values) if values else 0.0,
            "reference_hist": ref["hist"],
            "live_hist": live_hist,
            "edges": ref["edges"],
        }

    vocab = set(reference["vocab"])
    live_words = Counter(w for t in texts for w in words(t))
    total = sum(live_words.values())
    oov = Counter({w: c for w, c in live_words.items() if w not in vocab})
    oov_rate = sum(oov.values()) / total if total else 0.0

    return {
        "num_texts": len(texts),
        "numeric": numeric,
        "oov_rate": oov_rate,
        "reference_oov_rate": reference["oov_rate"],
        "top_new_words": oov.most_common(20),
        "positive_rate": sum(predictions) / len(predictions) if predictions else 0.0,
        "reference_positive_rate": reference["positive_rate"],
    }
