import random

import drift
import pytest

ENGLISH = [
    "Forest fire near La Ronge Sask. Canada",
    "Just happened a terrible car crash #accident",
    "@friend loving the sunshine today http://t.co/abc",
    "Heard about #earthquake is different cities, stay safe everyone",
    "What a goooooooaaaaaal!!!!!!",
] * 40


def test_text_features():
    features = drift.text_features("RT @bob #Fire at http://x.co NOW 🔥")
    assert features["mention_count"] == 1
    assert features["hashtag_count"] == 1
    assert features["url_count"] == 1
    assert features["non_ascii_ratio"] > 0


def test_psi_is_zero_for_identical_and_large_for_disjoint_distributions():
    assert drift.psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert drift.psi([1.0, 0.0], [0.0, 1.0]) > drift.PSI_MAJOR


def test_same_distribution_is_stable():
    reference = drift.build_reference(ENGLISH, [1, 1, 0, 1, 0] * 40)
    report = drift.compare(reference, ENGLISH, [1] * len(ENGLISH))
    assert all(f["level"] == "stable" for f in report["numeric"].values())
    assert report["oov_rate"] == pytest.approx(reference["oov_rate"])


def test_different_language_and_length_is_flagged():
    reference = drift.build_reference(ENGLISH, [1, 1, 0, 1, 0] * 40)
    rng = random.Random(0)
    shifted = [
        " ".join(
            rng.choice(["terremoto", "inundación", "banjir", "lindol", "😭"]) for _ in range(30)
        )
        for _ in range(200)
    ]
    report = drift.compare(reference, shifted, [0] * 200)
    assert report["numeric"]["char_length"]["level"] == "major"
    assert report["oov_rate"] > 0.9
    assert report["top_new_words"]
