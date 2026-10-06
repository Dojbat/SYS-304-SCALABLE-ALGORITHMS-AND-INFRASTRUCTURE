import random

from drift_texts import generate


def test_generated_tweets_are_varied_and_fully_filled():
    rng = random.Random(0)
    samples = [generate(rng) for _ in range(2000)]
    texts = [t for t, _ in samples]
    assert len(set(texts)) > 1000
    assert not any("{" in t or "}" in t for t in texts)
    assert {label for _, label in samples} == {0, 1}


def test_generation_is_reproducible_with_a_seed():
    assert [generate(random.Random(5)) for _ in range(3)] == [
        generate(random.Random(5)) for _ in range(3)
    ]
