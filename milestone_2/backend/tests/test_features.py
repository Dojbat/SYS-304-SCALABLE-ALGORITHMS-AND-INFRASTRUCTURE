from app.features import _HASHTAG_RE, FEATURE_COLS, build_features, count_pattern


def test_output_columns_match_feature_cols():
    df = build_features(["hello world"])
    assert list(df.columns) == FEATURE_COLS


def test_counts_hashtags_mentions_urls():
    df = build_features(["big #fire near @city see http://example.com now!!"])
    row = df.iloc[0]
    assert row["n_hashtags"] == 1
    assert row["n_mentions"] == 1
    assert row["n_urls"] == 1
    assert row["n_exclaim"] == 2
    assert row["has_url"] == 1


def test_has_url_is_strictly_binary():
    df = build_features(["http://a.com http://b.com http://c.com", "no links here"])
    assert set(df["has_url"].tolist()) <= {0, 1}
    assert df.iloc[0]["has_url"] == 1
    assert df.iloc[1]["has_url"] == 0


def test_no_surface_features_yields_all_zero_numeric():
    df = build_features(["plain text no symbols"])
    row = df.iloc[0]
    assert row["n_hashtags"] == 0
    assert row["n_mentions"] == 0
    assert row["n_urls"] == 0
    assert row["n_exclaim"] == 0
    assert row["has_url"] == 0


def test_char_and_word_len():
    df = build_features(["one two three"])
    row = df.iloc[0]
    assert row["char_len"] == len("one two three")
    assert row["word_len"] == 3


def test_unicode_and_emoji_do_not_crash():
    df = build_features(["🔥🔥 fire emergency 火事 #救助"])
    assert len(df) == 1
    assert df.iloc[0]["n_hashtags"] == 1


def test_count_pattern_matches_hashtag_regex():
    assert count_pattern("#a #b #c", _HASHTAG_RE) == 3
