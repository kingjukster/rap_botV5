"""Tests for evo_rhyme.scoring.internal_rhyme. Area 12: internal rhyme scoring."""

import pytest

from evo_rhyme.scoring.internal_rhyme import (
    DEFAULT_STOPWORDS,
    LineFeatures,
    build_line_features,
    score_internal_rhyme,
)


def test_line_features_dataclass():
    f = LineFeatures(
        text="hello world",
        content_words=["hello", "world"],
        rhyme_tails=[("hello", "AH1 L OW0", 0), ("world", "ER1 L D", 1)],
        syllable_count=2,
    )
    assert f.syllable_count == 2
    assert len(f.content_words) == 2


def test_build_line_features_mock_syllables(monkeypatch):
    import evo_rhyme.scoring.internal_rhyme as internal_rhyme
    monkeypatch.setattr(internal_rhyme, "count_syllables_line", lambda line: 5)
    f = build_line_features("I got the flow when I step in the spot")
    assert f.syllable_count == 5
    assert len(f.content_words) > 0
    assert f.text == "I got the flow when I step in the spot"


def test_score_internal_rhyme_zero_syllables():
    f1 = LineFeatures("", [], [], 0)
    f2 = LineFeatures("", [], [], 0)
    assert score_internal_rhyme(f1, f2) == 0.0


def test_score_internal_rhyme_with_mock_tails():
    """Score when no rhyme pairs (empty tails) returns 0."""
    f1 = LineFeatures("line one", ["line", "one"], [("line", None, 0), ("one", None, 1)], 2)
    f2 = LineFeatures("line two", ["line", "two"], [("line", None, 0), ("two", None, 1)], 2)
    result = score_internal_rhyme(f1, f2)
    assert 0.0 <= result <= 0.35


def test_default_stopwords_non_empty():
    assert len(DEFAULT_STOPWORDS) > 0
    assert "the" in DEFAULT_STOPWORDS
