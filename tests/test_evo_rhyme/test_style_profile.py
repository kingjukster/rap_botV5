"""Tests for evo_rhyme.style_profile: build_style_profile, _lines_from_corpus (via build), score_style_similarity.
Area 10: Style profile.
"""

import pytest

from evo_rhyme.fitness import score_style_similarity
from evo_rhyme.individual import CoupletIndividual, analyze_individual
from evo_rhyme.style_profile import StyleProfile, build_style_profile


def test_build_style_profile_empty_returns_defaults():
    profile = build_style_profile([])
    assert profile.avg_syllables_per_line == 10.0
    assert profile.avg_word_length == 4.0
    assert profile.word_freq_dist == []
    assert profile.rhyme_tail_dist == []
    assert profile.line_length_std == 0.0


def test_build_style_profile_single_line():
    profile = build_style_profile(["The quick brown fox jumps over the lazy dog"])
    assert profile.avg_syllables_per_line > 0
    assert profile.avg_word_length > 0
    assert len(profile.word_freq_dist) > 0
    assert profile.line_length_std >= 0.0


def test_build_style_profile_couplets_with_pipe():
    """_lines_from_corpus flattens couplets separated by |."""
    profile = build_style_profile(["line one here | line two there", "single line"])
    assert profile.avg_syllables_per_line > 0
    assert len(profile.word_freq_dist) >= 3


def test_build_style_profile_newline_separated():
    """_lines_from_corpus flattens newline-separated couplets."""
    profile = build_style_profile(["first line\nsecond line"])
    assert profile.avg_syllables_per_line > 0


def test_score_style_similarity_with_profile_and_couplet():
    profile = build_style_profile([
        "The burns of life bring many concerns",
        "We learn from pain as the whole world turns",
    ])
    ind = CoupletIndividual(
        line1="The burns of life bring many concerns",
        line2="We learn from pain as the whole world turns",
    )
    analyze_individual(ind)
    score = score_style_similarity(ind, profile)
    assert 0.0 <= score <= 1.0


def test_score_style_similarity_no_features_returns_neutral():
    profile = StyleProfile(
        avg_syllables_per_line=10.0,
        avg_word_length=4.0,
        word_freq_dist=[],
        rhyme_tail_dist=[],
        line_length_std=1.0,
    )
    ind = CoupletIndividual(line1="a", line2="b")
    # No analyze_individual -> no features
    score = score_style_similarity(ind, profile)
    assert score == 0.5
