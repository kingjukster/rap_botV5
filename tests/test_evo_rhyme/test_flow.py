"""Tests for evo_rhyme.flow: stress pattern extraction and flow scoring."""

import pytest

from evo_rhyme.flow import (
    DEFAULT_FLOW_TEMPLATE,
    compare_pattern,
    extract_stress_pattern,
    score_flow,
    score_verse_flow,
)
from evo_rhyme.individual import VerseFeatures


def test_default_flow_template():
    assert len(DEFAULT_FLOW_TEMPLATE) >= 8
    assert all(x in (0, 1) for x in DEFAULT_FLOW_TEMPLATE)


def test_compare_pattern_empty():
    assert compare_pattern([], [1, 0, 1]) == 0.5
    assert compare_pattern([1, 0], []) == 0.5


def test_compare_pattern_match():
    assert compare_pattern([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0


def test_compare_pattern_half_match():
    assert compare_pattern([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0
    assert compare_pattern([0, 1, 0, 1], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_extract_stress_pattern():
    pattern = extract_stress_pattern("hello world")
    assert isinstance(pattern, list)
    assert all(x in (0, 1) for x in pattern)


def test_score_flow():
    s = score_flow("I got the flow when I step in the spot")
    assert 0 <= s <= 1


def test_score_flow_custom_template():
    template = [0, 1, 0, 1]
    s = score_flow("test line here", template=template)
    assert 0 <= s <= 1


def test_score_verse_flow():
    lines = ["first line here", "second line there", "third", "fourth"]
    s = score_verse_flow(lines)
    assert 0 <= s <= 1


def test_score_verse_flow_empty_lines():
    """Empty lines returns neutral 0.5."""
    assert score_verse_flow([]) == 0.5


def test_score_verse_flow_with_features():
    """When features.stress_patterns is provided, uses them instead of re-extracting."""
    features = VerseFeatures(
        tokens_per_line=[["a", "b"], ["c", "d"]],
        phonemes=[[], []],
        syllable_counts=[2, 2],
        end_tails=[None, None],
        stress_patterns=[[0, 1], [1, 0]],
    )
    s = score_verse_flow(["line one", "line two"], features=features)
    assert 0 <= s <= 1


def test_implied_stress_pattern():
    from evo_rhyme.flow import implied_stress_pattern
    pattern = implied_stress_pattern("I {verb} through the {noun}")
    assert all(x in (0, 1) for x in pattern)
    assert len(pattern) == 5


def test_template_flow_score():
    from evo_rhyme.flow import template_flow_score
    s = template_flow_score("I got the flow")
    assert 0 <= s <= 1
