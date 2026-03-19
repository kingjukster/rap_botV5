"""Tests for evo_rhyme.scoring.penalties."""

import pytest

from evo_rhyme.scoring.penalties import (
    DEFAULT_WEAK_ENDINGS,
    repetition_penalty,
    weak_tail_penalty,
)


def test_weak_tail_penalty_weak_ending():
    assert weak_tail_penalty("I got the flow when I step in the") == 1.0
    assert weak_tail_penalty("running with the") == 1.0


def test_weak_tail_penalty_strong_ending():
    assert weak_tail_penalty("I got the flow when I step in the spot") == 0.0
    assert weak_tail_penalty("hello world") == 0.0


def test_weak_tail_penalty_empty():
    assert weak_tail_penalty("") == 0.0


def test_weak_tail_penalty_custom_set():
    assert weak_tail_penalty("line ending in foo", weak_endings={"foo"}) == 1.0
    assert weak_tail_penalty("line ending in bar", weak_endings={"foo"}) == 0.0


def test_repetition_penalty_empty():
    assert repetition_penalty([]) == 0.0
    assert repetition_penalty(["only one"]) == 0.0


def test_repetition_penalty_duplicates():
    assert repetition_penalty(["same line", "same line"]) > 0
    assert repetition_penalty(["a", "b", "a", "b"]) > 0


def test_repetition_penalty_all_unique():
    assert repetition_penalty(["first line", "second line"]) == 0.0


def test_default_weak_endings_loaded():
    assert "the" in DEFAULT_WEAK_ENDINGS
    assert "and" in DEFAULT_WEAK_ENDINGS
