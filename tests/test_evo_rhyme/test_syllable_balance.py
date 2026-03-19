"""Tests for evo_rhyme.scoring.syllable_balance."""

import pytest

from evo_rhyme.scoring.syllable_balance import score_syllable_balance


def test_score_syllable_balance_perfect():
    assert score_syllable_balance(10, 10) == 1.0
    assert score_syllable_balance(12, 12) == 1.0
    assert score_syllable_balance(10, 11) == 1.0


def test_score_syllable_balance_out_of_range():
    assert score_syllable_balance(5, 5) < 1.0
    assert score_syllable_balance(20, 20) < 1.0


def test_score_syllable_balance_imbalanced():
    # diff > 1 reduces balance_score
    s = score_syllable_balance(10, 14)
    assert 0 <= s <= 1
    assert score_syllable_balance(10, 12) >= score_syllable_balance(10, 15)


def test_score_syllable_balance_custom_range():
    assert score_syllable_balance(8, 8, target_min=8, target_max=12) == 1.0
    assert score_syllable_balance(6, 6, target_min=8, target_max=12) < 1.0
