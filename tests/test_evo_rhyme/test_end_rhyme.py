"""
Tests for evo_rhyme.scoring.end_rhyme: score_end_rhyme for matching and non-matching pairs.
"""

import pytest

from evo_rhyme.phonetics import extract_rhyme_tail
from evo_rhyme.scoring.end_rhyme import score_end_rhyme


class TestScoreEndRhyme:
    """Test score_end_rhyme for matching and non-matching pairs."""

    def test_matching_pair_burns_concerns(self):
        t1 = extract_rhyme_tail("burns")
        t2 = extract_rhyme_tail("concerns")
        assert t1 is not None
        assert t2 is not None
        score = score_end_rhyme(t1, t2)
        assert score >= 0.9

    def test_exact_match(self):
        tail = "ER1 N Z"
        assert score_end_rhyme(tail, tail) == 1.0

    def test_non_matching_pair(self):
        t1 = extract_rhyme_tail("cat")
        t2 = extract_rhyme_tail("dog")
        assert t1 is not None
        assert t2 is not None
        score = score_end_rhyme(t1, t2)
        assert score < 0.5

    def test_empty_tails_return_zero(self):
        assert score_end_rhyme("", "ER1 N Z") == 0.0
        assert score_end_rhyme("ER1 N Z", "") == 0.0
        assert score_end_rhyme("", "") == 0.0

    def test_none_like_empty(self):
        assert score_end_rhyme("  ", "ER1 N Z") == 0.0
