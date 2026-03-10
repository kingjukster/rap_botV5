"""
Tests for evo_rhyme.constraints: passes_constraints rejects weak endings, bad syllables.
"""

import pytest

from evo_rhyme.constraints import passes_constraints
from evo_rhyme.individual import CoupletIndividual, analyze_individual


class TestPassesConstraints:
    """Test passes_constraints rejects weak endings and bad syllables."""

    def test_rejects_weak_endings(self):
        """Lines ending in 'the', 'and', etc. should fail."""
        ind = CoupletIndividual(
            line1="I got the flow when I step in the",
            line2="You know I rock it hard when I hit the",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is False

    def test_rejects_too_few_words(self):
        """Lines with < 4 words should fail."""
        ind = CoupletIndividual(
            line1="Counting up",
            line2="Sipping cup",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is False

    def test_rejects_too_few_syllables(self):
        """Lines with < 6 syllables should fail."""
        ind = CoupletIndividual(
            line1="I go now",
            line2="You stay here",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is False

    def test_rejects_excessive_repetition(self):
        """Same content word > 2 times should fail (stopwords excluded)."""
        ind = CoupletIndividual(
            line1="pressure pressure pressure in the mask",
            line2="pressure pressure pressure on the path",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is False

    def test_accepts_valid_couplet(self):
        """Valid couplet with stressed endings and proper syllables should pass."""
        ind = CoupletIndividual(
            line1="The burns of life bring many concerns",
            line2="We learn from pain as the whole world turns",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is True

    def test_rejects_near_duplicate_lines(self):
        """Couplet with >82% word overlap should fail."""
        ind = CoupletIndividual(
            line1="Diamonds jump out the face",
            line2="Diamond jump out the face",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is False

    def test_rejects_identical_lines(self):
        """Couplet with line1 == line2 should fail."""
        ind = CoupletIndividual(
            line1="Everyday Halloween",
            line2="Everyday Halloween",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is False

    def test_accepts_couplet_with_repeated_stopwords(self):
        """Couplet with 'the' repeated 3x should pass (stopwords excluded from repetition check)."""
        ind = CoupletIndividual(
            line1="Pressure in the mask while the room still burns",
            line2="Measured every path like the truth got concerns",
        )
        analyze_individual(ind)
        assert passes_constraints(ind) is True

    def test_theme_presence_rejects_no_keyword(self):
        """When require_theme_presence and prompt_keywords, couplet without any keyword fails."""
        ind = CoupletIndividual(
            line1="The burns of life bring many concerns",
            line2="We learn from pain as the whole world turns",
        )
        analyze_individual(ind)
        config = {"require_theme_presence": True, "prompt_keywords": {"pressure", "mask", "survival"}}
        assert passes_constraints(ind, config) is False

    def test_theme_presence_accepts_with_keyword(self):
        """When require_theme_presence and prompt_keywords, couplet with at least one keyword passes."""
        ind = CoupletIndividual(
            line1="Pressure in the mask while the room still burns",
            line2="Measured every path like the truth got concerns",
        )
        analyze_individual(ind)
        config = {"require_theme_presence": True, "prompt_keywords": {"pressure", "mask", "survival"}}
        assert passes_constraints(ind, config) is True

    def test_theme_presence_ignored_when_disabled(self):
        """Without require_theme_presence, theme keywords are not checked."""
        ind = CoupletIndividual(
            line1="The burns of life bring many concerns",
            line2="We learn from pain as the whole world turns",
        )
        analyze_individual(ind)
        config = {"require_theme_presence": False, "prompt_keywords": {"pressure", "mask"}}
        assert passes_constraints(ind, config) is True
