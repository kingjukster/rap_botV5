"""
Tests for evo_rhyme.constraints: passes_constraints, passes_verse_constraints, ConstraintConfig.
Area 3: Constraints – couplet and verse; config dict and ConstraintConfig.
"""

import pytest

from evo_rhyme.constraints import (
    ConstraintConfig,
    passes_constraints,
    passes_verse_constraints,
)
from evo_rhyme.individual import (
    CoupletIndividual,
    VerseIndividual,
    analyze_individual,
    analyze_verse_individual,
)


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

    def test_passes_constraints_config_none(self):
        """config=None uses default ConstraintConfig."""
        ind = CoupletIndividual(
            line1="The burns of life bring many concerns",
            line2="We learn from pain as the whole world turns",
        )
        analyze_individual(ind)
        assert passes_constraints(ind, None) is True

    def test_passes_constraints_with_constraint_config_instance(self):
        """config as ConstraintConfig instance is respected."""
        ind = CoupletIndividual(
            line1="The burns of life bring many concerns",
            line2="We learn from pain as the whole world turns",
        )
        analyze_individual(ind)
        cfg = ConstraintConfig(min_syllables=6, max_syllables=18, min_words_per_line=4)
        assert passes_constraints(ind, cfg) is True

    def test_passes_constraints_dict_min_syllables(self):
        """Dict config with min_syllables rejects short lines."""
        ind = CoupletIndividual(line1="I go now", line2="You stay here")
        analyze_individual(ind)
        assert passes_constraints(ind, {"min_syllables": 6}) is False

    def test_passes_constraints_dict_min_words(self):
        """Dict config with min_words_per_line rejects few words."""
        ind = CoupletIndividual(line1="One two", line2="Three four")
        analyze_individual(ind)
        assert passes_constraints(ind, {"min_words_per_line": 4}) is False


class TestConstraintConfig:
    """ConstraintConfig construction and defaults."""

    def test_default_constraint_config(self):
        cfg = ConstraintConfig()
        assert cfg.min_syllables == 6
        assert cfg.max_syllables == 18
        assert cfg.min_words_per_line == 4
        assert cfg.require_stressed_end is True
        assert "the" in cfg.weak_words

    def test_constraint_config_custom_values(self):
        cfg = ConstraintConfig(
            min_syllables=8,
            max_syllables=16,
            min_words_per_line=5,
            require_theme_presence=True,
            prompt_keywords={"flow"},
        )
        assert cfg.min_syllables == 8
        assert cfg.max_syllables == 16
        assert cfg.min_words_per_line == 5
        assert cfg.require_theme_presence is True
        assert cfg.prompt_keywords == {"flow"}


class TestPassesVerseConstraints:
    """passes_verse_constraints for 4-line verses."""

    def test_passes_verse_constraints_accept_valid_verse(self):
        verse = VerseIndividual(
            lines=[
                "I fight through pressure every night inside the dark",
                "I carve a lane through the pain and leave a mark",
                "My words are blades every phrase got a sharpened spark",
                "I move with rage but the cadence stays precise and stark",
            ]
        )
        analyze_verse_individual(verse)
        assert passes_verse_constraints(verse) is True

    def test_passes_verse_constraints_reject_too_few_syllables(self):
        verse = VerseIndividual(
            lines=[
                "I go",
                "You stay",
                "We run",
                "They play",
            ]
        )
        analyze_verse_individual(verse)
        assert passes_verse_constraints(verse) is False

    def test_passes_verse_constraints_reject_identical_lines(self):
        same = "The burns of life bring many concerns"
        verse = VerseIndividual(lines=[same, same, same, same])
        analyze_verse_individual(verse)
        assert passes_verse_constraints(verse) is False

    def test_passes_verse_constraints_config_none(self):
        verse = VerseIndividual(
            lines=[
                "I fight through pressure every night inside the dark",
                "I carve a lane through the pain and leave a mark",
                "My words are blades every phrase got a sharpened spark",
                "I move with rage but the cadence stays precise and stark",
            ]
        )
        analyze_verse_individual(verse)
        assert passes_verse_constraints(verse, None) is True

    def test_passes_verse_constraints_theme_required_reject(self):
        verse = VerseIndividual(
            lines=[
                "The burns of life bring many concerns",
                "We learn from pain as the whole world turns",
                "Diamonds on my wrist they shining bright",
                "Running through the city every night",
            ]
        )
        analyze_verse_individual(verse)
        config = {"require_theme_presence": True, "prompt_keywords": {"nonexistentkeyword"}}
        assert passes_verse_constraints(verse, config) is False

    def test_passes_verse_constraints_theme_required_accept(self):
        verse = VerseIndividual(
            lines=[
                "Pressure in the mask while the room still burns",
                "Measured every path like the truth got concerns",
                "Diamonds on my wrist they shining bright",
                "Running through the city every night",
            ]
        )
        analyze_verse_individual(verse)
        config = {"require_theme_presence": True, "prompt_keywords": {"pressure", "mask"}}
        assert passes_verse_constraints(verse, config) is True

    def test_passes_verse_constraints_reject_orphan_line(self):
        """Plan 2: reject verses with orphan phrase + no theme keyword (template contamination)."""
        verse = VerseIndividual(
            lines=[
                "My crown and empire glow with fire",
                "Flow like a river rising higher",
                "crown the empire with a po sharp as a peg",
                "They played around, and she caught a shot in the leg",  # orphan, no crown/empire
            ]
        )
        analyze_verse_individual(verse)
        config = {"prompt_keywords": {"crown", "empire", "flow"}}
        assert passes_verse_constraints(verse, config) is False
