"""
Tests for evo_rhyme.beat: grid, prosody, alignment, scoring (Phase 1 beat-fit layer).
"""

import pytest

from evo_rhyme.beat import (
    BeatSlot,
    make_4_4_bar_grid,
    make_multi_bar_grid,
    SyllableUnit,
    extract_syllable_units,
    count_syllables_in_line,
    AlignmentResult,
    score_line_against_slots,
    pretty_print_alignment,
    LineBeatScore,
    VerseBeatScore,
    score_verse_lines,
)
from evo_rhyme.beat.grid import default_4_4_16_slot_strengths


class TestBeatGrid:
    """Phase 1: Beat grid construction."""

    def test_make_4_4_bar_grid_returns_16_slots(self):
        slots = make_4_4_bar_grid()
        assert len(slots) == 16

    def test_slot_labels_1_e_and_a(self):
        slots = make_4_4_bar_grid()
        labels = [s.label for s in slots]
        # 4/4 bar: 1 e & a | 2 e & a | 3 e & a | 4 e & a
        expected = ["1", "e", "&", "a", "2", "e", "&", "a", "3", "e", "&", "a", "4", "e", "&", "a"]
        assert labels == expected

    def test_slot_strengths_downbeats_higher(self):
        strengths = default_4_4_16_slot_strengths()
        assert strengths[0] > strengths[1]  # 1 stronger than e
        assert strengths[12] > strengths[13]  # 4 stronger than e

    def test_make_multi_bar_grid(self):
        slots = make_multi_bar_grid(2)
        assert len(slots) == 32
        assert slots[0].index == 0
        # Second bar starts at global index 16; first slot of bar 2 has beat_index 0
        assert slots[16].index == 16
        assert slots[16].beat_index == 0

    def test_make_multi_bar_grid_invalid_raises(self):
        """num_bars < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_bars must be"):
            make_multi_bar_grid(0)


class TestAlignmentHelpers:
    """density_penalty and overflow_penalty."""

    def test_density_penalty_zero_slots(self):
        from evo_rhyme.beat.alignment import density_penalty
        assert density_penalty(5, 0) == 2.0

    def test_density_penalty_low_density(self):
        from evo_rhyme.beat.alignment import density_penalty
        assert density_penalty(4, 8) == 0.0


class TestProsody:
    """Phase 1: Syllable extraction and stress estimation."""

    def test_extract_syllable_units_non_empty(self):
        units = extract_syllable_units("I fight through pressure")
        assert len(units) >= 3
        for u in units:
            assert isinstance(u, SyllableUnit)
            assert u.text
            assert 0 <= u.stress <= 1.0

    def test_last_syllable_is_line_final(self):
        units = extract_syllable_units("fight the pressure")
        assert units[-1].is_line_final

    def test_rhyme_zone_marked(self):
        units = extract_syllable_units("I run through pressure every night", rhyme_zone_last_n_syllables=2)
        assert any(u.is_rhyme_zone for u in units)

    def test_count_syllables_in_line(self):
        n = count_syllables_in_line("pressure")
        assert n >= 1
        assert n <= 3

    def test_empty_line_returns_empty(self):
        units = extract_syllable_units("")
        assert units == []


class TestAlignment:
    """Phase 1: DP syllable-to-grid alignment."""

    def test_score_line_against_slots_returns_result(self):
        result = score_line_against_slots("I fight through pressure every night", make_4_4_bar_grid())
        assert isinstance(result, AlignmentResult)
        assert isinstance(result.score, (int, float))
        assert result.syllables

    def test_empty_line_low_score(self):
        slots = make_4_4_bar_grid()
        result = score_line_against_slots("", slots)
        assert result.score < 0

    def test_long_line_handled(self):
        # Line with >16 syllables: DP may use rests/stretches or overflow; must not crash
        line = "I am fighting through the pressure every single night inside the dark"
        result = score_line_against_slots(line, make_4_4_bar_grid())
        assert isinstance(result.score, (int, float))
        assert len(result.syllables) > 16

    def test_pretty_print_alignment_non_empty(self):
        result = score_line_against_slots("I fight pressure", make_4_4_bar_grid())
        s = pretty_print_alignment(result, make_4_4_bar_grid())
        assert "1:" in s or "e:" in s


class TestVerseBeatScoring:
    """Phase 1: Verse-level beat-fit aggregation."""

    def test_score_verse_lines_returns_verse_beat_score(self):
        lines = [
            "I fight through pressure every night in the dark",
            "I carve a lane through the pain and leave a mark",
        ]
        verse = score_verse_lines(lines)
        assert isinstance(verse, VerseBeatScore)
        assert len(verse.line_scores) == 2
        assert verse.total_score == sum(ls.score for ls in verse.line_scores) - verse.line_balance_penalty

    def test_empty_lines_returns_negative_score(self):
        verse = score_verse_lines([])
        assert verse.total_score < 0
