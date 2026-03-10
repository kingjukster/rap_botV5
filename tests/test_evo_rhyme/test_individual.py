"""
Tests for evo_rhyme.individual: analyze_individual populates features correctly.
"""

import pytest

from evo_rhyme.individual import CoupletIndividual, analyze_individual


class TestAnalyzeIndividual:
    """Test analyze_individual populates features correctly."""

    def test_analyze_populates_features1_and_features2(self):
        ind = CoupletIndividual(
            line1="I got the flow when I step in the spot",
            line2="You know I rock it hard when I hit the block",
        )
        analyze_individual(ind)
        assert ind.features1 is not None
        assert ind.features2 is not None

    def test_features_have_required_fields(self):
        ind = CoupletIndividual(line1="The cat sat on the mat", line2="The dog ran in the fog")
        analyze_individual(ind)
        f1, f2 = ind.features1, ind.features2
        assert f1 is not None
        assert f2 is not None
        assert f1.text == "The cat sat on the mat"
        assert f2.text == "The dog ran in the fog"
        assert len(f1.tokens) > 0
        assert len(f2.tokens) > 0
        assert f1.syllable_count > 0
        assert f2.syllable_count > 0
        assert f1.end_tail is not None
        assert f2.end_tail is not None
        assert len(f1.phonemes) == len(f1.tokens)
        assert len(f2.phonemes) == len(f2.tokens)

    def test_stress_pattern_populated(self):
        ind = CoupletIndividual(line1="Hello world today", line2="Goodbye friend away")
        analyze_individual(ind)
        assert ind.features1 is not None
        assert ind.features2 is not None
        assert isinstance(ind.features1.stress_pattern, list)
        assert isinstance(ind.features2.stress_pattern, list)
