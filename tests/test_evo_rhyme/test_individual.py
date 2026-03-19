"""
Tests for evo_rhyme.individual: analyze_individual, analyze_verse_individual, VerseFeatures, edge cases.
Area 11: Individual and verse features.
"""

import pytest

from evo_rhyme.individual import (
    CoupletIndividual,
    LineFeatures,
    VerseIndividual,
    VerseFeatures,
    VerseStructure,
    analyze_individual,
    analyze_verse_individual,
    create_verse_individual,
)


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

    def test_analyze_individual_empty_line(self):
        """Empty line still produces LineFeatures with empty tokens."""
        ind = CoupletIndividual(line1="", line2="one word")
        analyze_individual(ind)
        assert ind.features1 is not None
        assert ind.features2 is not None
        assert ind.features1.tokens == []
        assert ind.features1.syllable_count == 0


class TestAnalyzeVerseIndividual:
    """analyze_verse_individual and VerseFeatures."""

    def test_analyze_verse_individual_populates_features(self):
        verse = VerseIndividual(
            lines=[
                "I fight through pressure every night inside the dark",
                "I carve a lane through the pain and leave a mark",
                "My words are blades every phrase got a sharpened spark",
                "I move with rage but the cadence stays precise and stark",
            ]
        )
        analyze_verse_individual(verse)
        assert verse.features is not None
        assert len(verse.features.syllable_counts) == 4
        assert len(verse.features.tokens_per_line) == 4
        assert len(verse.features.end_tails) == 4

    def test_analyze_verse_individual_wrong_line_count_unchanged(self):
        verse = VerseIndividual(lines=["one", "two"])
        analyze_verse_individual(verse)
        assert verse.features is None


class TestVerseStructure:
    """VerseStructure for_scheme and to_dict/from_dict."""

    def test_verse_structure_for_scheme(self):
        s = VerseStructure.for_scheme("AABB", num_lines=4)
        assert s.scheme == "AABB"
        assert len(s.roles) == 4

    def test_verse_structure_for_scheme_extended(self):
        """When num_lines > len(scheme), scheme is repeated (e.g. AB -> ABAB for 4)."""
        s = VerseStructure.for_scheme("AB", num_lines=4)
        assert s.scheme == "ABAB"
        assert len(s.roles) == 4

    def test_verse_structure_to_dict_from_dict(self):
        s = VerseStructure(scheme="ABAB", roles=["a", "b", "a", "b"])
        d = s.to_dict()
        s2 = VerseStructure.from_dict(d)
        assert s2.scheme == s.scheme
        assert s2.roles == s.roles


class TestCreateVerseIndividual:
    """create_verse_individual."""

    def test_create_verse_individual(self):
        ind = create_verse_individual(
            ["line1", "line2", "line3", "line4"],
            scheme="AABB",
        )
        assert ind.lines == ["line1", "line2", "line3", "line4"]
        assert ind.structure is not None
        assert ind.structure.scheme == "AABB"
        assert ind.metadata.get("scheme") == "AABB"

    def test_create_verse_individual_with_roles(self):
        """create_verse_individual with roles sets structure.roles; callbacks filled."""
        ind = create_verse_individual(
            ["a", "b", "c", "d"],
            scheme="AABB",
            roles=["setup", "flex", "flex", "punchline"],
        )
        assert ind.structure.roles == ["setup", "flex", "flex", "punchline"]
        assert ind.structure.callbacks == [None, None, None, None]
