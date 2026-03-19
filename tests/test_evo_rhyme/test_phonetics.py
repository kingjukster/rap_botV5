"""
Tests for evo_rhyme.phonetics: extract_rhyme_tail, count_syllables, tokenize_line, syllable_count_line, phonetic_similarity.
Area 11: Phonetics public helpers.
"""

import pytest

from evo_rhyme.phonetics import (
    count_syllables,
    extract_last_syllable,
    extract_rhyme_tail,
    has_custom_pronunciations,
    multisyllable_overlap,
    phonetic_similarity,
    syllable_count_line,
    tokenize_line,
)


class TestExtractRhymeTail:
    """Test extract_rhyme_tail for known words."""

    def test_burns(self):
        tail = extract_rhyme_tail("burns")
        assert tail is not None
        assert "ER1" in tail
        assert "N" in tail
        assert "Z" in tail

    def test_concerns(self):
        tail = extract_rhyme_tail("concerns")
        assert tail is not None
        assert "ER1" in tail
        assert "N" in tail
        assert "Z" in tail

    def test_burns_concerns_rhyme(self):
        """burns and concerns should have matching rhyme tails."""
        t1 = extract_rhyme_tail("burns")
        t2 = extract_rhyme_tail("concerns")
        assert t1 is not None
        assert t2 is not None
        assert t1 == t2


class TestCountSyllables:
    """Test count_syllables for known words."""

    def test_burns(self):
        assert count_syllables("burns") == 1

    def test_concerns(self):
        assert count_syllables("concerns") == 2

    def test_single_syllable_words(self):
        assert count_syllables("cat") == 1
        assert count_syllables("dog") == 1

    def test_multisyllabic(self):
        assert count_syllables("water") >= 2
        assert count_syllables("beautiful") >= 3


class TestPhoneticSimilarity:
    """Test phonetic_similarity for rhyme pairs."""

    def test_burns_concerns_high_similarity(self):
        f1 = extract_last_syllable("burns")
        f2 = extract_last_syllable("concerns")
        assert f1 is not None
        assert f2 is not None
        sim = phonetic_similarity(f1, f2)
        assert sim >= 0.8

    def test_unrelated_words_low_similarity(self):
        f1 = extract_last_syllable("cat")
        f2 = extract_last_syllable("dog")
        assert f1 is not None
        assert f2 is not None
        sim = phonetic_similarity(f1, f2)
        assert sim < 0.5

    def test_none_returns_zero(self):
        f1 = extract_last_syllable("cat")
        assert phonetic_similarity(None, f1) == 0.0
        assert phonetic_similarity(f1, None) == 0.0


@pytest.mark.skipif(
    not has_custom_pronunciations(),
    reason="custom_pronunciation.json not available (git-lfs pointer or missing)",
)
class TestCustomPronunciation:
    """Test custom_pronunciation.json OOV words (tryna, fiya, opp)."""

    def test_tryna_has_pronunciation(self):
        """tryna is OOV; custom_pronunciation should provide it."""
        tail = extract_rhyme_tail("tryna")
        assert tail is not None
        assert count_syllables("tryna") >= 1

    def test_opp_has_pronunciation(self):
        """opp is OOV; custom_pronunciation should provide it."""
        tail = extract_rhyme_tail("opp")
        assert tail is not None


class TestMultisyllableOverlap:
    """Test multisyllable_overlap for rhyme tail phoneme matching."""

    def test_exact_match_two_phonemes(self):
        assert multisyllable_overlap("IY1 N", "IY1 N") == 2

    def test_exact_match_four_phonemes(self):
        assert multisyllable_overlap("EY1 SH AH0 N", "EY1 SH AH0 N") == 4

    def test_partial_overlap(self):
        assert multisyllable_overlap("IY1 N", "AH0 N") == 1

    def test_no_overlap(self):
        assert multisyllable_overlap("IY1 N", "OW1 Z") == 0

    def test_none_returns_zero(self):
        assert multisyllable_overlap(None, "IY1 N") == 0
        assert multisyllable_overlap("IY1 N", None) == 0

    def test_empty_returns_zero(self):
        assert multisyllable_overlap("", "IY1 N") == 0
        assert multisyllable_overlap("IY1 N", "") == 0


class TestTokenizeLine:
    """tokenize_line splits on words."""

    def test_tokenize_simple(self):
        assert tokenize_line("hello world") == ["hello", "world"]
        assert tokenize_line("I got the flow") == ["i", "got", "the", "flow"]

    def test_tokenize_empty(self):
        assert tokenize_line("") == []


class TestSyllableCountLine:
    """syllable_count_line for full line."""

    def test_syllable_count_line(self):
        assert syllable_count_line("cat") >= 1
        assert syllable_count_line("hello world") >= 2
        assert syllable_count_line("") == 0
