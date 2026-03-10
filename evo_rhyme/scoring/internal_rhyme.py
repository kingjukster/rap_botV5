"""
Internal rhyme scoring for paired lines.

Compares rhyme tails within each line and across the paired lines.
Only counts pairs with stressed vowel in tail, not identical repeat, not too close.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import pronouncing

from evo_rhyme.syllables import count_syllables_line

# Use DEFAULT_STOPWORDS from scripts.tools.update_rhyme_groups when available,
# otherwise use local copy to avoid heavy imports (torch, etc.)
DEFAULT_STOPWORDS: Set[str] = {
    "a", "an", "and", "at", "be", "but", "by", "da", "for", "from",
    "go", "had", "he", "her", "him", "his", "i", "id", "im", "in",
    "is", "it", "me", "my", "no", "of", "on", "or", "our", "out",
    "she", "so", "that", "the", "them", "then", "there", "they",
    "this", "to", "was", "we", "ya", "yo", "you", "ya'll",
}


WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class LineFeatures:
    """Precomputed features for a line: content words, rhyme tails, syllable count."""

    text: str
    content_words: List[str]
    rhyme_tails: List[Tuple[str, Optional[str], int]]  # (word, tail, position)
    syllable_count: int


def _rhyme_tail(word: str) -> Optional[str]:
    """Extract rhyme tail (rhyming_part) from word using pronouncing."""
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return None
    part = pronouncing.rhyming_part(phones[0])
    return part if part else None


ARPA_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
}


def _has_stressed_vowel(word: str) -> bool:
    """Check if word has stressed vowel in its last syllable."""
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return False
    tokens = phones[0].split()
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        base = re.sub(r"\d", "", t.upper())
        if base in ARPA_VOWELS:
            return t[-1].isdigit() and int(t[-1]) > 0
    return False


def build_line_features(
    line: str,
    stopwords: Optional[Set[str]] = None,
    min_len: int = 2,
) -> LineFeatures:
    """Build LineFeatures from a line string."""
    stopwords = stopwords or DEFAULT_STOPWORDS
    cleaned = re.sub(r"\[[^\]]+\]", " ", line)
    tokens = WORD_RE.findall(cleaned.lower())
    content_words: List[str] = []
    rhyme_tails: List[Tuple[str, Optional[str], int]] = []
    for i, w in enumerate(tokens):
        clean = re.sub(r"[^a-z']", "", w).strip("'")
        if not clean or len(clean) < min_len or clean in stopwords:
            continue
        content_words.append(clean)
        tail = _rhyme_tail(clean)
        rhyme_tails.append((clean, tail, i))
    syl = count_syllables_line(line)
    return LineFeatures(
        text=line,
        content_words=content_words,
        rhyme_tails=rhyme_tails,
        syllable_count=syl,
    )


def _tails_match(t1: Optional[str], t2: Optional[str]) -> bool:
    """Check if two rhyme tails match (for internal rhyme)."""
    if not t1 or not t2:
        return False
    return t1 == t2


def _count_valid_pairs(
    entries1: List[Tuple[str, Optional[str], int]],
    entries2: List[Tuple[str, Optional[str], int]],
    min_distance: int = 2,
) -> int:
    """Count rhyme pairs that satisfy: stressed vowel, not identical, not too close."""
    count = 0
    # Within line 1
    for i, (w1, t1, pos1) in enumerate(entries1):
        if not _has_stressed_vowel(w1) or not t1:
            continue
        for j, (w2, t2, pos2) in enumerate(entries1):
            if i >= j:
                continue
            if not _has_stressed_vowel(w2) or not t2:
                continue
            if w1 == w2:  # identical repeat
                continue
            if abs(pos1 - pos2) < min_distance:  # too close
                continue
            if _tails_match(t1, t2):
                count += 1
    # Within line 2
    for i, (w1, t1, pos1) in enumerate(entries2):
        if not _has_stressed_vowel(w1) or not t1:
            continue
        for j, (w2, t2, pos2) in enumerate(entries2):
            if i >= j:
                continue
            if not _has_stressed_vowel(w2) or not t2:
                continue
            if w1 == w2:
                continue
            if abs(pos1 - pos2) < min_distance:
                continue
            if _tails_match(t1, t2):
                count += 1
    # Across lines
    for (w1, t1, _) in entries1:
        if not _has_stressed_vowel(w1) or not t1:
            continue
        for (w2, t2, _) in entries2:
            if not _has_stressed_vowel(w2) or not t2:
                continue
            if w1 == w2:
                continue
            if _tails_match(t1, t2):
                count += 1
    return count


def score_internal_rhyme(
    features1: LineFeatures,
    features2: LineFeatures,
    min_distance: int = 2,
    max_density: float = 0.35,
) -> float:
    """
    Score internal rhyme density for a pair of lines.

    Returns density (rhyme pairs per syllable), capped at max_density.
    """
    total_syllables = features1.syllable_count + features2.syllable_count
    if total_syllables <= 0:
        return 0.0
    pairs = _count_valid_pairs(
        features1.rhyme_tails,
        features2.rhyme_tails,
        min_distance=min_distance,
    )
    density = pairs / total_syllables
    return min(density, max_density)
