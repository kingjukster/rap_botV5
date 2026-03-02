"""
rhyme_similarity.py

Weighted phoneme edit distance for rhyme comparison.
Handles similar vowels, voiced/unvoiced swaps, multi-syllable tails, and shifted alignment.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from rapbot.phoneme_converter import (
    ARPA_VOWELS,
    PhonemeSequence,
    strip_stress,
    word_to_phonemes,
)


# Similar vowel pairs: low substitution penalty
SIMILAR_VOWELS = {
    frozenset({"IH", "IY"}),
    frozenset({"AH", "UH"}),
    frozenset({"AA", "AO"}),
    frozenset({"AE", "EH"}),
    frozenset({"UW", "UH"}),
    frozenset({"ER", "AH"}),
}

# Voiced/unvoiced consonant pairs: low substitution penalty
VOICED_UNVOICED_PAIRS = {
    frozenset({"T", "D"}),
    frozenset({"S", "Z"}),
    frozenset({"F", "V"}),
    frozenset({"P", "B"}),
    frozenset({"K", "G"}),
    frozenset({"CH", "JH"}),
    frozenset({"SH", "ZH"}),
}

# Stressed vowel change: higher penalty
STRESS_VOWELS = {"AA1", "AE1", "AH1", "AO1", "AW1", "AY1", "EH1", "ER1", "EY1",
                 "IH1", "IY1", "OW1", "OY1", "UH1", "UW1",
                 "AA2", "AE2", "AH2", "AO2", "AW2", "AY2", "EH2", "ER2", "EY2",
                 "IH2", "IY2", "OW2", "OY2", "UH2", "UW2"}


def _phoneme_substitution_cost(p1: str, p2: str) -> float:
    """
    Weighted substitution cost:
    - 0: identical
    - 0.2: similar vowels (IH/IY, AH/UH)
    - 0.3: voiced/unvoiced swap (T/D, S/Z)
    - 0.8: stressed vowel change
    - 1.0: default
    """
    b1 = strip_stress(p1)
    b2 = strip_stress(p2)
    if b1 == b2:
        return 0.0
    if {b1, b2} in SIMILAR_VOWELS:
        return 0.2
    if {b1, b2} in VOICED_UNVOICED_PAIRS:
        return 0.3
    if b1 in ARPA_VOWELS and b2 in ARPA_VOWELS:
        if p1 in STRESS_VOWELS or p2 in STRESS_VOWELS:
            return 0.8
        return 0.5
    return 1.0


def _phonemes_to_list(phones: Union[Sequence[str], PhonemeSequence]) -> List[str]:
    """Normalize input to list of phoneme strings."""
    if isinstance(phones, PhonemeSequence):
        return list(phones.phonemes)
    return list(phones)


def _is_vowel(phone: str) -> bool:
    """Check if phoneme is a vowel (stripped of stress)."""
    return strip_stress(phone) in ARPA_VOWELS


def extract_stressed_nucleus(phones: Union[PhonemeSequence, list]) -> str:
    """
    Return the first phoneme of the rhyme nucleus (the stressed vowel, e.g. UW1),
    or empty string if none.

    For PhonemeSequence: use .rhyme_nucleus and take the first phoneme.
    For raw list: find last vowel to end (rhyme nucleus approximation), take first.
    """
    if isinstance(phones, PhonemeSequence):
        nucleus = phones.rhyme_nucleus
        return nucleus[0] if nucleus else ""
    # Raw list: from last vowel to end, then take first
    ph_list = list(phones)
    for i in range(len(ph_list) - 1, -1, -1):
        if _is_vowel(ph_list[i]):
            return ph_list[i]
    return ""


def _get_tail_phonemes(
    phones: Union[Sequence[str], PhonemeSequence],
    tail_syllables: int,
) -> List[str]:
    """
    Extract tail phonemes for comparison.
    tail_syllables=1: rhyme nucleus (last stressed vowel to end)
    tail_syllables=2-4: multi-syllable tail from syllable boundaries
    """
    if isinstance(phones, PhonemeSequence):
        if tail_syllables <= 1:
            return list(phones.rhyme_nucleus)
        return list(phones.rhyme_tail(tail_syllables))
    ph_list = _phonemes_to_list(phones)
    # Fallback for raw list: from last vowel to end (approximate rhyme nucleus)
    for i in range(len(ph_list) - 1, -1, -1):
        if _is_vowel(ph_list[i]):
            return ph_list[i:]
    return ph_list[-8:] if len(ph_list) > 8 else ph_list


def get_tail_with_metadata(
    phones: Union[Sequence[str], PhonemeSequence],
    tail_syllables: int = 1,
) -> Tuple[List[str], bool, int]:
    """
    Return (tail_list, has_stressed_vowel, phoneme_count).
    Uses PhonemeSequence.rhyme_nucleus; checks if first nucleus phoneme has stress 1 or 2.
    """
    tail = _get_tail_phonemes(phones, tail_syllables)
    phoneme_count = len(tail)
    has_stressed_vowel = False
    if tail:
        first = tail[0]
        if _is_vowel(first) and len(first) > 0 and first[-1] in ("1", "2"):
            has_stressed_vowel = True
    return tail, has_stressed_vowel, phoneme_count


def _split_vowel_coda(tail: List[str]) -> Tuple[List[str], List[str]]:
    """Split tail into vowel sequence and coda (consonants after last vowel)."""
    vowel_bases: List[str] = []
    coda: List[str] = []
    for p in tail:
        b = strip_stress(p)
        if _is_vowel(p):
            vowel_bases.append(b)
            coda = []  # Reset coda; we're in vowel region
        elif vowel_bases:
            coda.append(b)
    return vowel_bases, coda


def classify_rhyme_type(
    phones1: Union[Sequence[str], PhonemeSequence],
    phones2: Union[Sequence[str], PhonemeSequence],
    tail_syllables: int = 2,
    has_stressed_tail1: Optional[bool] = None,
    has_stressed_tail2: Optional[bool] = None,
) -> Optional[str]:
    """
    Classify rhyme type based on vowel vs coda match.
    Returns: "perfect" | "slant" | "assonance" | "consonance" | "weak-tail" | None
    - perfect: tail phonemes IDENTICAL (exact equality); voiced/unvoiced -> slant
    - assonance: vowel match (requires vowel match)
    - weak-tail: tail too short for the classified type (caller may skip)
    """
    (tail1, fallback_stress1, count1) = get_tail_with_metadata(phones1, tail_syllables)
    (tail2, fallback_stress2, count2) = get_tail_with_metadata(phones2, tail_syllables)
    if not tail1 or not tail2:
        return "slant"

    # Resolve stress: param > PhonemeSequence.has_stressed_rhyme_vowel > get_tail_with_metadata
    has_stress1 = (
        has_stressed_tail1
        if has_stressed_tail1 is not None
        else (
            phones1.has_stressed_rhyme_vowel
            if isinstance(phones1, PhonemeSequence)
            else fallback_stress1
        )
    )
    has_stress2 = (
        has_stressed_tail2
        if has_stressed_tail2 is not None
        else (
            phones2.has_stressed_rhyme_vowel
            if isinstance(phones2, PhonemeSequence)
            else fallback_stress2
        )
    )

    v1_list, c1 = _split_vowel_coda(tail1)
    v2_list, c2 = _split_vowel_coda(tail2)
    v1 = v1_list[-1] if v1_list else ""
    v2 = v2_list[-1] if v2_list else ""
    vowel_count1 = len(v1_list)
    vowel_count2 = len(v2_list)
    syllable_count1 = vowel_count1
    syllable_count2 = vowel_count2

    def _vowels_similar(va: str, vb: str) -> bool:
        if not va or not vb:
            return False
        if va == vb:
            return True
        return frozenset({va, vb}) in SIMILAR_VOWELS

    def _codas_match_slant(ca: List[str], cb: List[str]) -> bool:
        """Match if identical or all pairs are voiced/unvoiced swaps (for slant)."""
        if ca == cb:
            return True
        if len(ca) != len(cb):
            return False
        for a, b in zip(ca, cb):
            if a == b:
                continue
            if frozenset({a, b}) not in VOICED_UNVOICED_PAIRS:
                return False
        return True

    def _assonance_vowel_ok(n1: int, n2: int) -> bool:
        """Vowel count sufficient for assonance: both stressed allow >=1, else >=2."""
        if has_stress1 and has_stress2:
            return n1 >= 1 and n2 >= 1
        return n1 >= 2 and n2 >= 2

    # Single-vowel-only: do NOT label as perfect
    if len(tail1) == 1 and len(tail2) == 1 and _is_vowel(tail1[0]) and _is_vowel(tail2[0]):
        if _vowels_similar(v1, v2):
            if _assonance_vowel_ok(vowel_count1, vowel_count2):
                return "assonance"
            return "weak-tail"
        return "slant"

    vowels_sim = _vowels_similar(v1, v2)
    codas_match_slant = _codas_match_slant(c1, c2)

    # Perfect: STRICT - tails must be IDENTICAL (exact equality)
    if tail1 == tail2:
        if (not has_stress1 or not has_stress2) or (count1 <= 1 or count2 <= 1):
            return "weak-tail"
        return "perfect"

    # Slant: vowels similar AND codas match (incl. voiced/unvoiced)
    if vowels_sim and codas_match_slant:
        tail_weak1 = count1 < 4 and syllable_count1 < 2 and not has_stress1
        tail_weak2 = count2 < 4 and syllable_count2 < 2 and not has_stress2
        if tail_weak1 or tail_weak2:
            return "weak-tail"
        return "slant"

    # Assonance: vowel match, no/few codas
    if vowels_sim and not codas_match_slant:
        if not c1 or not c2:
            if _assonance_vowel_ok(vowel_count1, vowel_count2):
                return "assonance"
            return "weak-tail"
        return "slant"

    # Consonance: consonant match (codas identical or voiced/unvoiced)
    if not vowels_sim and (c1 == c2 or _codas_match_slant(c1, c2)):
        if len(c1) < 3 or len(c2) < 3:
            return "weak-tail"
        return "consonance"

    return "slant"


def weighted_levenshtein(a: List[str], b: List[str]) -> float:
    """
    Weighted edit distance with phoneme-aware substitution costs.
    Returns total cost (not normalized).
    """
    n, m = len(a), len(b)
    if n == 0:
        return float(m)
    if m == 0:
        return float(n)

    # dp[i][j] = min cost to transform a[:i] to b[:j]
    dp: List[List[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + 1.0
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + 1.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete = dp[i - 1][j] + 1.0
            insert = dp[i][j - 1] + 1.0
            sub_cost = _phoneme_substitution_cost(a[i - 1], b[j - 1])
            substitute = dp[i - 1][j - 1] + sub_cost
            dp[i][j] = min(delete, insert, substitute)

    return dp[n][m]


def _best_shifted_alignment(
    short: List[str],
    long_seq: List[str],
    max_shift: int = 2,
) -> Tuple[float, int]:
    """
    Find best alignment when short may align to a shifted position in long_seq.
    Returns (min_cost, best_offset). Offset is how many phones to skip from end of long.
    """
    if len(short) >= len(long_seq):
        cost = weighted_levenshtein(short[-len(long_seq):], long_seq)
        return cost, 0

    best_cost = float("inf")
    best_offset = 0
    short_len = len(short)

    for offset in range(min(max_shift, len(long_seq) - short_len + 1)):
        long_tail = long_seq[-(short_len + offset):]
        if len(long_tail) < short_len:
            continue
        # Try aligning short to different positions in long_tail
        for start in range(len(long_tail) - short_len + 1):
            sub = long_tail[start : start + short_len]
            cost = weighted_levenshtein(short, sub)
            if cost < best_cost:
                best_cost = cost
                best_offset = offset

    # Also try no shift
    long_tail = long_seq[-short_len:] if len(long_seq) >= short_len else long_seq
    cost = weighted_levenshtein(short, long_tail)
    if cost < best_cost:
        best_cost = cost
        best_offset = 0

    return best_cost, best_offset


def _vowels_similar(va: str, vb: str) -> bool:
    """Return True if vowel bases match exactly or are in SIMILAR_VOWELS."""
    if not va or not vb:
        return False
    ba = strip_stress(va)
    bb = strip_stress(vb)
    if ba == bb:
        return True
    return frozenset({ba, bb}) in SIMILAR_VOWELS


def compute_tail_similarity(
    tail1: List[str],
    tail2: List[str],
    min_tail_phonemes: int = 0,
) -> float:
    """
    Compare two tail phoneme lists directly (no re-extraction).
    Use when both inputs are already rhyme tails, e.g. from _span_tail_phonemes.
    """
    if not tail1 or not tail2:
        return 0.0

    # Guard: stressed vowel (first vowel in tail) must match
    stressed1 = next((p for p in tail1 if _is_vowel(p)), "")
    stressed2 = next((p for p in tail2 if _is_vowel(p)), "")
    if stressed1 and stressed2 and not _vowels_similar(stressed1, stressed2):
        return 0.0

    penalty = 1.0
    if min_tail_phonemes > 0:
        if len(tail1) < min_tail_phonemes or len(tail2) < min_tail_phonemes:
            penalty = 0.5

    if abs(len(tail1) - len(tail2)) <= 2:
        short_t, long_t = (tail1, tail2) if len(tail1) <= len(tail2) else (tail2, tail1)
        cost, _ = _best_shifted_alignment(short_t, long_t, max_shift=2)
    else:
        cost = weighted_levenshtein(tail1, tail2)

    max_len = max(len(tail1), len(tail2))
    max_cost = max_len * 1.0
    similarity = max(0.0, 1.0 - (cost / max_cost)) * penalty
    return round(similarity, 4)


def compute_rhyme_similarity(
    phones1: Union[Sequence[str], PhonemeSequence],
    phones2: Union[Sequence[str], PhonemeSequence],
    tail_syllables: int = 1,
    allow_shifted: bool = True,
    max_shift: int = 2,
    min_tail_phonemes: int = 0,
) -> float:
    """
    Compute rhyme similarity between two phoneme sequences.
    Returns score in [0, 1] where 1 = perfect rhyme.

    Args:
        phones1: First phoneme sequence (list or PhonemeSequence)
        phones2: Second phoneme sequence
        tail_syllables: 1-4 syllables to compare from the end
        allow_shifted: Whether to try shifted alignment (e.g. "stools in 'em" vs "Jerusalem")
        max_shift: Max phonemes to shift for alignment
        min_tail_phonemes: If > 0 and either tail has fewer phonemes, multiply similarity by 0.5

    Returns:
        Similarity score 0-1
    """
    tail1 = _get_tail_phonemes(phones1, tail_syllables)
    tail2 = _get_tail_phonemes(phones2, tail_syllables)

    if not tail1 or not tail2:
        return 0.0

    # Guard: stressed vowel (primary rhyme vowel) must match or be in SIMILAR_VOWELS.
    # Use first vowel with stress 1 or 2; if none, use last vowel (rhyme nucleus).
    # Prevents false clustering e.g. "one" (AH1 N) with "pencils" (EH1 N S AH0 L Z).
    def _first_stressed_vowel(tail: List[str]) -> str:
        # Prefer stressed (1 or 2); rhyme nucleus is typically at the primary stress
        for p in tail:
            if _is_vowel(p) and len(p) > 0 and p[-1] in ("1", "2"):
                return p
        # Fallback: last vowel (often the rhyme-bearing one)
        for i in range(len(tail) - 1, -1, -1):
            if _is_vowel(tail[i]):
                return tail[i]
        return ""
    stressed1 = _first_stressed_vowel(tail1)
    stressed2 = _first_stressed_vowel(tail2)
    if stressed1 and stressed2 and not _vowels_similar(stressed1, stressed2):
        return 0.0

    # Penalty for short tails (weak rhyme nucleus)
    penalty = 1.0
    if min_tail_phonemes > 0:
        if len(tail1) < min_tail_phonemes or len(tail2) < min_tail_phonemes:
            penalty = 0.5

    # Penalty for single-vowel-only matches (both tails are 1 phoneme and match only on vowel)
    if len(tail1) == 1 and len(tail2) == 1:
        p1, p2 = tail1[0], tail2[0]
        if _is_vowel(p1) and _is_vowel(p2):
            b1, b2 = strip_stress(p1), strip_stress(p2)
            if b1 == b2 or frozenset({b1, b2}) in SIMILAR_VOWELS:
                penalty = min(penalty, 0.5)

    if allow_shifted and abs(len(tail1) - len(tail2)) <= max_shift:
        short_t, long_t = (tail1, tail2) if len(tail1) <= len(tail2) else (tail2, tail1)
        cost, _ = _best_shifted_alignment(short_t, long_t, max_shift)
    else:
        cost = weighted_levenshtein(tail1, tail2)

    max_len = max(len(tail1), len(tail2))
    if max_len == 0:
        return 1.0
    # Normalize: 0 cost -> 1.0, max possible cost ~ max_len -> 0.0
    max_cost = max_len * 1.0
    similarity = max(0.0, 1.0 - (cost / max_cost)) * penalty
    return round(similarity, 4)


def word_rhyme_similarity(
    word1: str,
    word2: str,
    tail_syllables: int = 1,
) -> float:
    """
    Convenience: compute rhyme similarity from two words.
    Uses phoneme_converter for lookup, falls back to 0.0 if either word is OOV.
    """
    seq1 = word_to_phonemes(word1.lower())
    seq2 = word_to_phonemes(word2.lower())
    if not seq1 or not seq2:
        return 0.0
    return compute_rhyme_similarity(seq1, seq2, tail_syllables=tail_syllables)
