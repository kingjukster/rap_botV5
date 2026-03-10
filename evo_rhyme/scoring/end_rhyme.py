"""
End rhyme scoring: compare two rhyme tails.

Exact match = 1.0.
Partial = vowel_sim + consonant_sim + length bonus.
Uses PhoneticFeature and phonetic_similarity from evo_rhyme.phonetics.
"""

from __future__ import annotations

from typing import Optional

from evo_rhyme.phonetics import (
    ARPA_VOWELS,
    PhoneticFeature,
    phonetic_similarity,
    strip_stress,
    vowel_group_key,
)


def _parse_tail_to_feature(tail: str) -> Optional[PhoneticFeature]:
    """
    Parse a rhyme tail string (from last stressed vowel onward) into PhoneticFeature.
    E.g. "AY1 N AH0" -> vowel=AY, stress=1, coda=(N,) [consonants before next vowel].
    """
    if not tail or not tail.strip():
        return None
    tokens = tail.split()
    for i, t in enumerate(tokens):
        base = strip_stress(t)
        if base in ARPA_VOWELS:
            vowel = base
            stress = int(t[-1]) if t[-1].isdigit() else 0
            rest = tokens[i + 1:]
            consonants: list[str] = []
            for x in rest:
                b = strip_stress(x)
                if b in ARPA_VOWELS:
                    break
                consonants.append(b)
            return PhoneticFeature(
                vowel=vowel,
                stress=stress,
                coda=tuple(consonants),
                vowel_group=vowel_group_key(vowel),
                raw=tuple(tokens),
            )
    return None


def _length_bonus(tail1: str, tail2: str, max_bonus: float = 0.1) -> float:
    """Small bonus when tails have similar length (same syllable/phone count)."""
    n1 = len(tail1.split())
    n2 = len(tail2.split())
    if n1 == 0 or n2 == 0:
        return 0.0
    ratio = min(n1, n2) / max(n1, n2)
    return max_bonus * ratio


def score_end_rhyme(tail1: str, tail2: str) -> float:
    """
    Score end rhyme similarity between two rhyme tails.

    - Exact match = 1.0
    - Partial = vowel_sim + consonant_sim + length bonus (using PhoneticFeature/phonetic_similarity)

    Args:
        tail1: Rhyme tail from first word (e.g. from extract_rhyme_tail)
        tail2: Rhyme tail from second word

    Returns:
        Float in [0, 1]; 1.0 = perfect rhyme.
    """
    if not tail1 or not tail2:
        return 0.0
    tail1 = tail1.strip()
    tail2 = tail2.strip()
    if tail1 == tail2:
        return 1.0

    f1 = _parse_tail_to_feature(tail1)
    f2 = _parse_tail_to_feature(tail2)
    if not f1 or not f2:
        return 0.0

    base_score = phonetic_similarity(f1, f2)
    bonus = _length_bonus(tail1, tail2)
    return min(1.0, base_score + bonus)
