"""
evo_rhyme/individual.py

Data model for evolutionary rhyme individuals: LineFeatures and CoupletIndividual.
Populates features from phonetics for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evo_rhyme.phonetics import (
    extract_last_syllable,
    phones_for_word,
    stress_pattern_from_phones,
    syllable_count_line,
    tokenize_line,
)


@dataclass
class LineFeatures:
    """Phonetic and structural features for a single line."""
    text: str
    tokens: List[str]
    phonemes: List[str]  # ARPA phoneme strings per word (first pronunciation)
    syllable_count: int
    stress_pattern: List[int]  # 0=unstressed, 1=primary, 2=secondary per vowel
    end_tail: Optional[Any]  # PhoneticFeature for last word
    internal_tails: List[Optional[Any]]  # PhoneticFeature per content word


@dataclass
class CoupletIndividual:
    """A couplet (two lines) with features and scores."""
    line1: str
    line2: str
    features1: Optional[LineFeatures] = None
    features2: Optional[LineFeatures] = None
    scores: Optional[Dict[str, float]] = None
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerseFeatures:
    """Phonetic and structural features for a 4-line verse."""
    tokens_per_line: List[List[str]]
    phonemes: List[List[str]]
    syllable_counts: List[int]
    end_tails: List[Optional[Any]]
    stress_patterns: List[List[int]]


@dataclass
class VerseIndividual:
    """A verse (4 lines) with features and scores."""
    lines: List[str]
    features: Optional[VerseFeatures] = None
    scores: Optional[Dict[str, float]] = None
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _analyze_line(text: str) -> LineFeatures:
    """Build LineFeatures for a single line from phonetics."""
    tokens = tokenize_line(text)
    phonemes: List[str] = []
    all_stress: List[int] = []
    internal_tails: List[Optional[Any]] = []

    for word in tokens:
        phones_list = phones_for_word(word)
        if phones_list:
            ph = phones_list[0]
            phonemes.append(ph)
            stress = stress_pattern_from_phones(ph)
            all_stress.extend(stress)
            feat = extract_last_syllable(word)
            internal_tails.append(feat)
        else:
            phonemes.append("")
            internal_tails.append(None)

    syllable_count = syllable_count_line(text)
    end_tail = extract_last_syllable(tokens[-1]) if tokens else None

    return LineFeatures(
        text=text,
        tokens=tokens,
        phonemes=phonemes,
        syllable_count=syllable_count,
        stress_pattern=all_stress,
        end_tail=end_tail,
        internal_tails=internal_tails,
    )


def analyze_individual(individual: CoupletIndividual) -> CoupletIndividual:
    """
    Populate features1 and features2 from phonetics.
    Mutates the individual in place and returns it.
    """
    individual.features1 = _analyze_line(individual.line1)
    individual.features2 = _analyze_line(individual.line2)
    return individual


def analyze_verse_individual(individual: VerseIndividual) -> VerseIndividual:
    """
    Populate features from phonetics for all 4 lines.
    Mutates the individual in place and returns it.
    """
    if len(individual.lines) != 4:
        return individual
    line_features = [_analyze_line(line) for line in individual.lines]
    individual.features = VerseFeatures(
        tokens_per_line=[lf.tokens for lf in line_features],
        phonemes=[lf.phonemes for lf in line_features],
        syllable_counts=[lf.syllable_count for lf in line_features],
        end_tails=[lf.end_tail for lf in line_features],
        stress_patterns=[lf.stress_pattern for lf in line_features],
    )
    return individual
