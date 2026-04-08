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
    template_id1: Optional[str] = None  # template used for line1
    template_id2: Optional[str] = None  # template used for line2
    scores: Optional[Dict[str, float]] = None
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerseFeatures:
    """Phonetic and structural features for an N-line verse."""
    tokens_per_line: List[List[str]]
    phonemes: List[List[str]]
    syllable_counts: List[int]
    end_tails: List[Optional[Any]]
    stress_patterns: List[List[int]]


@dataclass
class VerseStructure:
    """Structural metadata for a verse, evolved alongside content."""
    scheme: str = "AABB"
    roles: List[str] = field(default_factory=list)
    callbacks: List[Optional[int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.roles and self.scheme:
            n = len(self.scheme)
            defaults = {
                4: ["setup", "flex", "flex", "punchline"],
                8: ["setup", "flex", "threat", "flex", "setup", "flex", "threat", "punchline"],
                16: ["setup", "flex", "threat", "flex"] * 3 + ["setup", "flex", "introspection", "punchline"],
            }
            self.roles = defaults.get(n, ["flex"] * n)
        if not self.callbacks:
            self.callbacks = [None] * len(self.roles)

    @classmethod
    def for_scheme(cls, scheme: str, num_lines: int = 4) -> "VerseStructure":
        """Create a VerseStructure for a given scheme and line count."""
        full_scheme = scheme
        if len(scheme) < num_lines:
            repeats = (num_lines + len(scheme) - 1) // len(scheme)
            full_scheme = (scheme * repeats)[:num_lines]
        return cls(scheme=full_scheme)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "roles": self.roles,
            "callbacks": self.callbacks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerseStructure":
        return cls(
            scheme=data.get("scheme", "AABB"),
            roles=data.get("roles", []),
            callbacks=data.get("callbacks", []),
        )


@dataclass
class VerseIndividual:
    """A verse (N lines, typically 4 in evolution) with features and scores."""
    lines: List[str]
    structure: Optional[VerseStructure] = None
    features: Optional[VerseFeatures] = None
    template_ids: Optional[List[Optional[str]]] = None  # template used per line
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
    Populate features from phonetics for all lines (any length >= 1).
    Mutates the individual in place and returns it.
    """
    if not individual.lines:
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


def create_verse_individual(
    lines: List[str],
    scheme: str = "AABB",
    roles: Optional[List[str]] = None,
) -> VerseIndividual:
    """Create a VerseIndividual with structure metadata."""
    structure = VerseStructure.for_scheme(scheme, num_lines=len(lines))
    if roles:
        structure.roles = roles
    if not structure.callbacks:
        structure.callbacks = [None] * len(lines)
    ind = VerseIndividual(
        lines=lines,
        structure=structure,
    )
    ind.metadata["scheme"] = scheme
    return ind
