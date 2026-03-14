"""
Rhyme chain scoring: multi-syllable phoneme cluster detection across lines.

Detects repeated phoneme clusters (3+ phones) across word positions and lines,
scoring density, cross-line coverage, and internal placement. Complements the
existing end-rhyme and internal-rhyme scoring with deeper phonemic analysis.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from evo_rhyme.phonetics import (
    get_pronunciations,
    strip_stress,
    tokenize_line,
    ARPA_VOWELS,
)


@dataclass
class PhonemeCluster:
    """A multi-syllable phoneme segment repeated across words/lines."""
    phones: Tuple[str, ...]
    words: List[str] = field(default_factory=list)
    positions: List[Tuple[int, int]] = field(default_factory=list)  # (line_idx, word_idx)

    @property
    def length(self) -> int:
        return len(self.phones)

    @property
    def cross_line(self) -> bool:
        line_indices = {pos[0] for pos in self.positions}
        return len(line_indices) > 1

    @property
    def has_internal(self) -> bool:
        """True if any occurrence is at a non-end position."""
        return any(True for _, widx in self.positions if widx >= 0)


def _word_to_phones(word: str) -> List[str]:
    """Get stress-stripped phoneme list for a word."""
    prons = get_pronunciations(word.lower())
    if not prons:
        return []
    return [strip_stress(p) for p in prons[0].split()]


def _extract_substrings(phones: List[str], min_len: int, max_len: int) -> List[Tuple[str, ...]]:
    """Extract all phoneme substrings of length [min_len, max_len] that contain at least one vowel."""
    results: List[Tuple[str, ...]] = []
    for n in range(min_len, min(max_len + 1, len(phones) + 1)):
        for i in range(len(phones) - n + 1):
            sub = tuple(phones[i : i + n])
            if any(p in ARPA_VOWELS for p in sub):
                results.append(sub)
    return results


def extract_phoneme_clusters(
    lines: List[str],
    min_phones: int = 3,
    max_phones: int = 6,
) -> List[PhonemeCluster]:
    """Find repeated multi-syllable phoneme clusters across all lines.

    For each word in each line, extracts phoneme substrings of length
    [min_phones, max_phones]. Clusters appearing in 2+ different words
    are returned with their positions.
    """
    cluster_map: Dict[Tuple[str, ...], PhonemeCluster] = {}
    seen_word_clusters: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

    for line_idx, line in enumerate(lines):
        tokens = tokenize_line(line)
        for word_idx, word in enumerate(tokens):
            phones = _word_to_phones(word)
            if len(phones) < min_phones:
                continue
            subs = _extract_substrings(phones, min_phones, max_phones)
            for sub in subs:
                is_end = (word_idx == len(tokens) - 1)
                pos_word_idx = -1 if is_end else word_idx

                if sub not in cluster_map:
                    cluster_map[sub] = PhonemeCluster(phones=sub)
                cluster = cluster_map[sub]

                if word.lower() not in seen_word_clusters[sub]:
                    seen_word_clusters[sub].add(word.lower())
                    cluster.words.append(word.lower())
                    cluster.positions.append((line_idx, pos_word_idx))

    return [c for c in cluster_map.values() if len(c.words) >= 2]


def score_rhyme_chain_density(
    lines: List[str],
    min_phones: int = 3,
) -> float:
    """Score [0,1] based on multi-syllable chain richness.

    Rewards:
    - Longer clusters (4+ phones weighted higher than 3)
    - Cross-line matches (chains spanning 2+ lines)
    - More occurrences per cluster
    - Internal positions (not just end words)
    """
    clusters = extract_phoneme_clusters(lines, min_phones=min_phones)
    if not clusters:
        return 0.0

    total_words = sum(len(tokenize_line(line)) for line in lines)
    if total_words < 2:
        return 0.0

    score = 0.0
    for cluster in clusters:
        length_bonus = 1.0 + 0.5 * max(0, cluster.length - 3)
        occurrence_bonus = min(2.0, len(cluster.words) * 0.5)
        cross_line_bonus = 1.5 if cluster.cross_line else 1.0
        internal_bonus = 1.3 if cluster.has_internal else 1.0
        score += length_bonus * occurrence_bonus * cross_line_bonus * internal_bonus

    max_possible = total_words * 2.0
    normalized = min(1.0, score / max(1.0, max_possible))
    return normalized


def score_chain_structure(
    lines: List[str],
) -> Dict[str, float]:
    """Decomposed chain scores for detailed feedback.

    Returns:
        chain_density: overall multi-syllable match density [0,1]
        cross_line_chains: fraction of chains spanning 2+ lines [0,1]
        internal_chain_density: fraction of chains at non-end positions [0,1]
        max_chain_length: longest phoneme cluster found (raw count)
    """
    clusters = extract_phoneme_clusters(lines)

    if not clusters:
        return {
            "chain_density": 0.0,
            "cross_line_chains": 0.0,
            "internal_chain_density": 0.0,
            "max_chain_length": 0.0,
        }

    density = score_rhyme_chain_density(lines)
    cross_line_count = sum(1 for c in clusters if c.cross_line)
    internal_count = sum(1 for c in clusters if c.has_internal)
    max_length = max(c.length for c in clusters)

    return {
        "chain_density": density,
        "cross_line_chains": cross_line_count / len(clusters),
        "internal_chain_density": internal_count / len(clusters),
        "max_chain_length": float(max_length),
    }


def score_line_chain_potential(line: str, min_phones: int = 3) -> float:
    """Score a single line's potential for rhyme chain matching.

    Counts how many distinct phoneme clusters of length >= min_phones
    the line's words contain. Higher = more chain-matchable words.
    """
    tokens = tokenize_line(line)
    cluster_set: Set[Tuple[str, ...]] = set()
    for word in tokens:
        phones = _word_to_phones(word)
        if len(phones) < min_phones:
            continue
        subs = _extract_substrings(phones, min_phones, 6)
        cluster_set.update(subs)
    if not tokens:
        return 0.0
    return min(1.0, len(cluster_set) / max(1, len(tokens) * 3))
