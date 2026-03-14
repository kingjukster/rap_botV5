"""
evo_rhyme/flow.py

Flow scoring: stress-pattern similarity between lines and a template.
Uses evo_rhyme.individual for phonetic analysis.
"""

from __future__ import annotations

import re
from typing import List, Optional

from evo_rhyme.individual import VerseFeatures, _analyze_line

DEFAULT_FLOW_TEMPLATE = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


def extract_stress_pattern(line: str) -> List[int]:
    """
    Build a binarized stress vector from a line.

    Uses evo_rhyme.individual._analyze_line to get LineFeatures, then
    binarizes features.stress_pattern: 0->0, 1->1, 2->1 (stressed =
    primary or secondary).

    Args:
        line: The text line to analyze.

    Returns:
        List of 0/1 values (0=unstressed, 1=stressed).
    """
    features = _analyze_line(line)
    return [0 if s == 0 else 1 for s in features.stress_pattern]


def compare_pattern(pattern: List[int], template: List[int]) -> float:
    """
    Compute similarity [0, 1] between a line pattern and a template.

    Uses truncate/pad: takes min length, counts matches, returns
    matches / min_len. If either list is empty, returns 0.5.

    Args:
        pattern: Binarized stress pattern from a line.
        template: Target stress template.

    Returns:
        Similarity score in [0, 1].
    """
    if not pattern or not template:
        return 0.5
    min_len = min(len(pattern), len(template))
    matches = sum(1 for i in range(min_len) if pattern[i] == template[i])
    return matches / min_len


def score_flow(line: str, template: Optional[List[int]] = None) -> float:
    """
    Score how well a line's stress pattern matches a flow template.

    If template is None, uses DEFAULT_FLOW_TEMPLATE.

    Args:
        line: The text line to score.
        template: Optional stress template. Defaults to [0,1,0,1,...].

    Returns:
        Flow score in [0, 1].
    """
    if template is None:
        template = DEFAULT_FLOW_TEMPLATE
    pattern = extract_stress_pattern(line)
    return compare_pattern(pattern, template)


def score_verse_flow(
    lines: List[str],
    template: Optional[List[int]] = None,
    features: Optional[VerseFeatures] = None,
) -> float:
    """
    Score flow for a verse (multiple lines) against a template.

    If features and features.stress_patterns exist, uses those directly
    (binarized: 0->0, 1->1, 2->1) to avoid re-analysis. Otherwise
    extracts per line. Returns mean flow score across lines. Handles
    verses with fewer than 4 lines.

    Args:
        lines: List of text lines in the verse.
        template: Optional stress template. Defaults to DEFAULT_FLOW_TEMPLATE.
        features: Optional pre-computed VerseFeatures to avoid re-analysis.

    Returns:
        Mean flow score in [0, 1].
    """
    if template is None:
        template = DEFAULT_FLOW_TEMPLATE
    if not lines:
        return 0.5

    if features and features.stress_patterns:
        patterns = [
            [0 if s == 0 else 1 for s in sp]
            for sp in features.stress_patterns
        ]
    else:
        patterns = [extract_stress_pattern(line) for line in lines]

    scores = [compare_pattern(p, template) for p in patterns]
    return sum(scores) / len(scores)


_UNSTRESSED_WORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "out", "and", "or", "but", "if", "when", "while", "as",
    "till", "that", "this", "no", "my", "your", "our", "his", "her", "its",
    "their", "where", "how", "why", "like", "than", "so", "yet", "nor",
})

_STRESSED_WORDS = frozenset({
    "I", "now", "ain't", "got", "keep", "start", "still", "won't", "don't",
    "can't", "hit", "feel", "see", "know", "said", "both", "never", "every",
})

_PLACEHOLDER_RE_FLOW = re.compile(r"\{(\w+)\}")


def implied_stress_pattern(template: str) -> List[int]:
    """Compute the implied stress pattern of a template before word filling.

    Function words (determiners, prepositions, conjunctions) are unstressed (0).
    Placeholders are assumed stressed (1) since they'll be filled with content words.
    Known stressed words get 1.
    Unknown literal words default to 1 (assumed content).

    Args:
        template: Template string like "I {verb_past} through the {noun}"

    Returns:
        List of 0/1 stress values.
    """
    pattern: List[int] = []
    tokens = template.split()
    for token in tokens:
        if _PLACEHOLDER_RE_FLOW.match(token):
            pattern.append(1)
        elif token.lower().rstrip(",.!?;:") in _UNSTRESSED_WORDS:
            pattern.append(0)
        elif token.rstrip(",.!?;:") in _STRESSED_WORDS:
            pattern.append(1)
        else:
            pattern.append(1)  # unknown literal = content word = stressed
    return pattern


def template_flow_score(template: str, target: Optional[List[int]] = None) -> float:
    """Score how well a template's implied stress matches the target flow.

    Args:
        template: Template string.
        target: Target stress pattern. Defaults to DEFAULT_FLOW_TEMPLATE.

    Returns:
        Similarity score in [0, 1].
    """
    if target is None:
        target = DEFAULT_FLOW_TEMPLATE
    pattern = implied_stress_pattern(template)
    return compare_pattern(pattern, target)
