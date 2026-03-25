"""
evo_rhyme/crossover.py

Crossover operators for couplet individuals. Line-level crossover: line1 from
parent A, line2 from parent B. Optionally phrase-slice if structure matches.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from evo_rhyme.individual import CoupletIndividual
from evo_rhyme.operator_telemetry import record_operator_event
from evo_rhyme.phonetics import tokenize_line, syllable_count_line


def crossover(
    parent1: CoupletIndividual,
    parent2: CoupletIndividual,
    config: Optional[Any] = None,
) -> CoupletIndividual:
    """
    Crossover two parents into a new CoupletIndividual.
    Line-level: line1 from parent A, line2 from parent B (or vice versa).
    Optionally phrase-slice if structure matches (similar syllable counts).
    """
    swap = random.random() < 0.5
    if swap:
        p_a, p_b = parent2, parent1
    else:
        p_a, p_b = parent1, parent2

    # Check if phrase-slice is viable (structure matches)
    try_phrase_slice = False
    if config and isinstance(config, dict) and config.get("phrase_slice", False):
        s1_a = syllable_count_line(p_a.line1)
        s2_a = syllable_count_line(p_a.line2)
        s1_b = syllable_count_line(p_b.line1)
        s2_b = syllable_count_line(p_b.line2)
        diff_a = abs(s1_a - s2_a)
        diff_b = abs(s1_b - s2_b)
        if diff_a <= 3 and diff_b <= 3 and abs(s1_a - s1_b) <= 4:
            try_phrase_slice = random.random() < 0.3

    if try_phrase_slice:
        child = _phrase_slice_crossover(p_a, p_b)
        if child is not None:
            record_operator_event(
                scope="couplet",
                operator_kind="crossover",
                operator_name="phrase_slice",
                succeeded=True,
            )
            return child

    # Standard line-level crossover
    line1 = p_a.line1
    line2 = p_b.line2
    record_operator_event(
        scope="couplet",
        operator_kind="crossover",
        operator_name="line_level",
        succeeded=True,
    )
    return CoupletIndividual(
        line1=line1,
        line2=line2,
        features1=None,
        features2=None,
        scores=None,
        fitness=None,
        metadata={},
    )


def _phrase_slice_crossover(
    parent_a: CoupletIndividual,
    parent_b: CoupletIndividual,
) -> Optional[CoupletIndividual]:
    """
    If structure matches, take first half of line from A and second half from B
    (or similar split) to create blended lines.
    """
    tokens1_a = tokenize_line(parent_a.line1)
    tokens2_a = tokenize_line(parent_a.line2)
    tokens1_b = tokenize_line(parent_b.line1)
    tokens2_b = tokenize_line(parent_b.line2)

    if len(tokens1_a) < 4 or len(tokens1_b) < 4:
        return None

    # Slice at similar relative position
    n1 = min(len(tokens1_a), len(tokens1_b))
    split = max(2, n1 // 2)
    if random.random() < 0.5:
        # line1: first half from A, second from B
        line1 = " ".join(tokens1_a[:split] + tokens1_b[split : split + max(0, len(tokens1_a) - split)])
        if len(line1.split()) < 3:
            return None
        line2 = parent_a.line2
    else:
        # line2: first half from A, second from B
        line2_tokens_a = tokenize_line(parent_a.line2)
        line2_tokens_b = tokenize_line(parent_b.line2)
        if len(line2_tokens_a) < 4 or len(line2_tokens_b) < 4:
            return None
        split2 = max(2, min(len(line2_tokens_a), len(line2_tokens_b)) // 2)
        line2 = " ".join(
            line2_tokens_a[:split2] + line2_tokens_b[split2 : split2 + max(0, len(line2_tokens_a) - split2)]
        )
        if len(line2.split()) < 3:
            return None
        line1 = parent_a.line1

    return CoupletIndividual(
        line1=line1,
        line2=line2,
        features1=None,
        features2=None,
        scores=None,
        fitness=None,
        metadata={},
    )
