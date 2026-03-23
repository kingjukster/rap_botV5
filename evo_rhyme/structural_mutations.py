"""
Structural mutations for hierarchical verse evolution.

Operators that act on verse structure (e.g., swap couplets, rewrite transitions)
rather than on individual words. Used at 4-bar and higher levels.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from evo_rhyme.individual import VerseIndividual
from evo_rhyme.verse_builder import _optimize_transition

logger = logging.getLogger(__name__)


def swap_couplets(individual: VerseIndividual) -> VerseIndividual:
    """Swap the two couplets (lines 1-2 with lines 3-4) in a 4-bar verse."""
    lines = individual.lines
    if len(lines) < 4:
        return individual
    new_lines = list(lines[2:4]) + list(lines[0:2])
    return VerseIndividual(
        lines=new_lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={**individual.metadata, "mutation": "swap_couplets"},
    )


def rewrite_transition_line(
    individual: VerseIndividual,
    boundary_idx: int = 1,
    lm_budget: Optional[Dict[str, int]] = None,
) -> Optional[VerseIndividual]:
    """Rewrite the line at the couplet boundary to improve coherence.

    boundary_idx=1 rewrites line 2 (transition from couplet 1 to couplet 2).
    boundary_idx=3 would rewrite line 4 (end of verse).
    Uses LM paraphrase while preserving rhyme.
    """
    if lm_budget is None or lm_budget.get("remaining", 0) <= 0:
        return None
    lines = individual.lines
    if len(lines) < 4 or boundary_idx < 0 or boundary_idx >= len(lines):
        return None

    new_lines = _optimize_transition(lines, boundary_idx, lm_budget)
    if new_lines == lines:
        return None

    return VerseIndividual(
        lines=new_lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={**individual.metadata, "mutation": "rewrite_transition"},
    )


def couplet_swap_crossover(
    parent1: VerseIndividual,
    parent2: VerseIndividual,
) -> VerseIndividual:
    """Take couplet 1 from parent1, couplet 2 from parent2 (or vice versa)."""
    if len(parent1.lines) < 4 or len(parent2.lines) < 4:
        return VerseIndividual(lines=list(parent1.lines), metadata={})
    if random.random() < 0.5:
        lines = parent1.lines[:2] + parent2.lines[2:4]
    else:
        lines = parent2.lines[:2] + parent1.lines[2:4]
    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={"origin": "couplet_swap_crossover"},
    )
