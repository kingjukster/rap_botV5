"""
Structured verse fitness breakdown (parity with ``compute_verse_fitness``).

Use for tooling and audits without changing scalar selection behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from evo_rhyme.fitness import VERSE_DEFAULT_WEIGHTS, compute_verse_fitness


@dataclass
class VerseFitnessBreakdown:
    """Weighted contributions per key; ``aggregate`` matches ``compute_verse_fitness``."""

    aggregate: float
    terms: Dict[str, float] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)


def compute_verse_fitness_breakdown(
    scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> VerseFitnessBreakdown:
    """Mirror ``compute_verse_fitness`` logic with per-key term storage."""
    w = weights or VERSE_DEFAULT_WEIGHTS
    terms: Dict[str, float] = {}
    for key, weight in w.items():
        if key not in scores:
            continue
        if key == "cliche_penalty":
            cliche_val = scores.get("cliche_penalty", 0.0)
            cliche_scaled = min(1.0, cliche_val * 80)
            contrib = weight * cliche_scaled
            terms[key] = contrib
        else:
            contrib = weight * scores[key]
            terms[key] = contrib
    agg = compute_verse_fitness(scores, weights=w)
    return VerseFitnessBreakdown(
        aggregate=float(agg),
        terms=terms,
        weights_used=dict(w),
    )
