"""
Structured verse fitness breakdown (parity with ``compute_verse_fitness``).

Use for tooling and audits without changing scalar selection behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evo_rhyme.fitness import VERSE_DEFAULT_WEIGHTS, compute_verse_fitness

logger = logging.getLogger(__name__)


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


@dataclass
class PopulationFitnessAudit:
    """Per-component statistics across a population sample."""

    sample_size: int = 0
    mean_fitness: float = 0.0
    component_means: Dict[str, float] = field(default_factory=dict)
    component_stds: Dict[str, float] = field(default_factory=dict)
    contribution_means: Dict[str, float] = field(default_factory=dict)
    zero_rate: Dict[str, float] = field(default_factory=dict)
    near_cap_count: int = 0


def audit_population_fitness(
    population: List[Any],
    weights: Optional[Dict[str, float]] = None,
    log_results: bool = True,
) -> PopulationFitnessAudit:
    """Compute per-component contribution stats across a population.

    Logs a summary table when log_results=True. Identifies components with
    near-zero variance (potential waste), anti-correlated pairs, and
    components contributing disproportionately to final fitness.
    """
    w = weights or VERSE_DEFAULT_WEIGHTS
    scored = [ind for ind in population if ind.scores]
    if not scored:
        return PopulationFitnessAudit()

    n = len(scored)
    keys = sorted(w.keys())

    raw_vals: Dict[str, List[float]] = {k: [] for k in keys}
    contrib_vals: Dict[str, List[float]] = {k: [] for k in keys}
    fitnesses: List[float] = []

    for ind in scored:
        f = ind.fitness or 0.0
        fitnesses.append(f)
        for k in keys:
            raw = ind.scores.get(k, 0.0)
            if k == "cliche_penalty":
                raw = min(1.0, raw * 80)
            raw_vals[k].append(raw)
            contrib_vals[k].append(w[k] * raw)

    mean_fit = sum(fitnesses) / n
    near_cap = sum(1 for f in fitnesses if f >= 0.93)

    comp_means: Dict[str, float] = {}
    comp_stds: Dict[str, float] = {}
    contr_means: Dict[str, float] = {}
    zero_rates: Dict[str, float] = {}

    for k in keys:
        vals = raw_vals[k]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(1, n)
        std = var ** 0.5
        comp_means[k] = mean
        comp_stds[k] = std
        contr_means[k] = sum(contrib_vals[k]) / n
        zero_rates[k] = sum(1 for v in vals if abs(v) < 1e-6) / n

    audit = PopulationFitnessAudit(
        sample_size=n,
        mean_fitness=mean_fit,
        component_means=comp_means,
        component_stds=comp_stds,
        contribution_means=contr_means,
        zero_rate=zero_rates,
        near_cap_count=near_cap,
    )

    if log_results:
        sorted_by_contrib = sorted(
            contr_means.items(), key=lambda x: abs(x[1]), reverse=True,
        )
        logger.info(
            "Fitness audit: n=%d mean=%.4f near_cap=%d", n, mean_fit, near_cap,
        )
        for k, c in sorted_by_contrib[:15]:
            logger.info(
                "  %-35s  weight=%+.3f  raw_mean=%.3f  raw_std=%.3f  contrib=%.4f  zero_rate=%.0f%%",
                k, w.get(k, 0.0), comp_means[k], comp_stds[k], c, zero_rates[k] * 100,
            )
        low_var = [k for k in keys if comp_stds[k] < 0.01 and abs(w.get(k, 0)) > 0.01]
        if low_var:
            logger.info(
                "  Low-variance components (std<0.01, possible dead weight): %s",
                ", ".join(low_var),
            )

    return audit
