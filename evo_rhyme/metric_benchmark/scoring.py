"""Derived scores for benchmark evaluation (fluency composite, overall composite)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from evo_rhyme.experiment_metrics import fitness_vector_from_scores


def fluency_composite_from_scores(scores: Optional[Dict[str, float]]) -> float:
    """Mean of available fluency-like keys from score_verse."""
    if not scores:
        return 0.5
    keys = ("fluency", "lm_fluency", "lexical_validity")
    vals = []
    for k in keys:
        v = scores.get(k)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return 0.5
    return max(0.0, min(1.0, sum(vals) / len(vals)))


def overall_composite_from_scores(
    scores: Optional[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Weighted combination of fitness_vector + fluency_composite for pairwise vs labels.overall.
    Default weights from protocol overall_weights_v1.
    """
    if weights is None:
        weights = {
            "flow": 0.2,
            "rhyme": 0.2,
            "semantic": 0.2,
            "punchline": 0.15,
            "novelty": 0.15,
            "fluency_composite": 0.1,
        }
    fv = fitness_vector_from_scores(scores)
    flu = fluency_composite_from_scores(scores)
    total = 0.0
    wsum = 0.0
    for k, w in weights.items():
        if k == "fluency_composite":
            total += w * flu
        else:
            total += w * float(fv.get(k, 0.0))
        wsum += w
    if wsum <= 0:
        return 0.5
    return max(0.0, min(1.0, total / wsum))


def axis_scalar_for_pairwise(
    scores: Optional[Dict[str, float]],
    axis: str,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Scalar to compare for pairwise evaluation along axis."""
    if axis == "overall":
        return overall_composite_from_scores(scores, weights=weights)
    fv = fitness_vector_from_scores(scores)
    if axis == "flow":
        return float(fv.get("flow", 0.5))
    if axis == "rhyme":
        return float(fv.get("rhyme", 0.5))
    if axis == "semantic":
        return float(fv.get("semantic", 0.5))
    if axis == "punchline":
        return float(fv.get("punchline", 0.5))
    return overall_composite_from_scores(scores, weights=weights)
