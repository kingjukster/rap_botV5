"""
Pairwise accuracy with tie_policy v1 (epsilon threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Predicted = Literal["a", "b", "tie"]


def predict_winner(score_a: float, score_b: float, epsilon: float = 0.05) -> Predicted:
    d = score_a - score_b
    if d > epsilon:
        return "a"
    if -d > epsilon:
        return "b"
    return "tie"


def full_matrix_score(human: str, pred: Predicted) -> float:
    """v1: no partial credit."""
    h = human.lower()
    if h == "tie":
        return 1.0 if pred == "tie" else 0.0
    if pred == "tie":
        return 0.0
    if h == "a" and pred == "a":
        return 1.0
    if h == "b" and pred == "b":
        return 1.0
    return 0.0


def directional_aux_correct(human: str, pred: Predicted) -> Optional[bool]:
    """
    Only defined when human in (a, b). pred_tie -> False.
    Returns None if human is tie (exclude from directional denominator).
    """
    h = human.lower()
    if h == "tie":
        return None
    if pred == "tie":
        return False
    return (h == "a" and pred == "a") or (h == "b" and pred == "b")


@dataclass
class PairwiseEvalSummary:
    full_matrix_mean: float
    full_matrix_n: int
    directional_mean: float
    directional_n: int
    epsilon: float


def summarize_pairwise(
    judgments: list[Tuple[str, float, float]],
    *,
    epsilon: float = 0.05,
) -> PairwiseEvalSummary:
    """
    judgments: list of (human_better, score_a, score_b) with human in a|b|tie.
    """
    full_scores: list[float] = []
    dir_hits: list[float] = []
    for human, sa, sb in judgments:
        pred = predict_winner(sa, sb, epsilon=epsilon)
        full_scores.append(full_matrix_score(human, pred))
        aux = directional_aux_correct(human, pred)
        if aux is not None:
            dir_hits.append(1.0 if aux else 0.0)
    return PairwiseEvalSummary(
        full_matrix_mean=float(sum(full_scores) / len(full_scores)) if full_scores else 0.0,
        full_matrix_n=len(full_scores),
        directional_mean=float(sum(dir_hits) / len(dir_hits)) if dir_hits else 0.0,
        directional_n=len(dir_hits),
        epsilon=epsilon,
    )
