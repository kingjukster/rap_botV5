from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .alignment import AlignmentResult, score_line_against_slots
from .grid import make_4_4_bar_grid


@dataclass(frozen=True)
class LineBeatScore:
    line: str
    score: float
    alignment: AlignmentResult


@dataclass(frozen=True)
class VerseBeatScore:
    total_score: float
    line_scores: List[LineBeatScore]
    average_score: float
    line_balance_penalty: float


def score_verse_lines(lines: List[str]) -> VerseBeatScore:
    bar = make_4_4_bar_grid()
    line_scores: List[LineBeatScore] = []

    raw_scores: List[float] = []
    for line in lines:
        alignment = score_line_against_slots(line, bar)
        raw_scores.append(alignment.score)
        line_scores.append(LineBeatScore(line=line, score=alignment.score, alignment=alignment))

    if not raw_scores:
        return VerseBeatScore(
            total_score=-10.0,
            line_scores=[],
            average_score=-10.0,
            line_balance_penalty=0.0,
        )

    avg = sum(raw_scores) / len(raw_scores)

    # Penalize wildly uneven line-to-line flow
    spread = max(raw_scores) - min(raw_scores) if len(raw_scores) > 1 else 0.0
    line_balance_penalty = 0.15 * spread

    total = sum(raw_scores) - line_balance_penalty

    return VerseBeatScore(
        total_score=total,
        line_scores=line_scores,
        average_score=avg,
        line_balance_penalty=line_balance_penalty,
    )

