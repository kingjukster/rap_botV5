"""
Line-level penalty for banned or discouraged phrases.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set


def _load_penalty_phrases() -> Set[str]:
    """Load penalty phrases from data/evo_rhyme/penalty_phrases.txt."""
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "evo_rhyme" / "penalty_phrases.txt"
    if not path.exists():
        return set()
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().lower()
            if p and not p.startswith("#"):
                out.add(p)
    return out


DEFAULT_PENALTY_PHRASES = _load_penalty_phrases()


def score_line_penalty(
    line: str,
    penalty_phrases: Set[str] | None = None,
) -> float:
    """
    Penalty for lines containing banned or discouraged phrases.

    Returns a penalty in [0, 1]; 0 = no penalty, 1 = full penalty.
    """
    penalty_phrases = penalty_phrases or DEFAULT_PENALTY_PHRASES
    if not penalty_phrases:
        return 0.0
    lower = line.lower()
    for phrase in penalty_phrases:
        if phrase in lower:
            return 1.0
    return 0.0


def apply_line_penalty(
    lines: List[str],
    penalty_phrases: Set[str] | None = None,
) -> float:
    """
    Aggregate penalty across multiple lines.

    Returns the fraction of lines that contain a penalty phrase.
    """
    penalty_phrases = penalty_phrases or DEFAULT_PENALTY_PHRASES
    if not lines or not penalty_phrases:
        return 0.0
    hit_count = sum(
        1 for ln in lines
        if score_line_penalty(ln, penalty_phrases) > 0
    )
    return hit_count / len(lines)
