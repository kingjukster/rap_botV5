"""
Syllable balance scoring for paired lines.

Target 10-14 syllables per line; ideal pair difference <= 1.
"""

from __future__ import annotations


def score_syllable_balance(
    n1: int,
    n2: int,
    target_min: int = 10,
    target_max: int = 14,
) -> float:
    """
    Score how well a pair of syllable counts fits the target range and balance.

    - Target: each line 10-14 syllables
    - Ideal: pair difference <= 1

    Returns a score in [0, 1]; 1 = perfect fit.
    """
    # In-range score: both lines in [target_min, target_max]
    in_range_1 = 1.0 if target_min <= n1 <= target_max else 0.0
    in_range_2 = 1.0 if target_min <= n2 <= target_max else 0.0
    range_score = (in_range_1 + in_range_2) / 2.0

    # Balance score: pair difference <= 1 is ideal
    diff = abs(n1 - n2)
    if diff <= 1:
        balance_score = 1.0
    else:
        # Decay as difference grows (e.g. diff 2 -> 0.5, diff 4 -> 0.25)
        balance_score = max(0.0, 1.0 - (diff - 1) * 0.25)

    # Combine: average of range and balance
    return (range_score + balance_score) / 2.0
