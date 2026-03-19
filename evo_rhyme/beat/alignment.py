from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import List, Optional, Tuple

from .grid import BeatSlot
from .prosody import SyllableUnit, extract_syllable_units


@dataclass(frozen=True)
class AlignmentStep:
    syllable_index: Optional[int]  # None if this step is a rest/skip
    slot_index: int
    action: str  # "place", "rest", "stretch"


@dataclass(frozen=True)
class AlignmentResult:
    score: float
    steps: List[AlignmentStep]
    syllables: List[SyllableUnit]


def place_score(syll: SyllableUnit, slot: BeatSlot, total_slots: int) -> float:
    score = 0.0

    # Reward stress on stronger beat positions.
    score += syll.stress * slot.strength

    # Reward rhyme-zone placement near end of bar.
    near_bar_end = slot.index % 16 >= 12
    if syll.is_rhyme_zone and near_bar_end:
        score += 0.55

    # Reward if the very last syllable lands near end.
    if syll.is_line_final and near_bar_end:
        score += 0.75

    # Slight preference for word-final syllables landing on cleaner positions.
    if syll.is_word_final and slot.subdivision_index in (0, 2):  # beat or "&"
        score += 0.10

    return score


def rest_penalty(slot: BeatSlot) -> float:
    """
    Penalty for leaving a slot empty.
    Small on weak subdivisions, slightly larger on stronger slots.
    """
    return 0.08 + 0.10 * slot.strength


def stretch_score(syll: SyllableUnit, slot_a: BeatSlot, slot_b: BeatSlot, total_slots: int) -> float:
    """
    Place the same syllable across two slots.
    Reward is weaker than placing two distinct syllables, but useful
    for plausible held delivery.
    """
    base = place_score(syll, slot_a, total_slots)
    extension = 0.18 * slot_b.strength

    if syll.is_line_final:
        extension += 0.15

    return base + extension


def density_penalty(num_syllables: int, num_slots_used: int) -> float:
    if num_slots_used <= 0:
        return 2.0
    density = num_syllables / num_slots_used
    if density <= 0.85:
        return 0.0
    return (density - 0.85) * 3.0


def overflow_penalty(unplaced_syllables: int) -> float:
    return 1.25 * unplaced_syllables


def align_syllables_dp(syllables: List[SyllableUnit], slots: List[BeatSlot]) -> AlignmentResult:
    """
    Dynamic programming alignment.

    State:
      dp[i][j] = best score after consuming first i syllables
                 and first j slots.

    Transitions:
      1) place one syllable on one slot
      2) skip slot as rest
      3) stretch one syllable across two slots
    """
    n = len(syllables)
    m = len(slots)

    neg_inf = float("-inf")
    dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[Tuple[int, int, str]]]] = [[None] * (m + 1) for _ in range(n + 1)]

    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == neg_inf:
                continue

            # Option 1: rest/skip current slot
            if j < m:
                cand = cur - rest_penalty(slots[j])
                if cand > dp[i][j + 1]:
                    dp[i][j + 1] = cand
                    back[i][j + 1] = (i, j, "rest")

            # Option 2: place one syllable on one slot
            if i < n and j < m:
                cand = cur + place_score(syllables[i], slots[j], m)
                if cand > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = cand
                    back[i + 1][j + 1] = (i, j, "place")

            # Option 3: stretch one syllable across two slots
            if i < n and j + 1 < m:
                cand = cur + stretch_score(syllables[i], slots[j], slots[j + 1], m)
                if cand > dp[i + 1][j + 2]:
                    dp[i + 1][j + 2] = cand
                    back[i + 1][j + 2] = (i, j, "stretch")

    # Choose best terminal state among all j after placing all syllables.
    best_j = 0
    best_score = neg_inf
    for j in range(m + 1):
        cand = dp[n][j] - density_penalty(n, max(j, 1))
        if cand > best_score:
            best_score = cand
            best_j = j

    # If not all syllables fit, allow ending with overflow penalty.
    for i in range(n):
        cand = dp[i][m] - overflow_penalty(n - i)
        if cand > best_score:
            best_score = cand
            best_j = m

    steps: List[AlignmentStep] = []
    i, j = n, best_j

    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj, action = prev

        if action == "rest":
            steps.append(AlignmentStep(syllable_index=None, slot_index=pj, action="rest"))
        elif action == "place":
            steps.append(AlignmentStep(syllable_index=pi, slot_index=pj, action="place"))
        elif action == "stretch":
            steps.append(AlignmentStep(syllable_index=pi, slot_index=pj, action="stretch"))
            steps.append(AlignmentStep(syllable_index=pi, slot_index=pj + 1, action="stretch"))

        i, j = pi, pj

    steps.reverse()
    return AlignmentResult(score=float(best_score), steps=steps, syllables=syllables)


def score_line_against_slots(line: str, slots: List[BeatSlot]) -> AlignmentResult:
    syllables = extract_syllable_units(line)
    if not syllables:
        return AlignmentResult(score=-5.0, steps=[], syllables=[])
    return align_syllables_dp(syllables, slots)


def pretty_print_alignment(result: AlignmentResult, slots: List[BeatSlot]) -> str:
    """
    Debug helper for development and scoring inspection.
    """
    slot_to_text = {slot.index: slot.label for slot in slots}
    placed: dict[int, str] = {}

    for step in result.steps:
        if step.syllable_index is None:
            placed.setdefault(step.slot_index, "—")
        else:
            syll = result.syllables[step.syllable_index].text
            if step.action == "stretch":
                placed.setdefault(step.slot_index, f"{syll}~")
            else:
                placed.setdefault(step.slot_index, syll)

    ordered = []
    for slot in slots:
        ordered.append(f"{slot_to_text[slot.index]}:{placed.get(slot.index, '_')}")
    return " | ".join(ordered)

