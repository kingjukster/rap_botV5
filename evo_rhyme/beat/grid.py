from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BeatSlot:
    index: int
    beat_index: int  # 0,1,2,3 in a 4/4 bar
    subdivision_index: int  # 0,1,2,3 for 16th-note grid
    strength: float
    label: str  # e.g. "1", "e", "&", "a"


def default_4_4_16_slot_strengths() -> List[float]:
    """
    A simple generic 4/4 rap-friendly strength profile for one bar.
    16 slots: 1 e & a 2 e & a 3 e & a 4 e & a

    Strong downbeats get higher values.
    Offbeats get medium values.
    Weak subdivisions get lower values.
    """
    return [
        1.00,
        0.35,
        0.65,
        0.25,  # 1 e & a
        0.85,
        0.35,
        0.65,
        0.25,  # 2 e & a
        0.85,
        0.35,
        0.65,
        0.25,  # 3 e & a
        0.95,
        0.35,
        0.75,
        0.25,  # 4 e & a
    ]


def make_4_4_bar_grid() -> List[BeatSlot]:
    labels = ["1", "e", "&", "a", "2", "e", "&", "a", "3", "e", "&", "a", "4", "e", "&", "a"]
    strengths = default_4_4_16_slot_strengths()

    slots: List[BeatSlot] = []
    for i, (label, strength) in enumerate(zip(labels, strengths)):
        beat_index = i // 4
        subdivision_index = i % 4
        slots.append(
            BeatSlot(
                index=i,
                beat_index=beat_index,
                subdivision_index=subdivision_index,
                strength=float(strength),
                label=label,
            )
        )
    return slots


def make_multi_bar_grid(num_bars: int) -> List[BeatSlot]:
    if num_bars < 1:
        raise ValueError("num_bars must be >= 1")

    base = make_4_4_bar_grid()
    out: List[BeatSlot] = []

    for bar in range(num_bars):
        offset = bar * len(base)
        for slot in base:
            out.append(
                BeatSlot(
                    index=offset + slot.index,
                    beat_index=slot.beat_index,
                    subdivision_index=slot.subdivision_index,
                    strength=slot.strength,
                    label=slot.label,
                )
            )
    return out

