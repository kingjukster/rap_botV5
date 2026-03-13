"""
evo_rhyme/archive.py

MAP-Elites quality-diversity archive for verse evolution.
Maintains a grid of behavioral niches, each storing the best individual
for that behavioral profile. Every niche gets equal reproductive opportunity,
promoting diverse exploration across rhyme density, intensity, syllable
tightness, and thematic breadth.
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from evo_rhyme.individual import VerseIndividual

logger = logging.getLogger(__name__)

AGGRESSIVE_KEYWORDS: frozenset[str] = frozenset({
    "kill", "gun", "blood", "war", "fight", "enemy", "death", "destroy",
    "rage", "fury", "savage", "beast", "lethal", "venom", "strike", "crush",
    "burn", "fire", "flex", "king", "boss", "reign", "throne", "crown",
    "power", "dominate", "conquer", "slay", "murder", "grave",
})

_WORD_RE = re.compile(r"[A-Za-z']+")


# ---------------------------------------------------------------------------
# ArchiveDimension
# ---------------------------------------------------------------------------

@dataclass
class ArchiveDimension:
    """One axis of the MAP-Elites behavioral space."""
    name: str
    bins: int
    bin_labels: List[str]
    extractor: Callable[[VerseIndividual], int]


# ---------------------------------------------------------------------------
# Default dimension extractors
# ---------------------------------------------------------------------------

def _extract_rhyme_density(individual: VerseIndividual) -> int:
    score = (individual.scores or {}).get("internal_rhyme", 0.0)
    if score < 0.15:
        return 0
    if score <= 0.35:
        return 1
    return 2


def _extract_intensity(individual: VerseIndividual) -> int:
    words: list[str] = []
    for line in individual.lines:
        words.extend(_WORD_RE.findall(line.lower()))
    if not words:
        return 0
    hits = sum(1 for w in words if w in AGGRESSIVE_KEYWORDS)
    ratio = hits / len(words)
    if ratio < 0.05:
        return 0
    if ratio <= 0.15:
        return 1
    return 2


def _extract_syllable_tightness(individual: VerseIndividual) -> int:
    if individual.features and individual.features.syllable_counts:
        counts = individual.features.syllable_counts
    else:
        from evo_rhyme.phonetics import syllable_count_line
        counts = [syllable_count_line(line) for line in individual.lines]
    if not counts:
        return 1
    avg = sum(counts) / len(counts)
    if avg < 9:
        return 0
    if avg <= 12:
        return 1
    return 2


def _extract_theme_balance(individual: VerseIndividual) -> int:
    score = (individual.scores or {}).get("semantic", 0.0)
    return 1 if score > 0.3 else 0


def default_verse_dimensions() -> List[ArchiveDimension]:
    """Return the default 4-dimensional behavioral space (54 niches)."""
    return [
        ArchiveDimension(
            name="rhyme_density",
            bins=3,
            bin_labels=["low", "medium", "high"],
            extractor=_extract_rhyme_density,
        ),
        ArchiveDimension(
            name="intensity",
            bins=3,
            bin_labels=["calm", "moderate", "aggressive"],
            extractor=_extract_intensity,
        ),
        ArchiveDimension(
            name="syllable_tightness",
            bins=3,
            bin_labels=["sparse", "normal", "dense"],
            extractor=_extract_syllable_tightness,
        ),
        ArchiveDimension(
            name="theme_balance",
            bins=2,
            bin_labels=["single_theme", "multi_theme"],
            extractor=_extract_theme_balance,
        ),
    ]


# ---------------------------------------------------------------------------
# MAPElitesArchive
# ---------------------------------------------------------------------------

class MAPElitesArchive:
    """
    MAP-Elites quality-diversity archive.

    Stores at most one individual per behavioral niche.  An individual only
    enters a niche if the niche is empty or the newcomer has strictly higher
    fitness than the current occupant.
    """

    def __init__(self, dimensions: List[ArchiveDimension]) -> None:
        self.dimensions = dimensions
        self._grid: Dict[Tuple[int, ...], VerseIndividual] = {}
        self._total_niches = math.prod(d.bins for d in dimensions)

    # -- behaviour mapping --------------------------------------------------

    def _get_behavior(self, individual: VerseIndividual) -> Tuple[int, ...]:
        coords: list[int] = []
        for dim in self.dimensions:
            try:
                idx = dim.extractor(individual)
            except Exception:
                idx = 0
            idx = max(0, min(idx, dim.bins - 1))
            coords.append(idx)
        return tuple(coords)

    # -- insertion ----------------------------------------------------------

    @staticmethod
    def _fitness_of(individual: VerseIndividual) -> float:
        return individual.fitness if individual.fitness is not None else -math.inf

    def add(self, individual: VerseIndividual) -> bool:
        coord = self._get_behavior(individual)
        existing = self._grid.get(coord)
        if existing is None or self._fitness_of(individual) > self._fitness_of(existing):
            self._grid[coord] = individual
            return True
        return False

    def add_batch(self, individuals: List[VerseIndividual]) -> int:
        return sum(1 for ind in individuals if self.add(ind))

    # -- sampling -----------------------------------------------------------

    def sample_parents(self, n: int) -> List[VerseIndividual]:
        if not self._grid:
            return []
        occupants = list(self._grid.values())
        return random.choices(occupants, k=n)

    # -- queries ------------------------------------------------------------

    def best_per_niche(self) -> Dict[Tuple[int, ...], VerseIndividual]:
        return dict(self._grid)

    def top_k(self, k: int) -> List[VerseIndividual]:
        ranked = sorted(
            self._grid.values(),
            key=lambda ind: self._fitness_of(ind),
            reverse=True,
        )
        return ranked[:k]

    def coverage(self) -> float:
        if self._total_niches == 0:
            return 0.0
        return len(self._grid) / self._total_niches

    def total_niches(self) -> int:
        return self._total_niches

    def occupied_niches(self) -> int:
        return len(self._grid)

    # -- reporting ----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        fitnesses = [
            self._fitness_of(ind) for ind in self._grid.values()
            if ind.fitness is not None
        ]
        per_dim: Dict[str, Dict[str, int]] = {}
        for dim in self.dimensions:
            counts: Dict[str, int] = {label: 0 for label in dim.bin_labels}
            per_dim[dim.name] = counts
        for coord in self._grid:
            for i, dim in enumerate(self.dimensions):
                label = dim.bin_labels[coord[i]]
                per_dim[dim.name][label] += 1

        return {
            "coverage": self.coverage(),
            "occupied": self.occupied_niches(),
            "total": self.total_niches(),
            "best_fitness": max(fitnesses) if fitnesses else None,
            "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else None,
            "per_dimension": per_dim,
        }

    # -- serialization ------------------------------------------------------

    def _coord_labels(self, coord: Tuple[int, ...]) -> List[str]:
        labels: list[str] = []
        for i, dim in enumerate(self.dimensions):
            idx = coord[i] if i < len(coord) else 0
            if idx < len(dim.bin_labels):
                labels.append(dim.bin_labels[idx])
            else:
                labels.append(str(idx))
        return labels

    def to_json(self) -> List[Dict[str, Any]]:
        entries: list[Dict[str, Any]] = []
        for coord, ind in sorted(self._grid.items()):
            entries.append({
                "niche": list(coord),
                "bin_labels": self._coord_labels(coord),
                "fitness": ind.fitness,
                "lines": list(ind.lines),
                "scores": dict(ind.scores) if ind.scores else {},
            })
        return entries

    @classmethod
    def from_json(
        cls,
        data: List[Dict[str, Any]],
        dimensions: List[ArchiveDimension],
    ) -> MAPElitesArchive:
        archive = cls(dimensions)
        for entry in data:
            coord = tuple(entry.get("niche", []))
            lines = entry.get("lines", [])
            fitness = entry.get("fitness")
            scores = entry.get("scores", {})
            ind = VerseIndividual(
                lines=lines,
                scores=scores if scores else None,
                fitness=fitness,
            )
            if coord and len(coord) == len(dimensions):
                archive._grid[coord] = ind
            else:
                archive.add(ind)
        logger.debug(
            "Restored archive from JSON: %d niches occupied", archive.occupied_niches()
        )
        return archive


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_verse_archive(
    dimensions: Optional[List[ArchiveDimension]] = None,
) -> MAPElitesArchive:
    """Create a MAP-Elites archive with default or custom dimensions."""
    if dimensions is None:
        dimensions = default_verse_dimensions()
    return MAPElitesArchive(dimensions)
