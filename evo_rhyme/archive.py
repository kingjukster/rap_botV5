"""
evo_rhyme/archive.py

MAP-Elites quality-diversity archive for verse evolution.
Maintains a grid of behavioral niches, each storing the best individual
for that behavioral profile. Every niche gets equal reproductive opportunity,
promoting diverse exploration across rhyme density, intensity, syllable
tightness, and thematic breadth.
"""

from __future__ import annotations

import itertools
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
    if score < 0.08:
        return 0  # very_low
    if score < 0.15:
        return 1  # low
    if score < 0.25:
        return 2  # medium
    if score < 0.40:
        return 3  # high
    return 4  # very_high


def _extract_intensity(individual: VerseIndividual) -> int:
    words: list[str] = []
    for line in individual.lines:
        words.extend(_WORD_RE.findall(line.lower()))
    if not words:
        return 0
    hits = sum(1 for w in words if w in AGGRESSIVE_KEYWORDS)
    ratio = hits / len(words)
    if ratio < 0.03:
        return 0  # very_calm
    if ratio < 0.08:
        return 1  # calm
    if ratio < 0.15:
        return 2  # moderate
    if ratio < 0.25:
        return 3  # aggressive
    return 4  # very_aggressive


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


def _extract_line_length_variance(individual: VerseIndividual) -> int:
    if individual.features and individual.features.syllable_counts:
        counts = individual.features.syllable_counts
    else:
        from evo_rhyme.phonetics import syllable_count_line
        counts = [syllable_count_line(line) for line in individual.lines]
    if len(counts) < 2:
        return 1
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = math.sqrt(variance)
    if std < 1.5:
        return 0  # uniform
    if std <= 3.0:
        return 1  # moderate
    return 2  # varied


_METAPHOR_RE = re.compile(
    r"\b(?:like\s+a|like\s+the|i'?m\s+the|i'?m\s+a|as\s+(?:a|the))\b",
    re.IGNORECASE,
)


def _extract_metaphor_density(individual: VerseIndividual) -> int:
    text = " ".join(individual.lines).lower()
    hits = len(_METAPHOR_RE.findall(text))
    if hits == 0:
        return 0  # none
    if hits <= 1:
        return 1  # some
    return 2  # rich


_DARK_WORDS = frozenset({
    "pain", "death", "blood", "dark", "shadow", "grave", "suffer", "cry",
    "fear", "lost", "broke", "drown", "fall", "bleed", "scar", "wound",
    "trapped", "chains", "prison", "hell", "burn", "ashes", "cold", "alone",
    "hate", "rage", "agony", "doom", "curse",
})
_HOPEFUL_WORDS = frozenset({
    "light", "hope", "rise", "dream", "shine", "fly", "free", "love", "win",
    "gold", "crown", "glory", "bless", "peace", "faith", "bright", "heal",
    "grow", "build", "strength", "alive", "heaven", "soar", "triumph", "joy",
})


def _extract_sentiment_polarity(individual: VerseIndividual) -> int:
    words = []
    for line in individual.lines:
        words.extend(_WORD_RE.findall(line.lower()))
    dark = sum(1 for w in words if w in _DARK_WORDS)
    hopeful = sum(1 for w in words if w in _HOPEFUL_WORDS)
    total = dark + hopeful
    if total == 0:
        return 1  # neutral
    ratio = dark / total
    if ratio > 0.65:
        return 0  # dark
    if ratio < 0.35:
        return 2  # hopeful
    return 1  # neutral


def _extract_chain_length(individual: VerseIndividual) -> int:
    score = (individual.scores or {}).get("global_rhyme_chain_score", 0.0)
    if score < 0.25:
        return 0
    if score < 0.5:
        return 1
    if score < 0.75:
        return 2
    return 3


def _extract_graph_density(individual: VerseIndividual) -> int:
    score = (individual.scores or {}).get("rhyme_graph_density", 0.0)
    # Runtime observations show useful variation clustered well below 0.15.
    # Lower thresholds so the archive can "see" progress in graph complexity.
    if score < 0.03:
        return 0
    if score < 0.06:
        return 1
    if score < 0.12:
        return 2
    return 3


def _extract_graph_cluster(individual: VerseIndividual) -> int:
    score = (individual.scores or {}).get("rhyme_graph_cluster_coeff", 0.0)
    if score < 0.2:
        return 0
    if score < 0.4:
        return 1
    if score < 0.6:
        return 2
    return 3


def _extract_style_tone(individual: VerseIndividual) -> int:
    style = (individual.metadata or {}).get("style_genome_labels", {})
    tone = style.get("tone", "reflective")
    bins = ["reflective", "calm", "confident", "aggressive", "chaotic"]
    return bins.index(tone) if tone in bins else 0


def _extract_style_narrativity(individual: VerseIndividual) -> int:
    style = (individual.metadata or {}).get("style_genome_labels", {})
    narr = style.get("narrativity", "medium")
    bins = ["low", "medium", "high"]
    return bins.index(narr) if narr in bins else 1


def default_verse_dimensions() -> List[ArchiveDimension]:
    """Return the default 7-dimensional behavioral space (4050 niches)."""
    return [
        ArchiveDimension(
            name="rhyme_density",
            bins=5,
            bin_labels=["very_low", "low", "medium", "high", "very_high"],
            extractor=_extract_rhyme_density,
        ),
        ArchiveDimension(
            name="intensity",
            bins=5,
            bin_labels=["very_calm", "calm", "moderate", "aggressive", "very_aggressive"],
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
        ArchiveDimension(
            name="line_length_variance",
            bins=3,
            bin_labels=["uniform", "moderate", "varied"],
            extractor=_extract_line_length_variance,
        ),
        ArchiveDimension(
            name="metaphor_density",
            bins=3,
            bin_labels=["none", "some", "rich"],
            extractor=_extract_metaphor_density,
        ),
        ArchiveDimension(
            name="sentiment_polarity",
            bins=3,
            bin_labels=["dark", "neutral", "hopeful"],
            extractor=_extract_sentiment_polarity,
        ),
    ]


def style_chain_dimensions() -> List[ArchiveDimension]:
    """Extended archive dimensions for style/rhyme-chain exploration."""
    return default_verse_dimensions() + [
        ArchiveDimension(
            name="chain_length",
            bins=4,
            bin_labels=["short", "medium", "long", "very_long"],
            extractor=_extract_chain_length,
        ),
        ArchiveDimension(
            name="graph_density",
            bins=4,
            bin_labels=["sparse", "medium", "dense", "very_dense"],
            extractor=_extract_graph_density,
        ),
        ArchiveDimension(
            name="graph_cluster",
            bins=4,
            bin_labels=["loose", "mixed", "clustered", "highly_clustered"],
            extractor=_extract_graph_cluster,
        ),
        ArchiveDimension(
            name="style_tone",
            bins=5,
            bin_labels=["reflective", "calm", "confident", "aggressive", "chaotic"],
            extractor=_extract_style_tone,
        ),
        ArchiveDimension(
            name="style_narrativity",
            bins=3,
            bin_labels=["low", "medium", "high"],
            extractor=_extract_style_narrativity,
        ),
    ]


def compact_style_dimensions() -> List[ArchiveDimension]:
    """Compact 6-axis behavioral space for efficient QD coverage."""
    return [
        ArchiveDimension(
            name="rhyme_density",
            bins=5,
            bin_labels=["very_low", "low", "medium", "high", "very_high"],
            extractor=_extract_rhyme_density,
        ),
        ArchiveDimension(
            name="chain_length",
            bins=4,
            bin_labels=["short", "medium", "long", "very_long"],
            extractor=_extract_chain_length,
        ),
        ArchiveDimension(
            name="style_tone",
            bins=5,
            bin_labels=["reflective", "calm", "confident", "aggressive", "chaotic"],
            extractor=_extract_style_tone,
        ),
        ArchiveDimension(
            name="style_narrativity",
            bins=3,
            bin_labels=["low", "medium", "high"],
            extractor=_extract_style_narrativity,
        ),
        ArchiveDimension(
            name="metaphor_density",
            bins=3,
            bin_labels=["none", "some", "rich"],
            extractor=_extract_metaphor_density,
        ),
        ArchiveDimension(
            name="syllable_tightness",
            bins=3,
            bin_labels=["sparse", "normal", "dense"],
            extractor=_extract_syllable_tightness,
        ),
    ]


def _extract_rhyme_density_3bin(individual: VerseIndividual) -> int:
    """3 bins: low (0,1), medium (2), high (3,4)."""
    idx = _extract_rhyme_density(individual)
    if idx <= 1:
        return 0
    if idx == 2:
        return 1
    return 2


def _extract_chain_length_3bin(individual: VerseIndividual) -> int:
    """3 bins: short (0), medium (1), long (2,3)."""
    idx = _extract_chain_length(individual)
    if idx == 0:
        return 0
    if idx == 1:
        return 1
    return 2


def _extract_style_tone_3bin(individual: VerseIndividual) -> int:
    """3 bins: reflective (0,1), confident (2), aggressive (3,4)."""
    idx = _extract_style_tone(individual)
    if idx <= 1:
        return 0
    if idx == 2:
        return 1
    return 2


def ultra_compact_dimensions() -> List[ArchiveDimension]:
    """Ultra-compact 6-axis space (729 niches) for easier 50% coverage target."""
    return [
        ArchiveDimension(
            name="rhyme_density",
            bins=3,
            bin_labels=["low", "medium", "high"],
            extractor=_extract_rhyme_density_3bin,
        ),
        ArchiveDimension(
            name="chain_length",
            bins=3,
            bin_labels=["short", "medium", "long"],
            extractor=_extract_chain_length_3bin,
        ),
        ArchiveDimension(
            name="style_tone",
            bins=3,
            bin_labels=["reflective", "confident", "aggressive"],
            extractor=_extract_style_tone_3bin,
        ),
        ArchiveDimension(
            name="style_narrativity",
            bins=3,
            bin_labels=["low", "medium", "high"],
            extractor=_extract_style_narrativity,
        ),
        ArchiveDimension(
            name="metaphor_density",
            bins=3,
            bin_labels=["none", "some", "rich"],
            extractor=_extract_metaphor_density,
        ),
        ArchiveDimension(
            name="syllable_tightness",
            bins=3,
            bin_labels=["sparse", "normal", "dense"],
            extractor=_extract_syllable_tightness,
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

    # -- exploration helpers ------------------------------------------------

    def empty_niches(self) -> List[Tuple[int, ...]]:
        """Return list of unoccupied niche coordinates."""
        all_coords: List[Tuple[int, ...]] = []
        ranges = [range(d.bins) for d in self.dimensions]
        for coord in itertools.product(*ranges):
            if coord not in self._grid:
                all_coords.append(coord)
        return all_coords

    def sample_empty_niches(self, n: int, max_attempts: int = 5000) -> List[Tuple[int, ...]]:
        """Sample up to n empty niche coordinates without enumerating full grid."""
        if n <= 0:
            return []
        results: List[Tuple[int, ...]] = []
        seen: set[Tuple[int, ...]] = set()
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            coord = tuple(random.randrange(d.bins) for d in self.dimensions)
            if coord in seen:
                continue
            seen.add(coord)
            if coord not in self._grid:
                results.append(coord)
        return results

    def nearest_occupied(self, target: Tuple[int, ...]) -> Optional[VerseIndividual]:
        """Find the archive occupant nearest to *target* niche (Manhattan distance)."""
        if not self._grid:
            return None
        best_dist = float("inf")
        best_ind: Optional[VerseIndividual] = None
        for coord, ind in self._grid.items():
            dist = sum(abs(a - b) for a, b in zip(coord, target))
            if dist < best_dist:
                best_dist = dist
                best_ind = ind
        return best_ind

    def sample_distant_parents(
        self, n: int
    ) -> List[Tuple[VerseIndividual, VerseIndividual]]:
        """Sample *n* parent pairs from maximally distant niches."""
        if len(self._grid) < 2:
            return []
        coords = list(self._grid.keys())
        pairs: List[Tuple[VerseIndividual, VerseIndividual]] = []
        for _ in range(n):
            c1 = random.choice(coords)
            best_c2 = max(
                (c for c in coords if c != c1),
                key=lambda c: sum(abs(a - b) for a, b in zip(c1, c)),
                default=c1,
            )
            pairs.append((self._grid[c1], self._grid[best_c2]))
        return pairs

    def niche_label(self, coord: Tuple[int, ...]) -> Dict[str, str]:
        """Return human-readable labels for a niche coordinate."""
        labels: Dict[str, str] = {}
        for i, dim in enumerate(self.dimensions):
            idx = coord[i] if i < len(coord) else 0
            if idx < len(dim.bin_labels):
                labels[dim.name] = dim.bin_labels[idx]
            else:
                labels[dim.name] = str(idx)
        return labels

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
