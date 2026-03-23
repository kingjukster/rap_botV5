"""
Couplet archive for hierarchical evolution: stores scored couplets organized by rhyme pair.

Each couplet (2 lines) is scored on rhyme quality, fluency, semantic relevance, etc.
Couplets are grouped by (end_tail_1, end_tail_2) for AABB/ABAB verse assembly.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from evo_rhyme.phonetics import extract_rhyme_tail, tokenize_line

logger = logging.getLogger(__name__)


def _get_end_tail(line: str) -> str:
    """Extract rhyme tail from last word of line."""
    tokens = tokenize_line(line)
    if not tokens:
        return "__unknown__"
    tail = extract_rhyme_tail(tokens[-1])
    return str(tail) if tail else "__unknown__"


@dataclass
class ScoredCouplet:
    """A scored couplet (2 lines) with rhyme and quality dimensions."""

    line1: str
    line2: str
    end_tail_1: str = "__unknown__"
    end_tail_2: str = "__unknown__"
    fitness: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_couplet(
        line1: str,
        line2: str,
        fitness: float = 0.0,
        scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ScoredCouplet":
        """Create a ScoredCouplet from raw lines, computing end tails."""
        return ScoredCouplet(
            line1=line1,
            line2=line2,
            end_tail_1=_get_end_tail(line1),
            end_tail_2=_get_end_tail(line2),
            fitness=fitness,
            scores=scores or {},
            metadata=metadata or {},
        )

    def group_key(self) -> Tuple[str, str]:
        """Canonical key for grouping: (tail1, tail2) normalized for consistency."""
        t1, t2 = self.end_tail_1, self.end_tail_2
        if t1 > t2:
            t1, t2 = t2, t1
        return (t1, t2)


class CoupletArchive:
    """Archive of scored couplets grouped by rhyme pair (end_tail_1, end_tail_2).

    Supports sampling couplets for 4-bar verse assembly. For AABB, we need
    two couplets from different rhyme groups (A-A and B-B).
    """

    def __init__(self, max_per_group: int = 100):
        self._groups: Dict[Tuple[str, str], List[ScoredCouplet]] = defaultdict(list)
        self._max_per_group = max_per_group
        self._seen: Set[Tuple[str, str]] = set()  # (line1.lower, line2.lower)

    def add(self, couplet: ScoredCouplet) -> bool:
        """Add a couplet to the archive. Returns True if it was kept."""
        key = (couplet.line1.strip().lower(), couplet.line2.strip().lower())
        if key in self._seen:
            return False
        self._seen.add(key)

        gk = couplet.group_key()
        group = self._groups[gk]
        if len(group) < self._max_per_group:
            group.append(couplet)
            group.sort(key=lambda c: c.fitness, reverse=True)
            return True
        if couplet.fitness > group[-1].fitness:
            evicted = group.pop()
            self._seen.discard(
                (evicted.line1.strip().lower(), evicted.line2.strip().lower())
            )
            group.append(couplet)
            group.sort(key=lambda c: c.fitness, reverse=True)
            return True
        return False

    def add_batch(self, couplets: List[ScoredCouplet]) -> int:
        """Add multiple couplets. Returns count actually kept."""
        return sum(1 for c in couplets if self.add(c))

    def get_group(self, group_key: Tuple[str, str]) -> List[ScoredCouplet]:
        """Get all couplets in a rhyme group."""
        return list(self._groups.get(group_key, []))

    def sample(self, k: int = 1) -> List[ScoredCouplet]:
        """Sample k couplets from the archive, fitness-proportional across all groups."""
        all_couplets: List[ScoredCouplet] = []
        for group in self._groups.values():
            all_couplets.extend(group)
        if not all_couplets or k <= 0:
            return []
        k = min(k, len(all_couplets))
        weights = [max(0.01, c.fitness) for c in all_couplets]
        total = sum(weights)
        probs = [w / total for w in weights]
        indices = random.choices(range(len(all_couplets)), weights=probs, k=k)
        return [all_couplets[i] for i in indices]

    def sample_from_group(self, group_key: Tuple[str, str], k: int = 1) -> List[ScoredCouplet]:
        """Sample k couplets from a specific rhyme group."""
        group = self._groups.get(group_key, [])
        if not group or k <= 0:
            return []
        k = min(k, len(group))
        weights = [max(0.01, c.fitness) for c in group]
        total = sum(weights)
        probs = [w / total for w in weights]
        indices = random.choices(range(len(group)), weights=probs, k=k)
        return [group[i] for i in indices]

    def top_k(self, k: int = 50) -> List[ScoredCouplet]:
        """Return top k couplets across all groups by fitness."""
        all_couplets: List[ScoredCouplet] = []
        for group in self._groups.values():
            all_couplets.extend(group)
        all_couplets.sort(key=lambda c: c.fitness, reverse=True)
        return all_couplets[:k]

    def all_group_keys(self) -> List[Tuple[str, str]]:
        """Return all rhyme group keys with at least one couplet."""
        return [gk for gk, g in self._groups.items() if g]

    def eligible_pairs(self, min_per_group: int = 1) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Return pairs of group keys that can form a 4-bar verse (AABB: need 2 different groups)."""
        eligible = [
            gk for gk, g in self._groups.items()
            if len(g) >= min_per_group and "__unknown__" not in gk
        ]
        if len(eligible) < 2:
            return []
        result: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
        for i, g1 in enumerate(eligible):
            for g2 in eligible[i + 1 :]:
                result.append((g1, g2))
        return result

    def size(self) -> int:
        """Total number of couplets in the archive."""
        return sum(len(g) for g in self._groups.values())

    def group_count(self) -> int:
        """Number of non-empty rhyme groups."""
        return len([g for g in self._groups.values() if g])

    @classmethod
    def from_results_json(cls, path: Path, max_per_group: int = 100) -> "CoupletArchive":
        """Load CoupletArchive from run_couplet_evolution output JSON.

        Expected format: {"candidates": [{"line1": "...", "line2": "...", "fitness": ..., "scores": {...}}, ...]}
        """
        archive = cls(max_per_group=max_per_group)
        path = Path(path)
        if not path.exists():
            logger.warning("Couplet results path does not exist: %s", path)
            return archive

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        candidates = data.get("candidates", [])
        for c in candidates:
            line1 = c.get("line1", "")
            line2 = c.get("line2", "")
            if not line1 or not line2:
                continue
            fitness = float(c.get("fitness", 0.0))
            scores = c.get("scores")
            if isinstance(scores, dict):
                scores = {str(k): float(v) for k, v in scores.items()}
            else:
                scores = {}
            sc = ScoredCouplet.from_couplet(
                line1=line1,
                line2=line2,
                fitness=fitness,
                scores=scores,
            )
            archive.add(sc)

        logger.info(
            "Loaded CoupletArchive from %s: %d couplets, %d groups",
            path,
            archive.size(),
            archive.group_count(),
        )
        return archive

    def save_to_seed_bank(self, run_id: Optional[int] = None, seed_key: str = "couplet_archive") -> int:
        """Save archive to DB seed_bank when RAPBOT_USE_DB=1. Returns row id or -1."""
        try:
            from evo_rhyme import db
            if db.db_enabled():
                return db.insert_seed_bank(seed_key, self.to_json(), run_id=run_id)
        except Exception:
            pass
        return -1

    @classmethod
    def load_from_seed_bank(
        cls,
        seed_key: str = "couplet_archive",
        run_id: Optional[int] = None,
        max_per_group: int = 100,
    ) -> Optional["CoupletArchive"]:
        """Load CoupletArchive from DB seed_bank when available. Returns None if not found."""
        try:
            from evo_rhyme import db
            if not db.db_enabled():
                return None
            entries = db.list_seed_bank(run_id=run_id, limit=50)
            for e in entries:
                if e.get("seed_key") == seed_key and e.get("seed_data"):
                    data = e["seed_data"]
                    if not isinstance(data, dict):
                        continue
                    archive = cls(max_per_group=max_per_group)
                    for gk_str, couplets_data in data.get("groups", {}).items():
                        for c in couplets_data:
                            line1 = c.get("line1", "")
                            line2 = c.get("line2", "")
                            fitness = float(c.get("fitness", 0))
                            scores = c.get("scores", {})
                            if line1 and line2:
                                sc = ScoredCouplet.from_couplet(line1, line2, fitness=fitness, scores=scores)
                                archive.add(sc)
                    logger.info("Loaded CoupletArchive from seed_bank: %d couplets", archive.size())
                    return archive
        except Exception:
            pass
        return None

    def to_json(self) -> Dict[str, Any]:
        """Serialize archive to JSON-compatible dict."""
        return {
            "total_couplets": self.size(),
            "group_count": self.group_count(),
            "groups": {
                f"{gk[0]}|{gk[1]}": [
                    {
                        "line1": c.line1,
                        "line2": c.line2,
                        "fitness": c.fitness,
                        "scores": c.scores,
                    }
                    for c in group[:20]
                ]
                for gk, group in self._groups.items()
                if group
            },
        }
