"""
Block archive: stores scored 4-bar blocks organized by structural role.

Blocks are the building units for 16-bar verse composition. Each block
is a scored 4-bar verse with a role assignment (setup, development,
escalation, punchline).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScoredBlock:
    """A scored 4-bar block with role metadata."""
    lines: List[str]
    end_tails: List[Optional[str]] = field(default_factory=list)
    fitness: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    role: str = "flex"
    template_ids: Optional[List[Optional[str]]] = None

    @property
    def end_words(self) -> List[str]:
        """Last word of each line."""
        import re
        word_re = re.compile(r"[A-Za-z']+")
        result = []
        for line in self.lines:
            words = word_re.findall(line)
            result.append(words[-1].lower() if words else "")
        return result


VERSE_ROLES = ["setup", "development", "escalation", "punchline"]

ROLE_ASSIGNMENT_RULES = {
    # role -> (min_coherence, min_punchline, preferred_semantic_range)
    "setup": (0.3, 0.0, (0.3, 0.7)),
    "development": (0.35, 0.0, (0.4, 0.8)),
    "escalation": (0.3, 0.2, (0.5, 1.0)),
    "punchline": (0.2, 0.35, (0.4, 1.0)),
}


def assign_role(block: ScoredBlock) -> str:
    """Assign a structural role to a block based on its scores."""
    scores = block.scores
    punchline = scores.get("punchline", 0.0)
    coherence = scores.get("coherence", 0.0)
    semantic = scores.get("semantic", 0.0)

    if punchline >= 0.4:
        return "punchline"
    if semantic >= 0.6 and coherence >= 0.35:
        return "escalation"
    if coherence >= 0.4:
        return "development"
    return "setup"


class BlockArchive:
    """Archive of scored 4-bar blocks grouped by role."""

    def __init__(self, max_per_role: int = 50):
        self._max_per_role = max_per_role
        self._blocks: Dict[str, List[ScoredBlock]] = {
            role: [] for role in VERSE_ROLES
        }
        self._all_line_hashes: Set[int] = set()

    def add(self, block: ScoredBlock) -> bool:
        """Add a block to the archive. Returns True if accepted."""
        line_hash = hash(tuple(l.strip().lower() for l in block.lines))
        if line_hash in self._all_line_hashes:
            return False

        if not block.role or block.role not in VERSE_ROLES:
            block.role = assign_role(block)

        role_list = self._blocks[block.role]
        role_list.append(block)
        role_list.sort(key=lambda b: b.fitness, reverse=True)

        if len(role_list) > self._max_per_role:
            removed = role_list.pop()
            removed_hash = hash(tuple(l.strip().lower() for l in removed.lines))
            self._all_line_hashes.discard(removed_hash)

        self._all_line_hashes.add(line_hash)
        return True

    def add_batch(self, blocks: List[ScoredBlock]) -> int:
        """Add multiple blocks. Returns count of accepted blocks."""
        return sum(1 for b in blocks if self.add(b))

    def sample_by_role(self, role: str, k: int = 1) -> List[ScoredBlock]:
        """Sample k blocks for a given role, weighted by fitness."""
        role_list = self._blocks.get(role, [])
        if not role_list:
            all_blocks = [b for bs in self._blocks.values() for b in bs]
            if not all_blocks:
                return []
            role_list = all_blocks

        k = min(k, len(role_list))
        weights = [max(0.01, b.fitness) for b in role_list]
        total = sum(weights)
        probs = [w / total for w in weights]
        indices = random.choices(range(len(role_list)), weights=probs, k=k)
        return [role_list[i] for i in indices]

    def get_role_blocks(self, role: str) -> List[ScoredBlock]:
        """Get all blocks for a role, sorted by fitness."""
        return list(self._blocks.get(role, []))

    def size(self) -> int:
        """Total number of blocks across all roles."""
        return sum(len(bs) for bs in self._blocks.values())

    def role_sizes(self) -> Dict[str, int]:
        """Number of blocks per role."""
        return {role: len(bs) for role, bs in self._blocks.items()}

    def top_k(self, k: int = 10) -> List[ScoredBlock]:
        """Return top k blocks across all roles by fitness."""
        all_blocks = [b for bs in self._blocks.values() for b in bs]
        all_blocks.sort(key=lambda b: b.fitness, reverse=True)
        return all_blocks[:k]

    @classmethod
    def from_verse_individuals(
        cls,
        verses: List,  # List[VerseIndividual]
        max_per_role: int = 50,
    ) -> "BlockArchive":
        """Build a BlockArchive from a list of scored VerseIndividuals."""
        archive = cls(max_per_role=max_per_role)
        for verse in verses:
            if not hasattr(verse, 'lines') or len(verse.lines) != 4:
                continue
            end_tails = []
            try:
                from evo_rhyme.phonetics import extract_rhyme_tail, tokenize_line
                for line in verse.lines:
                    tokens = tokenize_line(line)
                    if tokens:
                        tail = extract_rhyme_tail(tokens[-1])
                        end_tails.append(str(tail) if tail else None)
                    else:
                        end_tails.append(None)
            except Exception:
                end_tails = [None] * 4

            block = ScoredBlock(
                lines=list(verse.lines),
                end_tails=end_tails,
                fitness=verse.fitness or 0.0,
                scores=dict(verse.scores) if verse.scores else {},
                template_ids=getattr(verse, 'template_ids', None),
            )
            block.role = assign_role(block)
            archive.add(block)

        logger.info(
            "Built BlockArchive from %d verses: %s",
            len(verses), archive.role_sizes(),
        )
        return archive
