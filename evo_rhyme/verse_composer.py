"""
Verse composer: assembles 16-bar verses from 4-bar blocks.

Uses a BlockArchive to select blocks by role, arranges them into
a full verse structure, and provides evolutionary operators at
the block level (swap, crossover, mutation).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from evo_rhyme.block_archive import BlockArchive, ScoredBlock, VERSE_ROLES
from evo_rhyme.individual import VerseIndividual, VerseStructure

logger = logging.getLogger(__name__)

DEFAULT_16_STRUCTURE = ["setup", "development", "escalation", "punchline"]

SCHEME_PRESETS_16 = [
    "AABB" * 4,           # each block independent AABB
    "AABBCCDDAABBCCDD",   # two rhyme cycles
    "AAAABBBBCCCCDDDD",   # each block one rhyme
    "AABBCCDDEEFFGGHH",   # maximally diverse
    "ABABABABCDCDCDCD",    # interleaved
]


def compose_verse(
    block_archive: BlockArchive,
    structure: Optional[List[str]] = None,
    scheme: str = "AABB",
    used_blocks: Optional[set] = None,
) -> Optional[VerseIndividual]:
    """Assemble a 16-bar verse from 4-bar blocks.

    Selects one block per role from the archive, concatenates their
    lines into a 16-line verse.

    Args:
        block_archive: Archive of scored 4-bar blocks.
        structure: List of 4 role names. Defaults to DEFAULT_16_STRUCTURE.
        scheme: Rhyme scheme string (will be extended to 16 chars).
        used_blocks: Set of block line hashes to avoid reuse.

    Returns:
        VerseIndividual with 16 lines, or None if not enough blocks.
    """
    if structure is None:
        structure = list(DEFAULT_16_STRUCTURE)
    if used_blocks is None:
        used_blocks = set()

    blocks: List[ScoredBlock] = []
    for role in structure:
        candidates = block_archive.sample_by_role(role, k=5)
        selected = None
        for cand in candidates:
            h = hash(tuple(l.strip().lower() for l in cand.lines))
            if h not in used_blocks:
                selected = cand
                used_blocks.add(h)
                break
        if selected is None and candidates:
            selected = candidates[0]
        if selected is None:
            return None
        blocks.append(selected)

    lines: List[str] = []
    template_ids: List[Optional[str]] = []
    for block in blocks:
        lines.extend(block.lines)
        if block.template_ids:
            template_ids.extend(block.template_ids)
        else:
            template_ids.extend([None] * len(block.lines))

    full_scheme = scheme
    if len(full_scheme) < 16:
        repeats = (16 + len(full_scheme) - 1) // len(full_scheme)
        full_scheme = (full_scheme * repeats)[:16]

    vs = VerseStructure.for_scheme(full_scheme, num_lines=16)

    verse = VerseIndividual(
        lines=lines,
        structure=vs,
        features=None,
        scores=None,
        fitness=None,
        metadata={
            "origin": "block_composition",
            "block_roles": [b.role for b in blocks],
            "block_fitnesses": [b.fitness for b in blocks],
        },
    )
    if hasattr(verse, 'template_ids'):
        verse.template_ids = template_ids

    return verse


def compose_verse_batch(
    block_archive: BlockArchive,
    count: int,
    structure: Optional[List[str]] = None,
    scheme: str = "AABB",
) -> List[VerseIndividual]:
    """Build multiple 16-bar verses, maximizing diversity."""
    verses: List[VerseIndividual] = []
    used_blocks: set = set()
    attempts = 0
    max_attempts = count * 5

    while len(verses) < count and attempts < max_attempts:
        attempts += 1
        verse = compose_verse(
            block_archive, structure, scheme, used_blocks,
        )
        if verse is not None:
            verses.append(verse)

    logger.info("Composed %d 16-bar verses (%d attempts)", len(verses), attempts)
    return verses


def block_swap_crossover(
    parent1: VerseIndividual,
    parent2: VerseIndividual,
    block_size: int = 4,
) -> VerseIndividual:
    """Crossover two 16-bar verses by swapping blocks.

    Randomly selects a crossover point (block boundary) and takes
    blocks before the point from parent1, blocks after from parent2.
    """
    n1 = len(parent1.lines)
    n2 = len(parent2.lines)
    num_blocks = min(n1, n2) // block_size

    if num_blocks < 2:
        return VerseIndividual(
            lines=list(parent1.lines),
            metadata={"origin": "block_crossover"},
        )

    split = random.randint(1, num_blocks - 1)
    split_idx = split * block_size

    lines = list(parent1.lines[:split_idx]) + list(parent2.lines[split_idx:n2])

    return VerseIndividual(
        lines=lines[:num_blocks * block_size],
        features=None,
        scores=None,
        fitness=None,
        metadata={"origin": "block_crossover", "split_block": split},
    )


def single_block_swap(
    parent1: VerseIndividual,
    parent2: VerseIndividual,
    block_size: int = 4,
) -> VerseIndividual:
    """Replace one random block in parent1 with the corresponding block from parent2."""
    n = min(len(parent1.lines), len(parent2.lines))
    num_blocks = n // block_size

    if num_blocks < 1:
        return VerseIndividual(lines=list(parent1.lines))

    block_idx = random.randint(0, num_blocks - 1)
    start = block_idx * block_size
    end = start + block_size

    lines = list(parent1.lines)
    lines[start:end] = parent2.lines[start:end]

    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={"origin": "single_block_swap", "block_idx": block_idx},
    )


def block_mutate(
    individual: VerseIndividual,
    block_size: int = 4,
    config: Optional[Dict[str, Any]] = None,
    lm_budget: Optional[Dict[str, int]] = None,
) -> VerseIndividual:
    """Mutate one random block within a 16-bar verse using existing verse mutation."""
    from evo_rhyme.individual import CoupletIndividual
    from evo_rhyme.mutation import mutate as couplet_mutate
    from evo_rhyme.constraints import passes_constraints

    n = len(individual.lines)
    num_blocks = n // block_size
    if num_blocks < 1:
        return individual

    block_idx = random.randint(0, num_blocks - 1)
    start = block_idx * block_size

    lines = list(individual.lines)

    pair_in_block = random.randint(0, (block_size // 2) - 1)
    line_start = start + pair_in_block * 2

    if line_start + 1 >= n:
        return individual

    couplet = CoupletIndividual(
        line1=lines[line_start],
        line2=lines[line_start + 1],
    )
    mutated = couplet_mutate(couplet, config, lm_budget=lm_budget)
    if passes_constraints(mutated, config):
        lines[line_start] = mutated.line1
        lines[line_start + 1] = mutated.line2

    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata=dict(individual.metadata),
    )


def verse_16_crossover(
    parent1: VerseIndividual,
    parent2: VerseIndividual,
    block_size: int = 4,
) -> VerseIndividual:
    """Crossover operator for 16-bar verses. Randomly picks strategy."""
    r = random.random()
    if r < 0.5:
        return block_swap_crossover(parent1, parent2, block_size)
    else:
        return single_block_swap(parent1, parent2, block_size)


def verse_16_mutate(
    individual: VerseIndividual,
    block_size: int = 4,
    config: Optional[Dict[str, Any]] = None,
    lm_budget: Optional[Dict[str, int]] = None,
) -> VerseIndividual:
    """Mutation operator for 16-bar verses."""
    return block_mutate(individual, block_size, config, lm_budget)
