"""
evo_rhyme/emitters.py

Emitter-based MAP-Elites for quality-diversity rap verse evolution.
Four specialized emitters generate candidates through different strategies,
coordinated by an adaptive scheduler that allocates budget based on success.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from evo_rhyme.individual import VerseIndividual, analyze_verse_individual

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = ["AABB", "ABAB", "ABBA", "ABCB", "AABA", "AAAA"]


@dataclass
class EmitResult:
    """Result of an emitter batch."""
    candidates: List[VerseIndividual]
    new_niches: int = 0
    improved_niches: int = 0
    total_inserted: int = 0


class BaseEmitter(ABC):
    """Abstract base for MAP-Elites emitters."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._total_new_niches = 0
        self._total_improved = 0

    @abstractmethod
    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        """Generate a batch of candidate verse individuals."""
        ...

    def update(self, result: EmitResult) -> None:
        """Update internal state after batch evaluation."""
        self._total_new_niches += result.new_niches
        self._total_improved += result.improved_niches


class RandomEmitter(BaseEmitter):
    """Pure exploration: random templates, random schemes, random vocab."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("random", config)
        self._corpus_path = config.get("corpus_path")
        self._theme_keywords = config.get("theme_keywords", [])

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.population import VerseSeedGenerator
        candidates = []
        schemes_to_try = self.config.get("schemes", ALLOWED_SCHEMES)

        for _ in range(batch_size):
            scheme = random.choice(schemes_to_try)
            try:
                gen = VerseSeedGenerator(
                    corpus_path=self._corpus_path,
                    init_mode="mixed",
                    scheme=scheme,
                )
                verses = gen.generate_seed_verses(
                    theme_keywords=self._theme_keywords,
                    size=1,
                )
                if verses:
                    v = verses[0]
                    v.metadata["origin"] = "random_emitter"
                    v.metadata["scheme"] = scheme
                    candidates.append(v)
            except Exception:
                logger.debug("RandomEmitter: generation failed", exc_info=True)

        return candidates


class MutationEmitter(BaseEmitter):
    """Mutate archive elites at multiple scales: small/medium/large/huge."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("mutation", config)

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.verse_evolution import verse_crossover, verse_mutate
        from evo_rhyme.constraints import passes_verse_constraints
        from evo_rhyme.mutation import MUTATION_WEIGHTS

        parents = archive.sample_parents(batch_size * 2)
        if not parents:
            return []

        mutation_config = {
            "theme_keywords": self.config.get("theme_keywords", []),
            "min_syllables": self.config.get("min_syllables", 6),
            "max_syllables": self.config.get("max_syllables", 18),
        }
        constraint_config = {
            "min_syllables": self.config.get("min_syllables", 6),
            "max_syllables": self.config.get("max_syllables", 18),
        }
        crossover_rate = self.config.get("crossover_rate", 0.5)

        candidates = []
        attempts = 0
        max_attempts = batch_size * 3

        while len(candidates) < batch_size and attempts < max_attempts:
            attempts += 1
            p1 = random.choice(parents)
            p2 = random.choice(parents)

            if random.random() < crossover_rate and len(parents) >= 2:
                child = verse_crossover(p1, p2, {})
            else:
                child = VerseIndividual(lines=list(p1.lines))

            r = random.random()
            if r < 0.10:
                from evo_rhyme.verse_evolution import _lm_verse_rewrite
                rewritten = _lm_verse_rewrite(child, mutation_config)
                if rewritten:
                    child = rewritten
                else:
                    child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS, constraint_config=constraint_config)
            elif r < 0.25:
                child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS, constraint_config=constraint_config)
                child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS, constraint_config=constraint_config)
            elif r < 0.50:
                heavy_weights = dict(MUTATION_WEIGHTS)
                for k in heavy_weights:
                    if k.startswith("lm_"):
                        heavy_weights[k] *= 1.5
                    if k in ("line_replace", "block_replace"):
                        heavy_weights[k] *= 2.0
                child = verse_mutate(child, mutation_config, heavy_weights, constraint_config=constraint_config)
            else:
                child = verse_mutate(child, mutation_config, MUTATION_WEIGHTS, constraint_config=constraint_config)

            if passes_verse_constraints(child, constraint_config):
                analyze_verse_individual(child)
                child.metadata["origin"] = "mutation_emitter"
                candidates.append(child)

        return candidates


class NicheTargetingEmitter(BaseEmitter):
    """Deliberately target empty niches by mutating nearest occupied elite."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("niche_targeting", config)
        self._target_queue: List[Tuple[int, ...]] = []

    def _refresh_targets(self, archive: Any) -> None:
        """Refresh the queue of empty niches to target."""
        empty = archive.empty_niches()
        random.shuffle(empty)
        self._target_queue = empty[:200]

    def _get_niche_mutation_hints(
        self, target_labels: Dict[str, str], archive: Any,
    ) -> Dict[str, Any]:
        """Convert niche labels into mutation guidance."""
        hints: Dict[str, Any] = {}
        theme_kw = list(self.config.get("theme_keywords", []))

        intensity = target_labels.get("intensity", "")
        if intensity in ("aggressive", "very_aggressive"):
            hints["force_aggressive"] = True
            hints["extra_keywords"] = [
                "kill", "slay", "venom", "strike", "crush", "burn", "fire",
                "flex", "king", "reign", "throne", "conquer", "savage",
            ]

        rhyme_d = target_labels.get("rhyme_density", "")
        if rhyme_d in ("high", "very_high"):
            hints["force_internal_rhyme"] = True

        syl = target_labels.get("syllable_tightness", "")
        if syl == "dense":
            hints["target_syllables"] = (12, 18)
        elif syl == "sparse":
            hints["target_syllables"] = (6, 9)

        sentiment = target_labels.get("sentiment_polarity", "")
        if sentiment == "dark":
            hints["extra_keywords"] = hints.get("extra_keywords", []) + [
                "pain", "shadow", "bleed", "drown", "cold", "grave",
            ]
        elif sentiment == "hopeful":
            hints["extra_keywords"] = hints.get("extra_keywords", []) + [
                "rise", "light", "dream", "shine", "free", "glory",
            ]

        metaphor = target_labels.get("metaphor_density", "")
        if metaphor == "rich":
            hints["force_metaphor"] = True

        return hints

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.verse_evolution import verse_mutate
        from evo_rhyme.constraints import passes_verse_constraints
        from evo_rhyme.mutation import MUTATION_WEIGHTS

        if not self._target_queue or generation % 5 == 0:
            self._refresh_targets(archive)

        if not self._target_queue:
            logger.info("NicheTargetingEmitter: no empty niches left")
            return []

        mutation_config = {
            "theme_keywords": list(self.config.get("theme_keywords", [])),
            "min_syllables": self.config.get("min_syllables", 6),
            "max_syllables": self.config.get("max_syllables", 18),
        }
        constraint_config = {
            "min_syllables": self.config.get("min_syllables", 6),
            "max_syllables": self.config.get("max_syllables", 18),
        }

        candidates = []
        targets_tried = 0

        while len(candidates) < batch_size and self._target_queue and targets_tried < batch_size * 3:
            targets_tried += 1
            target_coord = self._target_queue.pop(0)
            target_labels = archive.niche_label(target_coord)

            parent = archive.nearest_occupied(target_coord)
            if parent is None:
                continue

            hints = self._get_niche_mutation_hints(target_labels, archive)

            mut_cfg = dict(mutation_config)
            if hints.get("extra_keywords"):
                mut_cfg["theme_keywords"] = mut_cfg["theme_keywords"] + hints["extra_keywords"]
            if hints.get("target_syllables"):
                mut_cfg["min_syllables"] = hints["target_syllables"][0]
                mut_cfg["max_syllables"] = hints["target_syllables"][1]
                constraint_config_local = {
                    "min_syllables": hints["target_syllables"][0],
                    "max_syllables": hints["target_syllables"][1],
                }
            else:
                constraint_config_local = constraint_config

            heavy_weights = dict(MUTATION_WEIGHTS)
            for k in heavy_weights:
                if k.startswith("lm_"):
                    heavy_weights[k] *= 2.0
                if k in ("line_replace", "block_replace"):
                    heavy_weights[k] *= 3.0

            for attempt in range(3):
                child = VerseIndividual(lines=list(parent.lines), metadata={"origin": "niche_targeting"})
                child = verse_mutate(child, mut_cfg, heavy_weights, constraint_config=constraint_config_local)

                if hints.get("force_aggressive"):
                    aggressive_words = hints.get("extra_keywords", [])
                    if aggressive_words:
                        line_idx = random.randint(0, len(child.lines) - 1)
                        tokens = child.lines[line_idx].split()
                        if len(tokens) >= 3:
                            inject_pos = random.randint(1, len(tokens) - 2)
                            tokens[inject_pos] = random.choice(aggressive_words)
                            child.lines[line_idx] = " ".join(tokens)

                if passes_verse_constraints(child, constraint_config_local):
                    analyze_verse_individual(child)
                    child.metadata["target_niche"] = list(target_coord)
                    child.metadata["target_labels"] = target_labels
                    candidates.append(child)
                    break

        return candidates


class RepairEmitter(BaseEmitter):
    """Repair broken-but-promising candidates via LM rewriting."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("repair", config)
        self._broken_pool: List[VerseIndividual] = []

    def add_broken(self, individuals: List[VerseIndividual]) -> None:
        """Add broken candidates to the repair pool."""
        self._broken_pool.extend(individuals)
        if len(self._broken_pool) > 500:
            self._broken_pool = self._broken_pool[-500:]

    def emit(
        self,
        archive: Any,
        batch_size: int,
        generation: int,
    ) -> List[VerseIndividual]:
        from evo_rhyme.constraints import passes_verse_constraints
        from evo_rhyme.phonetics import tokenize_line

        if not self._broken_pool:
            return []

        mutation_config = {
            "theme_keywords": self.config.get("theme_keywords", []),
            "min_syllables": self.config.get("min_syllables", 6),
            "max_syllables": self.config.get("max_syllables", 18),
        }
        constraint_config = {
            "min_syllables": self.config.get("min_syllables", 6),
            "max_syllables": self.config.get("max_syllables", 18),
        }

        candidates = []
        to_repair = self._broken_pool[:batch_size]
        self._broken_pool = self._broken_pool[batch_size:]

        try:
            from evo_rhyme.mutation import get_rewriter
            rewriter = get_rewriter(mutation_config)
        except Exception:
            logger.warning("RepairEmitter: could not get rewriter")
            return []

        min_syl = mutation_config.get("min_syllables", 6)
        max_syl = mutation_config.get("max_syllables", 18)

        for individual in to_repair:
            try:
                repaired_lines = list(individual.lines)
                did_repair = False

                for i in range(0, len(individual.lines), 2):
                    if i + 1 >= len(individual.lines):
                        break

                    line1 = individual.lines[i]
                    line2 = individual.lines[i + 1]

                    tokens2 = tokenize_line(line2)
                    rhyme_target = tokens2[-1] if tokens2 else "night"

                    result = rewriter.repair(
                        line1, rhyme_target, (min_syl, max_syl),
                    )
                    if result:
                        repaired_lines[i] = result[0]
                        did_repair = True

                if did_repair:
                    new_ind = VerseIndividual(
                        lines=repaired_lines,
                        metadata={"origin": "repair_emitter"},
                    )
                    if passes_verse_constraints(new_ind, constraint_config):
                        analyze_verse_individual(new_ind)
                        candidates.append(new_ind)
            except Exception:
                logger.debug("RepairEmitter: repair failed for one individual", exc_info=True)

        return candidates


@dataclass
class EmitterStats:
    """Track per-emitter success over a sliding window."""
    new_niches: deque = field(default_factory=lambda: deque(maxlen=10))
    improved: deque = field(default_factory=lambda: deque(maxlen=10))

    def record(self, new_n: int, improved_n: int) -> None:
        self.new_niches.append(new_n)
        self.improved.append(improved_n)

    def success_score(self) -> float:
        """success = 2*(new niches) + 1*(improved elites) over window."""
        return 2.0 * sum(self.new_niches) + 1.0 * sum(self.improved)


class EmitterScheduler:
    """Adaptive scheduler: allocates budget proportional to emitter success."""

    def __init__(
        self,
        emitters: List[BaseEmitter],
        initial_weights: Optional[Dict[str, float]] = None,
        min_weight: float = 0.05,
    ):
        self.emitters = emitters
        self._stats: Dict[str, EmitterStats] = {e.name: EmitterStats() for e in emitters}
        self._min_weight = min_weight

        if initial_weights:
            self._weights = {e.name: initial_weights.get(e.name, 1.0 / len(emitters)) for e in emitters}
        else:
            self._weights = {e.name: 1.0 / len(emitters) for e in emitters}

    def choose(self) -> BaseEmitter:
        """Choose an emitter proportional to current weights."""
        names = [e.name for e in self.emitters]
        weights = [self._weights[n] for n in names]
        chosen_name = random.choices(names, weights=weights, k=1)[0]
        return next(e for e in self.emitters if e.name == chosen_name)

    def allocate_budget(self, total_budget: int) -> Dict[str, int]:
        """Allocate batch sizes to each emitter based on weights."""
        allocation: Dict[str, int] = {}
        remaining = total_budget
        emitter_list = list(self.emitters)

        for i, emitter in enumerate(emitter_list):
            if i == len(emitter_list) - 1:
                allocation[emitter.name] = remaining
            else:
                share = max(1, int(total_budget * self._weights[emitter.name]))
                share = min(share, remaining - (len(emitter_list) - i - 1))
                allocation[emitter.name] = share
                remaining -= share

        return allocation

    def update(self, emitter: BaseEmitter, result: EmitResult) -> None:
        """Update stats and recompute weights."""
        self._stats[emitter.name].record(result.new_niches, result.improved_niches)

        total_success = sum(s.success_score() for s in self._stats.values())

        if total_success > 0:
            for name, stats in self._stats.items():
                raw = stats.success_score() / total_success
                self._weights[name] = max(self._min_weight, raw)

        total_w = sum(self._weights.values())
        if total_w > 0:
            self._weights = {k: v / total_w for k, v in self._weights.items()}

    def summary(self) -> Dict[str, Any]:
        """Return current scheduler state."""
        return {
            "weights": dict(self._weights),
            "stats": {
                name: {
                    "success_score": stats.success_score(),
                    "recent_new_niches": list(stats.new_niches),
                    "recent_improved": list(stats.improved),
                }
                for name, stats in self._stats.items()
            },
        }


def create_emitters(config: Dict[str, Any]) -> Tuple[List[BaseEmitter], EmitterScheduler]:
    """Factory: create all 4 emitters and the adaptive scheduler.

    config should contain:
        theme_keywords: List[str]
        corpus_path: Optional[Path]
        min_syllables: int
        max_syllables: int
        schemes: List[str]  (allowed rhyme schemes)
        crossover_rate: float
    """
    random_emitter = RandomEmitter(config)
    mutation_emitter = MutationEmitter(config)
    niche_emitter = NicheTargetingEmitter(config)
    repair_emitter = RepairEmitter(config)

    emitters = [random_emitter, mutation_emitter, niche_emitter, repair_emitter]

    initial_weights = {
        "random": 0.30,
        "mutation": 0.35,
        "niche_targeting": 0.25,
        "repair": 0.10,
    }

    scheduler = EmitterScheduler(emitters, initial_weights=initial_weights)
    return emitters, scheduler


def run_emitter_generation(
    archive: Any,
    emitters: List[BaseEmitter],
    scheduler: EmitterScheduler,
    total_budget: int,
    generation: int,
    score_fn: Callable,
    fitness_fn: Callable,
    novelty_weight: float = 0.3,
) -> Dict[str, Any]:
    """Run one generation of emitter-based MAP-Elites.

    Args:
        archive: MAPElitesArchive instance
        emitters: List of emitter instances
        scheduler: EmitterScheduler
        total_budget: Total candidates to generate this generation
        generation: Current generation number
        score_fn: Callable(List[VerseIndividual]) -> List[Dict[str, float]]
        fitness_fn: Callable(Dict[str, float]) -> float
        novelty_weight: Weight for novelty in effective fitness (0.0-1.0)

    Returns:
        Dict with generation stats
    """
    allocation = scheduler.allocate_budget(total_budget)

    all_candidates: List[VerseIndividual] = []
    emitter_map: List[Tuple[BaseEmitter, int, int]] = []
    broken_candidates: List[VerseIndividual] = []

    for emitter in emitters:
        budget = allocation.get(emitter.name, 0)
        if budget <= 0:
            continue

        batch = emitter.emit(archive, budget, generation)
        start = len(all_candidates)
        all_candidates.extend(batch)
        emitter_map.append((emitter, start, len(all_candidates)))

    if not all_candidates:
        return {"total_candidates": 0, "inserted": 0, "new_niches": 0}

    try:
        all_scores = score_fn(all_candidates)
        for cand, sc in zip(all_candidates, all_scores):
            cand.scores = sc
            cand.fitness = fitness_fn(sc)
    except Exception:
        logger.warning("Batch scoring failed", exc_info=True)
        return {"total_candidates": len(all_candidates), "inserted": 0, "new_niches": 0}

    if novelty_weight > 0:
        try:
            from evo_rhyme.scoring.novelty import embed_texts, NoveltyArchive
            texts = [" ".join(c.lines) for c in all_candidates]
            embs = embed_texts(texts)
            archive_texts = [" ".join(ind.lines) for ind in archive.top_k(min(50, archive.occupied_niches()))]
            if archive_texts:
                archive_embs = embed_texts(archive_texts)
                import numpy as np
                for i, cand in enumerate(all_candidates):
                    dists = [
                        float(np.linalg.norm(embs[i] - ae))
                        for ae in archive_embs
                    ]
                    novelty = min(dists) if dists else 0.5
                    cand.scores["novelty"] = min(1.0, novelty)
                    effective = (1.0 - novelty_weight) * (cand.fitness or 0) + novelty_weight * novelty
                    cand.metadata["effective_fitness"] = effective
        except Exception:
            logger.debug("Novelty computation failed, using raw fitness", exc_info=True)

    occupied_before = archive.occupied_niches()

    per_emitter_results: Dict[str, EmitResult] = {}

    for emitter, start, end in emitter_map:
        batch = all_candidates[start:end]
        new_niches = 0
        improved = 0

        for cand in batch:
            occ_pre = archive.occupied_niches()
            inserted = archive.add(cand)
            if inserted:
                if archive.occupied_niches() > occ_pre:
                    new_niches += 1
                else:
                    improved += 1
            elif cand.scores and cand.scores.get("garbled_line_penalty", 0) > 0.15:
                broken_candidates.append(cand)

        result = EmitResult(
            candidates=batch,
            new_niches=new_niches,
            improved_niches=improved,
            total_inserted=new_niches + improved,
        )
        per_emitter_results[emitter.name] = result
        emitter.update(result)
        scheduler.update(emitter, result)

    repair = next((e for e in emitters if isinstance(e, RepairEmitter)), None)
    if repair and broken_candidates:
        repair.add_broken(broken_candidates)

    total_new = archive.occupied_niches() - occupied_before
    total_inserted = sum(r.total_inserted for r in per_emitter_results.values())

    return {
        "total_candidates": len(all_candidates),
        "inserted": total_inserted,
        "new_niches": total_new,
        "per_emitter": {
            name: {
                "budget": allocation.get(name, 0),
                "generated": len(r.candidates),
                "new_niches": r.new_niches,
                "improved": r.improved_niches,
            }
            for name, r in per_emitter_results.items()
        },
        "scheduler_weights": scheduler._weights,
        "broken_pool_size": len(repair._broken_pool) if repair else 0,
    }
