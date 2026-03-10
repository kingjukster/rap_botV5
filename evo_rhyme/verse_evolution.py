"""
evo_rhyme/verse_evolution.py

Evolution loop for 4-line verse optimization. Crossover swaps 2-line halves or
phrase slices; mutation reuses couplet mutation per line pair.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from evo_rhyme.constraints import passes_verse_constraints
from evo_rhyme.fitness import (
    VERSE_DEFAULT_WEIGHTS,
    compute_verse_fitness,
    score_verse,
)
from evo_rhyme.individual import (
    CoupletIndividual,
    VerseIndividual,
    analyze_individual,
    analyze_verse_individual,
)
from evo_rhyme.mutation import MUTATION_WEIGHTS, mutate
from evo_rhyme.phonetics import tokenize_line, syllable_count_line

logger = logging.getLogger(__name__)


@dataclass
class VerseEvolutionConfig:
    """Configuration for verse evolution loop."""
    population_size: int = 80
    num_elites: int = 5
    tournament_k: int = 3
    random_immigrants_per_gen: int = 5
    fitness_weights: Optional[Dict[str, float]] = None
    mutation_weights: Optional[Dict[str, float]] = None
    constraint_config: Optional[Any] = None
    max_offspring_attempts: int = 50
    phrase_slice: bool = False
    rhyme_scheme: str = "AABB"
    output_dir: Optional[Path] = None
    use_embeddings: bool = False
    embedding_weight: float = 0.5
    use_niching: bool = False


def verse_crossover(
    parent1: VerseIndividual,
    parent2: VerseIndividual,
    config: Optional[Any] = None,
) -> VerseIndividual:
    """
    Crossover two verse parents.
    - Swap 2-line halves: lines 1-2 from A, 3-4 from B (or vice versa)
    - Or phrase-slice: swap phrase slices between verses when structure matches
    """
    cfg = config or {}
    try_phrase_slice = cfg.get("phrase_slice", False)

    if try_phrase_slice and random.random() < 0.3:
        child = _verse_phrase_slice_crossover(parent1, parent2)
        if child is not None:
            return child

    # Standard 2-line half swap
    if random.random() < 0.5:
        lines = parent1.lines[:2] + parent2.lines[2:]
    else:
        lines = parent2.lines[:2] + parent1.lines[2:]

    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={},
    )


def _verse_phrase_slice_crossover(
    p_a: VerseIndividual,
    p_b: VerseIndividual,
) -> Optional[VerseIndividual]:
    """Swap phrase slices between verses when structure matches."""
    if len(p_a.lines) != 4 or len(p_b.lines) != 4:
        return None

    # Pick a line index to slice
    line_idx = random.randint(0, 3)
    tokens_a = tokenize_line(p_a.lines[line_idx])
    tokens_b = tokenize_line(p_b.lines[line_idx])

    if len(tokens_a) < 4 or len(tokens_b) < 4:
        return None

    n = min(len(tokens_a), len(tokens_b))
    split = max(2, n // 2)

    if random.random() < 0.5:
        new_tokens = tokens_a[:split] + tokens_b[split : split + max(0, len(tokens_a) - split)]
    else:
        new_tokens = tokens_b[:split] + tokens_a[split : split + max(0, len(tokens_b) - split)]

    if len(new_tokens) < 3:
        return None

    new_line = " ".join(new_tokens)
    lines = list(p_a.lines)
    lines[line_idx] = new_line
    return VerseIndividual(lines=lines, features=None, scores=None, fitness=None, metadata={})


def verse_mutate(
    individual: VerseIndividual,
    config: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
) -> VerseIndividual:
    """
    Mutate one line pair (1-2 or 3-4) using couplet mutation.
    Reuses mutate() from mutation.py per line pair.
    """
    pair_idx = random.randint(0, 1)  # 0 = lines 1-2, 1 = lines 3-4
    line1 = individual.lines[pair_idx * 2]
    line2 = individual.lines[pair_idx * 2 + 1]

    couplet = CoupletIndividual(line1=line1, line2=line2)
    mutated = mutate(couplet, config, weights)

    lines = list(individual.lines)
    lines[pair_idx * 2] = mutated.line1
    lines[pair_idx * 2 + 1] = mutated.line2

    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata=dict(individual.metadata),
    )


def _get_verse_rhyme_family(ind: VerseIndividual) -> tuple:
    """Rhyme family = (end_tail_1, end_tail_2, end_tail_3, end_tail_4) for grouping."""
    f = ind.features
    if not f or len(f.end_tails) != 4:
        return ("", "", "", "")
    return tuple(
        (t.raw if t else ()) for t in f.end_tails
    )


def _select_elites_niching_verse(
    population: List[VerseIndividual],
    k: int,
) -> List[VerseIndividual]:
    """
    Niching: group by rhyme family (end tails of 4 lines), preserve top 1 per
    family up to k, then fill remaining with next best regardless of family.
    population must be sorted by fitness (best first).
    """
    seen_families: Set[tuple] = set()
    elites: List[VerseIndividual] = []
    rest: List[VerseIndividual] = []
    for ind in population:
        family = _get_verse_rhyme_family(ind)
        if family not in seen_families and len(elites) < k:
            seen_families.add(family)
            elites.append(ind)
        else:
            rest.append(ind)
    for ind in rest:
        if len(elites) >= k:
            break
        elites.append(ind)
    return elites


def _tournament_select_verse(
    population: List[VerseIndividual],
    k: int,
) -> VerseIndividual:
    """Select one winner from k random individuals (higher fitness wins)."""
    if not population:
        raise ValueError("Empty population")
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda ind: ind.fitness or -1e9)


def evolve_verse_population(
    population: List[VerseIndividual],
    generations: int,
    config: Optional[VerseEvolutionConfig] = None,
    prompt_keywords: Optional[Set[str]] = None,
    immigrant_generator: Optional[Callable[[int], List[VerseIndividual]]] = None,
) -> List[VerseIndividual]:
    """
    Run verse evolution loop.

    Loop: analyze all -> compute fitness -> sort -> elites to next ->
          tournament select parents -> crossover -> mutate ->
          if passes_constraints append -> inject immigrants
    """
    cfg = config or VerseEvolutionConfig()
    weights = cfg.fitness_weights or VERSE_DEFAULT_WEIGHTS
    mut_weights = cfg.mutation_weights or MUTATION_WEIGHTS
    num_elites = min(cfg.num_elites, cfg.population_size)
    target_size = cfg.population_size
    scheme = cfg.rhyme_scheme or "AABB"

    semantic_scorer: Optional[Any] = None
    if cfg.use_embeddings:
        try:
            from rapbot.rhyme_scorer import SiameseRhymeScorer, SIAMESE_MODEL_DIR
            semantic_scorer = SiameseRhymeScorer(str(SIAMESE_MODEL_DIR), device="cpu")
            logger.info("Loaded SiameseRhymeScorer for verse semantic scoring")
        except Exception as e:
            logger.warning("Failed to load SiameseRhymeScorer: %s", e)

    theme_string = " ".join(prompt_keywords) if prompt_keywords else None
    kw = set(w.lower() for w in (prompt_keywords or [])) if prompt_keywords else None

    run_logger: Optional[VerseRunLogger] = None
    if cfg.output_dir:
        run_logger = VerseRunLogger(run_dir=cfg.output_dir)
        run_logger.write_config(
            cfg,
            extra={
                "generations": generations,
                "theme_keywords": list(prompt_keywords) if prompt_keywords else None,
                "rhyme_scheme": scheme,
            },
        )

    cc = cfg.constraint_config
    min_syl, max_syl = 6, 18
    if cc is not None:
        if isinstance(cc, dict):
            min_syl = cc.get("min_syllables", 6)
            max_syl = cc.get("max_syllables", 18)
        else:
            min_syl = getattr(cc, "min_syllables", 6)
            max_syl = getattr(cc, "max_syllables", 18)
    mutation_config = {
        "theme_keywords": list(prompt_keywords or []),
        "min_syllables": min_syl,
        "max_syllables": max_syl,
    }
    crossover_config = {"phrase_slice": cfg.phrase_slice}

    for gen in range(generations):
        for ind in population:
            analyze_verse_individual(ind)

        for ind in population:
            ind.scores = score_verse(
                ind,
                scheme=scheme,
                prompt_keywords=kw,
                theme_string=theme_string,
                semantic_scorer=semantic_scorer if cfg.use_embeddings else None,
                embedding_weight=cfg.embedding_weight,
            )
            ind.fitness = compute_verse_fitness(ind.scores, weights)

        population.sort(key=lambda x: x.fitness or -1e9, reverse=True)

        best = population[0].fitness if population else 0.0
        avg = sum(p.fitness or 0 for p in population) / max(1, len(population))
        top5 = population[:5]

        if run_logger:
            run_logger.log_generation(gen + 1, best, avg, top5)

        top5_preview = [
            f"  {i+1}. [{p.fitness:.3f}] " + " | ".join(
                (l[:25] + ("..." if len(l) > 25 else "")) for l in p.lines
            )
            for i, p in enumerate(top5)
        ]
        logger.info(f"Gen {gen + 1}/{generations} | best={best:.4f} avg={avg:.4f}")
        for line in top5_preview:
            logger.info(line)

        if gen == generations - 1:
            break

        # Elites
        elite_pool_size = num_elites * 5
        elite_pool = (
            _select_elites_niching_verse(population, elite_pool_size)
            if cfg.use_niching
            else population[:elite_pool_size]
        )
        elite_candidates = []
        for p in elite_pool:
            if len(elite_candidates) >= num_elites:
                break
            if not passes_verse_constraints(p, cfg.constraint_config):
                continue
            elite_candidates.append(p)
        next_pop: List[VerseIndividual] = list(elite_candidates)

        # Offspring
        attempts = 0
        accepted = 0
        max_attempts = (target_size - num_elites) * cfg.max_offspring_attempts
        while len(next_pop) < target_size - cfg.random_immigrants_per_gen and attempts < max_attempts:
            attempts += 1
            p1 = _tournament_select_verse(population, cfg.tournament_k)
            p2 = _tournament_select_verse(population, cfg.tournament_k)
            child = verse_crossover(p1, p2, crossover_config)
            child = verse_mutate(child, mutation_config, mut_weights)
            if passes_verse_constraints(child, cfg.constraint_config):
                accepted += 1
                next_pop.append(child)

        if immigrant_generator and cfg.random_immigrants_per_gen > 0:
            immigrants = immigrant_generator(cfg.random_immigrants_per_gen)
            for ind in immigrants:
                if passes_verse_constraints(ind, cfg.constraint_config):
                    next_pop.append(ind)
                    if len(next_pop) >= target_size:
                        break

        while len(next_pop) < target_size and len(population) > len(next_pop):
            next_pop.append(population[len(next_pop)])

        population = next_pop[:target_size]

    population.sort(key=lambda x: x.fitness or -1e9, reverse=True)

    if run_logger:
        run_logger.flush()

    return population


@dataclass
class VerseRunLogger:
    """Write verse run artifacts to run_dir."""

    run_dir: Path
    score_history: List[Dict[str, Any]] = field(default_factory=list)
    top_candidates_by_gen: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(
        self,
        config: VerseEvolutionConfig,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        data: Dict[str, Any] = {
            "population_size": config.population_size,
            "num_elites": config.num_elites,
            "tournament_k": config.tournament_k,
            "random_immigrants_per_gen": config.random_immigrants_per_gen,
            "max_offspring_attempts": config.max_offspring_attempts,
            "phrase_slice": config.phrase_slice,
            "rhyme_scheme": config.rhyme_scheme,
            "use_embeddings": config.use_embeddings,
            "embedding_weight": config.embedding_weight,
        }
        if extra:
            data.update(extra)
        path = self.run_dir / "config.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def log_generation(
        self,
        gen: int,
        best_fitness: float,
        avg_fitness: float,
        top5: List[VerseIndividual],
    ) -> None:
        self.score_history.append({
            "gen": gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
        })
        self.top_candidates_by_gen[gen] = [
            {
                "lines": ind.lines,
                "fitness": ind.fitness,
                "scores": ind.scores,
            }
            for ind in top5
        ]

    def flush(self) -> None:
        csv_path = self.run_dir / "score_history.csv"
        if self.score_history:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["gen", "best_fitness", "avg_fitness"],
                )
                writer.writeheader()
                writer.writerows(self.score_history)
        json_path = self.run_dir / "top_candidates.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.top_candidates_by_gen, f, indent=2)
