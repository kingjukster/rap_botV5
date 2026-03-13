"""
evo_rhyme/evolution.py

Evolution loop for couplet optimization. Tournament selection, crossover,
mutation, elite preservation, and immigrant injection.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from evo_rhyme.constraints import ConstraintConfig, passes_constraints
from evo_rhyme.crossover import crossover
from evo_rhyme.fitness import (
    DEFAULT_WEIGHTS,
    NGRAM_FLOOR,
    _get_rhyme_family,
    compute_fitness,
    score_couplet,
)
from evo_rhyme.style_profile import StyleProfile
from evo_rhyme.individual import CoupletIndividual, analyze_individual
from evo_rhyme.mutation import MUTATION_WEIGHTS, mutate
from evo_rhyme.phonetics import extract_rhyme_tail, tokenize_line

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for the evolution loop."""
    population_size: int = 100
    num_elites: int = 5
    tournament_k: int = 3
    random_immigrants_per_gen: int = 5
    fitness_weights: Optional[Dict[str, float]] = None
    mutation_weights: Optional[Dict[str, float]] = None
    constraint_config: Optional[Any] = None
    max_offspring_attempts: int = 50  # retries for constraint-violating offspring
    phrase_slice: bool = False
    use_niching: bool = False
    multiobjective: bool = False  # when True, use evolve_multiobjective (Pareto ranking)
    output_dir: Optional[Path] = None  # run artifacts: config, score_history, top_candidates
    use_embeddings: bool = False
    embedding_weight: float = 0.5  # weight of embedding score in semantic blend: semantic = (1-alpha)*keyword + alpha*embedding
    population_init: str = "mixed"  # "mixed" | "random" | "template"
    style_profile: Optional[StyleProfile] = None
    style_weight: float = 0.1  # weight of style similarity in fitness when style_profile provided
    corpus_lines: Optional[List[str]] = None  # for corpus_overlap_penalty (avoid overfitting to seed)
    min_fluency_accept: float = 0.0  # reject mutations with fluency below this (0=disabled)
    min_semantic_accept: float = 0.0  # reject mutations with semantic below this (0=disabled)
    min_lexical_accept: float = 0.0  # reject mutations with lexical_validity below this (0=disabled)
    min_ngram_fluency_accept: float = 0.0  # reject mutations with ngram_fluency below this (0=disabled). Use 0.2 to block nonsense phrase structure.
    use_lm_fluency: bool = False  # blend ngram with LM perplexity for phrase plausibility (stronger nonsense detection)
    lm_fluency_weight: float = 0.5  # weight of LM score in ngram_fluency blend (0.5 = 50% ngram, 50% LM)
    require_theme_presence: bool = False  # when True + prompt_keywords, enforce at least one keyword in couplet


# ---------------------------------------------------------------------------
# Pareto ranking for multi-objective evolution
# ---------------------------------------------------------------------------

OBJECTIVE_KEYS = ["end_rhyme", "fluency", "semantic", "corpus_novelty", "rhyme_family_novelty"]


def _objectives_from_scores(scores: Optional[Dict[str, float]]) -> List[float]:
    """Extract objectives from scores dict. Includes rhyme_family_novelty when present."""
    if not scores:
        return [0.0, 0.0, 0.0, 1.0, 1.0]
    corpus_overlap = scores.get("corpus_overlap_penalty", 0.0)
    corpus_novelty = 1.0 - corpus_overlap
    rhyme_family_novelty = scores.get("rhyme_family_novelty", 1.0)
    return [
        scores.get("end_rhyme", 0.0),
        scores.get("fluency", 0.0),
        scores.get("semantic", 0.0),
        corpus_novelty,
        rhyme_family_novelty,
    ]


def _dominates(a_obj: List[float], b_obj: List[float]) -> bool:
    """A dominates B if A >= B on all objectives and strictly better on at least one."""
    if len(a_obj) != len(b_obj):
        return False
    better_or_equal = all(ai >= bi for ai, bi in zip(a_obj, b_obj))
    strictly_better = any(ai > bi for ai, bi in zip(a_obj, b_obj))
    return better_or_equal and strictly_better


def _compute_pareto_ranks(
    population: List[CoupletIndividual],
) -> Dict[int, int]:
    """
    Non-dominated sorting: assign rank to each individual.
    Rank 0 = Pareto frontier (not dominated by anyone).
    Returns dict: index_in_population -> rank.
    """
    n = len(population)
    objectives = [_objectives_from_scores(ind.scores) for ind in population]
    domination_count: List[int] = [0] * n  # how many individuals dominate this one
    dominated_by: List[Set[int]] = [set() for _ in range(n)]  # indices this one dominates

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j]):
                dominated_by[i].add(j)
            elif _dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

    ranks: Dict[int, int] = {}
    current_rank = 0
    remaining = set(range(n))
    while remaining:
        frontier = {i for i in remaining if domination_count[i] == 0}
        for i in frontier:
            ranks[i] = current_rank
        for i in frontier:
            for j in dominated_by[i]:
                domination_count[j] -= 1
        remaining -= frontier
        current_rank += 1
    return ranks


def _crowding_distance_for_indices(
    population: List[CoupletIndividual],
    indices: List[int],
) -> Dict[int, float]:
    """
    Crowding distance for given indices (e.g. same rank). Higher = less crowded.
    """
    if len(indices) <= 2:
        return {i: float("inf") for i in indices}
    objectives = [_objectives_from_scores(population[i].scores) for i in indices]
    n_obj = len(OBJECTIVE_KEYS)
    distances: Dict[int, float] = {i: 0.0 for i in indices}
    for m in range(n_obj):
        vals = [(objectives[j][m], indices[j]) for j in range(len(indices))]
        vals.sort(key=lambda x: x[0])
        lo, hi = vals[0][0], vals[-1][0]
        span = hi - lo if hi > lo else 1.0
        for j, (_, idx) in enumerate(vals):
            if j == 0 or j == len(vals) - 1:
                distances[idx] = float("inf")
            else:
                prev_val = vals[j - 1][0]
                next_val = vals[j + 1][0]
                distances[idx] += (next_val - prev_val) / span
    return distances


def _compute_crowding_by_rank(
    population: List[CoupletIndividual],
    ranks: Dict[int, int],
) -> Dict[int, float]:
    """Compute crowding distance per individual, within each rank level."""
    by_rank: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(population)):
        by_rank[ranks.get(i, 999)].append(i)
    crowding: Dict[int, float] = {}
    for rank_indices in by_rank.values():
        sub = _crowding_distance_for_indices(population, rank_indices)
        crowding.update(sub)
    return crowding


def _effective_constraint_config(
    config: Optional[EvolutionConfig],
    prompt_keywords: Optional[Set[str]],
) -> Optional[Any]:
    """Build constraint config with theme presence merged when require_theme_presence and prompt_keywords."""
    if not config:
        return None
    base = config.constraint_config
    if not config.require_theme_presence or not prompt_keywords:
        return base
    kw_lower = set(w.lower() for w in prompt_keywords)
    if isinstance(base, dict):
        return {**base, "require_theme_presence": True, "prompt_keywords": kw_lower}
    if isinstance(base, ConstraintConfig):
        return {
            "min_syllables": base.min_syllables,
            "max_syllables": base.max_syllables,
            "min_words_per_line": base.min_words_per_line,
            "weak_words": base.weak_words,
            "max_token_repeats": base.max_token_repeats,
            "require_stressed_end": base.require_stressed_end,
            "repetition_stopwords": base.repetition_stopwords,
            "require_theme_presence": True,
            "prompt_keywords": kw_lower,
        }
    return {"require_theme_presence": True, "prompt_keywords": kw_lower}


def _tournament_select(
    population: List[CoupletIndividual],
    k: int,
) -> CoupletIndividual:
    """Select one winner from k random individuals (higher fitness wins)."""
    if not population:
        raise ValueError("Empty population")
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda ind: ind.fitness or -1e9)


def _tournament_select_multiobjective(
    population: List[CoupletIndividual],
    k: int,
    ranks: Dict[int, int],
    crowding: Dict[int, float],
) -> CoupletIndividual:
    """Select one winner from k random individuals (lower rank wins; tie-break by crowding, then raw fitness)."""
    if not population:
        raise ValueError("Empty population")
    indices = list(range(len(population)))
    candidates = random.sample(indices, min(k, len(indices)))
    # Prefer lower rank, then higher crowding, then higher raw fitness
    return population[
        min(candidates, key=lambda i: (ranks.get(i, 999), -crowding.get(i, 0.0), -(population[i].fitness or 0)))
    ]


def _end_words_from_population(population: List[CoupletIndividual]) -> List[str]:
    """Extract end words (last word of each line) from all couplets."""
    words: List[str] = []
    for ind in population:
        for line in (ind.line1, ind.line2):
            tokens = tokenize_line(line)
            if tokens:
                words.append(tokens[-1].lower())
    return words


def _compute_diversity(population: List[CoupletIndividual]) -> float:
    """Diversity = len(unique_tails) / max(1, len(population)*2)."""
    words = _end_words_from_population(population)
    tails = [extract_rhyme_tail(w) for w in words]
    unique_tails = {t for t in tails if t is not None}
    return len(unique_tails) / max(1, len(population) * 2)


def _top_rhyme_tails(population: List[CoupletIndividual], k: int = 5) -> List[tuple]:
    """Return top k (tail, count) from end-word rhyme tails."""
    words = _end_words_from_population(population)
    tails = [extract_rhyme_tail(w) for w in words if extract_rhyme_tail(w)]
    counts = Counter(tails)
    return counts.most_common(k)


def _select_elites_niching(
    population: List[CoupletIndividual],
    k: int,
) -> List[CoupletIndividual]:
    """
    Niching: group by rhyme family (end tail line1+line2), preserve top 1 per
    family up to k, then fill remaining with next best regardless of family.
    population must be sorted by fitness (best first).
    """
    seen_families: Set[tuple] = set()
    elites: List[CoupletIndividual] = []
    rest: List[CoupletIndividual] = []
    for ind in population:
        family = _get_rhyme_family(ind)
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


# ---------------------------------------------------------------------------
# EvolutionRunLogger
# ---------------------------------------------------------------------------


@dataclass
class EvolutionRunLogger:
    """Write run artifacts to run_dir (e.g. data/evo_rhyme/runs/{timestamp}/)."""

    run_dir: Path
    score_history: List[Dict[str, Any]] = field(default_factory=list)
    top_candidates_by_gen: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: EvolutionConfig, extra: Optional[Dict[str, Any]] = None) -> None:
        """Write config.json to run dir."""
        data: Dict[str, Any] = {
            "population_size": config.population_size,
            "num_elites": config.num_elites,
            "tournament_k": config.tournament_k,
            "random_immigrants_per_gen": config.random_immigrants_per_gen,
            "max_offspring_attempts": config.max_offspring_attempts,
            "phrase_slice": config.phrase_slice,
            "use_niching": config.use_niching,
            "multiobjective": config.multiobjective,
            "use_embeddings": config.use_embeddings,
            "embedding_weight": config.embedding_weight,
            "population_init": config.population_init,
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
        diversity: float,
        acceptance_rate: float,
        top5: List[CoupletIndividual],
        best_raw: Optional[float] = None,
    ) -> None:
        """Append to score_history and store top candidates."""
        entry: Dict[str, Any] = {
            "gen": gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "diversity": diversity,
            "acceptance_rate": acceptance_rate,
        }
        if best_raw is not None:
            entry["best_raw"] = best_raw
        self.score_history.append(entry)
        self.top_candidates_by_gen[gen] = [
            {
                "line1": ind.line1,
                "line2": ind.line2,
                "fitness": ind.fitness,
                "scores": ind.scores,
            }
            for ind in top5
        ]

    def flush(self) -> None:
        """Write score_history.csv and top_candidates.json to run dir."""
        # score_history.csv
        csv_path = self.run_dir / "score_history.csv"
        if self.score_history:
            fieldnames = ["gen", "best_fitness", "avg_fitness", "diversity", "acceptance_rate"]
            if any("best_raw" in e for e in self.score_history):
                fieldnames.append("best_raw")
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.score_history)

        # top_candidates.json
        json_path = self.run_dir / "top_candidates.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.top_candidates_by_gen, f, indent=2)


def evolve(
    population: List[CoupletIndividual],
    generations: int,
    config: Optional[EvolutionConfig] = None,
    prompt_keywords: Optional[Set[str]] = None,
    immigrant_generator: Optional[Callable[[int], List[CoupletIndividual]]] = None,
) -> List[CoupletIndividual]:
    """
    Run evolution loop.

    Loop: analyze all -> compute fitness -> sort -> elites to next ->
          tournament select parents -> crossover -> mutate ->
          if passes_constraints append -> inject immigrants

    Args:
        population: Initial population (will be analyzed and scored).
        generations: Number of generations.
        config: EvolutionConfig (or None for defaults).
        prompt_keywords: Theme keywords for semantic fitness.
        immigrant_generator: Callable(size) -> List[CoupletIndividual] for new immigrants.
                             If None, no immigrants injected.

    Returns:
        Final population sorted by fitness (best first).
    """
    cfg = config or EvolutionConfig()
    if cfg.multiobjective:
        return evolve_multiobjective(
            population, generations, config, prompt_keywords, immigrant_generator
        )
    weights = cfg.fitness_weights or DEFAULT_WEIGHTS
    mut_weights = cfg.mutation_weights or MUTATION_WEIGHTS
    num_elites = min(cfg.num_elites, cfg.population_size)
    target_size = cfg.population_size

    # Siamese scorer for embedding-based semantic when use_embeddings is True
    semantic_scorer: Optional[Any] = None
    if cfg.use_embeddings:
        try:
            from rapbot.rhyme_scorer import SiameseRhymeScorer, SIAMESE_MODEL_DIR
            semantic_scorer = SiameseRhymeScorer(str(SIAMESE_MODEL_DIR), device="cpu")
            logger.info("Loaded SiameseRhymeScorer for embedding-based semantic scoring")
        except Exception as e:
            logger.warning("Failed to load SiameseRhymeScorer, falling back to keyword-only semantic: %s", e)

    theme_string = " ".join(prompt_keywords) if prompt_keywords else None

    run_logger: Optional[EvolutionRunLogger] = None
    if cfg.output_dir:
        run_logger = EvolutionRunLogger(run_dir=cfg.output_dir)
        run_logger.write_config(
            cfg,
            extra={
                "generations": generations,
                "theme_keywords": list(prompt_keywords) if prompt_keywords else None,
            },
        )

    prev_acceptance_rate: Optional[float] = None

    for gen in range(generations):
        # Analyze all
        for ind in population:
            analyze_individual(ind)

        # Compute fitness
        kw = set(w.lower() for w in (prompt_keywords or [])) if prompt_keywords else None
        for ind in population:
            ind.scores = score_couplet(
                ind,
                prompt_keywords=kw,
                theme_string=theme_string,
                semantic_scorer=semantic_scorer if cfg.use_embeddings else None,
                embedding_weight=cfg.embedding_weight,
                corpus_lines=cfg.corpus_lines,
                use_lm_fluency=cfg.use_lm_fluency,
                lm_fluency_weight=cfg.lm_fluency_weight,
            )
            ngram_floor = cfg.min_ngram_fluency_accept if cfg.min_ngram_fluency_accept > 0 else (NGRAM_FLOOR if cfg.corpus_lines else None)  # default floor when corpus available
            ind.fitness = compute_fitness(
                ind.scores,
                weights,
                individual=ind,
                population=population,
                style_profile=cfg.style_profile,
                style_weight=cfg.style_weight if cfg.style_profile else 0.0,
                ngram_floor=ngram_floor,
            )

        # Sort by fitness (best first)
        population.sort(key=lambda x: x.fitness or -1e9, reverse=True)

        best = population[0].fitness if population else 0.0
        avg = sum(p.fitness or 0 for p in population) / max(1, len(population))
        top5 = population[:5]
        ngram_floor = cfg.min_ngram_fluency_accept if cfg.min_ngram_fluency_accept > 0 else (NGRAM_FLOOR if cfg.corpus_lines else None)
        best_raw = (
            compute_fitness(
                population[0].scores,
                weights,
                individual=population[0],
                population=population,
                style_profile=cfg.style_profile,
                style_weight=cfg.style_weight if cfg.style_profile else 0.0,
                ngram_floor=ngram_floor,
            )
            if population else None
        )

        diversity = _compute_diversity(population)
        top_tails = _top_rhyme_tails(population, k=5)
        acceptance_rate = prev_acceptance_rate if prev_acceptance_rate is not None else 0.0

        top5_preview = [
            f"  {i+1}. [{p.fitness:.3f}] {p.line1[:35]}{'...' if len(p.line1)>35 else ''} | {p.line2[:35]}{'...' if len(p.line2)>35 else ''}"
            for i, p in enumerate(top5)
        ]
        log_line = (
            f"Gen {gen + 1}/{generations} | best={best:.4f} avg={avg:.4f} "
            f"diversity={diversity:.3f} acceptance_rate={acceptance_rate:.3f}"
        )
        if best_raw is not None:
            log_line += f" best_raw={best_raw:.4f}"
        logger.info(log_line)
        for line in top5_preview:
            logger.info(line)
        logger.info(f"  Top 5 rhyme tails: {[(t, c) for t, c in top_tails]}")

        if run_logger:
            run_logger.log_generation(
                gen + 1, best, avg, diversity, acceptance_rate, top5, best_raw=best_raw
            )

        if gen == generations - 1:
            break

        # Next generation: elites (must pass full constraints + identical/near-duplicate checks)
        def _line_overlap(l1: str, l2: str) -> float:
            import re
            w = re.compile(r"[A-Za-z']+")
            t1, t2 = set(w.findall(l1.lower())), set(w.findall(l2.lower()))
            if not t1 or not t2:
                return 0.0
            return len(t1 & t2) / max(len(t1), len(t2))

        elite_pool_size = num_elites * 5
        elite_pool = (
            _select_elites_niching(population, elite_pool_size)
            if cfg.use_niching
            else population[:elite_pool_size]
        )
        elites_candidates = []
        for p in elite_pool:
            if len(elites_candidates) >= num_elites:
                break
            if p.line1.strip().lower() == p.line2.strip().lower():
                continue
            if _line_overlap(p.line1, p.line2) > 0.78:
                continue
            if not passes_constraints(p, _effective_constraint_config(cfg, prompt_keywords)):
                continue
            elites_candidates.append(p)
        next_pop: List[CoupletIndividual] = list(elites_candidates)

        # Config for crossover and mutation
        crossover_config = {"phrase_slice": cfg.phrase_slice}
        cc = cfg.constraint_config
        min_syl, max_syl = 6, 18
        if cc is not None:
            if isinstance(cc, dict):
                min_syl = cc.get("min_syllables", 6)
                max_syl = cc.get("max_syllables", 18)
            else:
                min_syl = getattr(cc, "min_syllables", 6)
                max_syl = getattr(cc, "max_syllables", 18)
        corpus_vocab: Optional[Set[str]] = None
        if cfg.corpus_lines:
            import re
            word_re = re.compile(r"[A-Za-z']+")
            corpus_vocab = set()
            for line in cfg.corpus_lines[:2000]:
                corpus_vocab.update(w.lower() for w in word_re.findall(line.lower()))
        mutation_config = {
            "theme_keywords": list(prompt_keywords or []),
            "min_syllables": min_syl,
            "max_syllables": max_syl,
            "corpus_vocab": corpus_vocab,
        }

        # Offspring via tournament select -> crossover -> mutate
        attempts = 0
        accepted = 0
        max_attempts = (target_size - num_elites) * cfg.max_offspring_attempts
        while len(next_pop) < target_size - cfg.random_immigrants_per_gen and attempts < max_attempts:
            attempts += 1
            p1 = _tournament_select(population, cfg.tournament_k)
            p2 = _tournament_select(population, cfg.tournament_k)
            child = crossover(p1, p2, crossover_config)
            child = mutate(child, mutation_config, mut_weights)
            if not passes_constraints(child, _effective_constraint_config(cfg, prompt_keywords)):
                continue
            needs_scores = (cfg.min_fluency_accept > 0 or cfg.min_semantic_accept > 0 or cfg.min_lexical_accept > 0
                           or cfg.min_ngram_fluency_accept > 0 or (cfg.corpus_lines and cfg.min_ngram_fluency_accept == 0))
            if needs_scores:
                analyze_individual(child)
                child_scores = score_couplet(
                    child,
                    prompt_keywords=kw,
                    theme_string=theme_string,
                    semantic_scorer=semantic_scorer if cfg.use_embeddings else None,
                    embedding_weight=cfg.embedding_weight,
                    corpus_lines=cfg.corpus_lines,
                    use_lm_fluency=cfg.use_lm_fluency,
                    lm_fluency_weight=cfg.lm_fluency_weight,
                )
                if cfg.min_fluency_accept > 0 and child_scores.get("fluency", 0) < cfg.min_fluency_accept:
                    continue
                if cfg.min_semantic_accept > 0 and child_scores.get("semantic", 0) < cfg.min_semantic_accept:
                    continue
                if cfg.min_lexical_accept > 0 and child_scores.get("lexical_validity", 0.5) < cfg.min_lexical_accept:
                    continue
                ngram_floor_val = cfg.min_ngram_fluency_accept if cfg.min_ngram_fluency_accept > 0 else (NGRAM_FLOOR if cfg.corpus_lines else None)
                if ngram_floor_val is not None and child_scores.get("ngram_fluency", 0.5) < ngram_floor_val:
                    continue
            accepted += 1
            next_pop.append(child)
        prev_acceptance_rate = accepted / max(1, attempts)
        logger.info(f"  Mutation acceptance: {accepted}/{attempts} = {prev_acceptance_rate:.2%}")

        # Inject immigrants (use effective config so theme presence is enforced when required)
        if cfg.random_immigrants_per_gen > 0 and immigrant_generator:
            immigrants = immigrant_generator(cfg.random_immigrants_per_gen)
            eff_cc = _effective_constraint_config(cfg, prompt_keywords)
            for ind in immigrants:
                if passes_constraints(ind, eff_cc):
                    next_pop.append(ind)
                    if len(next_pop) >= target_size:
                        break

        # Pad with elites if under target (e.g. too many constraint failures)
        while len(next_pop) < target_size and len(population) > len(next_pop):
            next_pop.append(population[len(next_pop)])

        population = next_pop[:target_size]

    population.sort(key=lambda x: x.fitness or -1e9, reverse=True)

    if run_logger:
        run_logger.flush()

    return population


def evolve_multiobjective(
    population: List[CoupletIndividual],
    generations: int,
    config: Optional[EvolutionConfig] = None,
    prompt_keywords: Optional[Set[str]] = None,
    immigrant_generator: Optional[Callable[[int], List[CoupletIndividual]]] = None,
) -> List[CoupletIndividual]:
    """
    Multi-objective evolution using Pareto ranking.

    Same loop as evolve() but selection uses Pareto rank instead of raw fitness.
    Objectives: [rhyme_score (end_rhyme), fluency_score, semantic_score].
    Fitness = 1 / (1 + rank); rank 0 = Pareto frontier.
    Uses crowding distance for tie-breaking within same rank.
    """
    cfg = config or EvolutionConfig()
    weights = cfg.fitness_weights or DEFAULT_WEIGHTS
    mut_weights = cfg.mutation_weights or MUTATION_WEIGHTS
    num_elites = min(cfg.num_elites, cfg.population_size)
    target_size = cfg.population_size

    semantic_scorer: Optional[Any] = None
    if cfg.use_embeddings:
        try:
            from rapbot.rhyme_scorer import SiameseRhymeScorer, SIAMESE_MODEL_DIR
            semantic_scorer = SiameseRhymeScorer(str(SIAMESE_MODEL_DIR), device="cpu")
            logger.info("Loaded SiameseRhymeScorer for embedding-based semantic scoring")
        except Exception as e:
            logger.warning("Failed to load SiameseRhymeScorer, falling back to keyword-only semantic: %s", e)

    theme_string = " ".join(prompt_keywords) if prompt_keywords else None

    run_logger: Optional[EvolutionRunLogger] = None
    if cfg.output_dir:
        run_logger = EvolutionRunLogger(run_dir=cfg.output_dir)
        run_logger.write_config(
            cfg,
            extra={
                "generations": generations,
                "theme_keywords": list(prompt_keywords) if prompt_keywords else None,
                "multiobjective": True,
            },
        )

    prev_acceptance_rate: Optional[float] = None

    for gen in range(generations):
        for ind in population:
            analyze_individual(ind)

        kw = set(w.lower() for w in (prompt_keywords or [])) if prompt_keywords else None
        for ind in population:
            ind.scores = score_couplet(
                ind,
                prompt_keywords=kw,
                theme_string=theme_string,
                semantic_scorer=semantic_scorer if cfg.use_embeddings else None,
                embedding_weight=cfg.embedding_weight,
                corpus_lines=cfg.corpus_lines,
                use_lm_fluency=cfg.use_lm_fluency,
                lm_fluency_weight=cfg.lm_fluency_weight,
            )
        for ind in population:
            family = _get_rhyme_family(ind)
            count = sum(1 for o in population if _get_rhyme_family(o) == family)
            novelty = 1.0 - (count / max(1, len(population)))
            ind.scores["rhyme_family_novelty"] = novelty

        ranks = _compute_pareto_ranks(population)
        crowding = _compute_crowding_by_rank(population, ranks)
        for i, ind in enumerate(population):
            raw_fitness = compute_fitness(
                ind.scores,
                weights,
                individual=ind,
                population=population,
                style_profile=cfg.style_profile,
                style_weight=cfg.style_weight if cfg.style_profile else 0.0,
            )
            ind.fitness = raw_fitness
            ind.metadata["pareto_rank"] = ranks.get(i, 999)

        # Sort by rank (asc), then crowding (desc), then raw fitness (desc) for tie-break
        def _sort_key(idx: int) -> tuple:
            return (ranks.get(idx, 999), -crowding.get(idx, 0.0), -(population[idx].fitness or 0))

        sorted_order = sorted(range(len(population)), key=_sort_key)
        population = [population[i] for i in sorted_order]
        # Remap ranks/crowding to new indices for tournament selection
        ranks_by_pos = {k: ranks[sorted_order[k]] for k in range(len(population))}
        crowding_by_pos = {k: crowding[sorted_order[k]] for k in range(len(population))}

        best = population[0].fitness if population else 0.0
        avg = sum(p.fitness or 0 for p in population) / max(1, len(population))
        top5 = population[:5]

        diversity = _compute_diversity(population)
        top_tails = _top_rhyme_tails(population, k=5)
        acceptance_rate = prev_acceptance_rate if prev_acceptance_rate is not None else 0.0

        top5_preview = [
            f"  {i+1}. [rank={p.metadata.get('pareto_rank', 999)} fit={p.fitness:.3f}] {p.line1[:35]}{'...' if len(p.line1)>35 else ''} | {p.line2[:35]}{'...' if len(p.line2)>35 else ''}"
            for i, p in enumerate(top5)
        ]
        log_line = (
            f"Gen {gen + 1}/{generations} (multiobj) | best_fit={best:.4f} avg={avg:.4f} "
            f"diversity={diversity:.3f} acceptance_rate={acceptance_rate:.3f}"
        )
        logger.info(log_line)
        for line in top5_preview:
            logger.info(line)
        logger.info(f"  Top 5 rhyme tails: {[(t, c) for t, c in top_tails]}")

        if run_logger:
            run_logger.log_generation(
                gen + 1, best, avg, diversity, acceptance_rate, top5
            )

        if gen == generations - 1:
            break

        def _line_overlap(l1: str, l2: str) -> float:
            import re
            w = re.compile(r"[A-Za-z']+")
            t1, t2 = set(w.findall(l1.lower())), set(w.findall(l2.lower()))
            if not t1 or not t2:
                return 0.0
            return len(t1 & t2) / max(len(t1), len(t2))

        elite_pool_size = num_elites * 5
        elite_pool = (
            _select_elites_niching(population, elite_pool_size)
            if cfg.use_niching
            else population[:elite_pool_size]
        )
        elites_candidates = []
        for p in elite_pool:
            if len(elites_candidates) >= num_elites:
                break
            if p.line1.strip().lower() == p.line2.strip().lower():
                continue
            if _line_overlap(p.line1, p.line2) > 0.78:
                continue
            if not passes_constraints(p, _effective_constraint_config(cfg, prompt_keywords)):
                continue
            elites_candidates.append(p)
        next_pop: List[CoupletIndividual] = list(elites_candidates)

        crossover_config = {"phrase_slice": cfg.phrase_slice}
        cc = cfg.constraint_config
        min_syl, max_syl = 6, 18
        if cc is not None:
            if isinstance(cc, dict):
                min_syl = cc.get("min_syllables", 6)
                max_syl = cc.get("max_syllables", 18)
            else:
                min_syl = getattr(cc, "min_syllables", 6)
                max_syl = getattr(cc, "max_syllables", 18)
        corpus_vocab_mo: Optional[Set[str]] = None
        if cfg.corpus_lines:
            import re
            word_re = re.compile(r"[A-Za-z']+")
            corpus_vocab_mo = set()
            for line in cfg.corpus_lines[:2000]:
                corpus_vocab_mo.update(w.lower() for w in word_re.findall(line.lower()))
        mutation_config = {
            "theme_keywords": list(prompt_keywords or []),
            "min_syllables": min_syl,
            "max_syllables": max_syl,
            "corpus_vocab": corpus_vocab_mo,
        }

        attempts = 0
        accepted = 0
        max_attempts = (target_size - num_elites) * cfg.max_offspring_attempts
        while len(next_pop) < target_size - cfg.random_immigrants_per_gen and attempts < max_attempts:
            attempts += 1
            p1 = _tournament_select_multiobjective(population, cfg.tournament_k, ranks_by_pos, crowding_by_pos)
            p2 = _tournament_select_multiobjective(population, cfg.tournament_k, ranks_by_pos, crowding_by_pos)
            child = crossover(p1, p2, crossover_config)
            child = mutate(child, mutation_config, mut_weights)
            if not passes_constraints(child, _effective_constraint_config(cfg, prompt_keywords)):
                continue
            needs_scores = (cfg.min_fluency_accept > 0 or cfg.min_semantic_accept > 0 or cfg.min_lexical_accept > 0
                           or cfg.min_ngram_fluency_accept > 0 or (cfg.corpus_lines and cfg.min_ngram_fluency_accept == 0))
            if needs_scores:
                analyze_individual(child)
                child_scores = score_couplet(
                    child,
                    prompt_keywords=kw,
                    theme_string=theme_string,
                    semantic_scorer=semantic_scorer if cfg.use_embeddings else None,
                    embedding_weight=cfg.embedding_weight,
                    corpus_lines=cfg.corpus_lines,
                    use_lm_fluency=cfg.use_lm_fluency,
                    lm_fluency_weight=cfg.lm_fluency_weight,
                )
                if cfg.min_fluency_accept > 0 and child_scores.get("fluency", 0) < cfg.min_fluency_accept:
                    continue
                if cfg.min_semantic_accept > 0 and child_scores.get("semantic", 0) < cfg.min_semantic_accept:
                    continue
                if cfg.min_lexical_accept > 0 and child_scores.get("lexical_validity", 0.5) < cfg.min_lexical_accept:
                    continue
                ngram_floor_val = cfg.min_ngram_fluency_accept if cfg.min_ngram_fluency_accept > 0 else (NGRAM_FLOOR if cfg.corpus_lines else None)
                if ngram_floor_val is not None and child_scores.get("ngram_fluency", 0.5) < ngram_floor_val:
                    continue
            accepted += 1
            next_pop.append(child)
        prev_acceptance_rate = accepted / max(1, attempts)
        logger.info(f"  Mutation acceptance: {accepted}/{attempts} = {prev_acceptance_rate:.2%}")

        if cfg.random_immigrants_per_gen > 0 and immigrant_generator:
            immigrants = immigrant_generator(cfg.random_immigrants_per_gen)
            for ind in immigrants:
                if passes_constraints(ind, _effective_constraint_config(cfg, prompt_keywords)):
                    next_pop.append(ind)
                    if len(next_pop) >= target_size:
                        break

        while len(next_pop) < target_size and len(population) > len(next_pop):
            next_pop.append(population[len(next_pop)])

        population = next_pop[:target_size]

    # Final sort by Pareto rank + crowding + raw fitness
    ranks = _compute_pareto_ranks(population)
    crowding = _compute_crowding_by_rank(population, ranks)
    for i, ind in enumerate(population):
        ind.fitness = compute_fitness(
            ind.scores,
            weights,
            individual=ind,
            population=population,
            style_profile=cfg.style_profile,
            style_weight=cfg.style_weight if cfg.style_profile else 0.0,
        )
    population = [
        population[i]
        for i in sorted(
            range(len(population)),
            key=lambda idx: (ranks.get(idx, 999), -crowding.get(idx, 0.0), -(population[idx].fitness or 0)),
        )
    ]

    if run_logger:
        run_logger.flush()

    return population
