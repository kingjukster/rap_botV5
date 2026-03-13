"""
evo_rhyme/weight_tuner.py

Outer-loop evolutionary weight tuner. Evolves fitness weights so that
the inner lyric evolution produces better outputs across a benchmark.

Meta-fitness uses signals that are harder to game: fluency, lexical_validity,
ngram_fluency, diversity, nonsense penalty, repetition penalty.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from evo_rhyme.evolution import EvolutionConfig, evolve
from evo_rhyme.fitness import DEFAULT_WEIGHTS
from evo_rhyme.individual import CoupletIndividual, analyze_individual
from evo_rhyme.population import create_mixed_population
from evo_rhyme.constraints import passes_constraints
from evo_rhyme.seed_generator import load_corpus_lines

logger = logging.getLogger(__name__)

# Benchmark prompts for meta-evaluation (theme keywords)
DEFAULT_BENCHMARK_PROMPTS: List[Tuple[str, ...]] = [
    ("flow", "show"),
    ("pressure", "mask"),
    ("pain", "growth"),
    ("money", "trust"),
    ("night", "light"),
]

# Keys we evolve (Phase 1: scoring weights + 2-3 penalty strengths)
EVOLVEABLE_POSITIVE_KEYS: List[str] = [
    "end_rhyme",
    "internal_rhyme",
    "rhyme_graph",
    "multisyllabic",
    "syllable_balance",
    "stress_alignment",
    "semantic",
    "fluency",
    "lexical_validity",
    "ngram_fluency",
    "novelty",
]

EVOLVEABLE_PENALTY_KEYS: List[str] = [
    "repetition_penalty",
    "rhyme_family_repetition_penalty",
]

# Fixed penalties (not evolved in Phase 1)
FIXED_PENALTIES: Dict[str, float] = {
    "weak_tail_penalty": -0.03,
    "identical_line_penalty": -0.20,
    "near_duplicate_penalty": -0.15,
    "template_penalty": -0.08,
    "corpus_overlap_penalty": -0.20,
    "theme_penalty": -0.18,
    "theme_word_repetition_penalty": -0.12,
}


def genome_to_weights(genome: Dict[str, float]) -> Dict[str, float]:
    """
    Convert genome dict to full fitness weights.
    Normalizes positive weights to sum to 1.0; penalties stay negative.
    """
    weights = dict(FIXED_PENALTIES)
    positive_total = 0.0
    for k in EVOLVEABLE_POSITIVE_KEYS:
        v = genome.get(k, DEFAULT_WEIGHTS.get(k, 0.1))
        v = max(0.01, min(0.5, v))  # clamp
        weights[k] = v
        positive_total += v
    if positive_total > 0:
        scale = 1.0 / positive_total
        for k in EVOLVEABLE_POSITIVE_KEYS:
            weights[k] *= scale
    for k in EVOLVEABLE_PENALTY_KEYS:
        v = genome.get(k, DEFAULT_WEIGHTS.get(k, -0.05))
        v = max(-0.5, min(-0.01, v))  # clamp penalty strength
        weights[k] = v
    return weights


def weights_to_genome(weights: Dict[str, float]) -> Dict[str, float]:
    """Extract evolveable keys from weights dict."""
    genome: Dict[str, float] = {}
    for k in EVOLVEABLE_POSITIVE_KEYS + EVOLVEABLE_PENALTY_KEYS:
        if k in weights:
            genome[k] = weights[k]
    return genome


def random_genome() -> Dict[str, float]:
    """Random genome from DEFAULT_WEIGHTS with small noise."""
    genome = weights_to_genome(DEFAULT_WEIGHTS)
    for k in genome:
        genome[k] = genome[k] * (0.7 + 0.6 * random.random())
    return genome


def mutate_weight(x: float, sigma: float = 0.08, is_penalty: bool = False) -> float:
    """Gaussian mutation. Penalties stay negative."""
    new = x + random.gauss(0, sigma)
    if is_penalty:
        return max(-0.5, min(-0.01, new))
    return max(0.01, min(0.5, new))


def mutate_genome(genome: Dict[str, float], mutation_rate: float = 0.3) -> Dict[str, float]:
    """Mutate a subset of genome entries."""
    child = copy.deepcopy(genome)
    for k in child:
        if random.random() < mutation_rate:
            child[k] = mutate_weight(
                child[k],
                sigma=0.08,
                is_penalty=k in EVOLVEABLE_PENALTY_KEYS,
            )
    return child


def crossover_genomes(
    a: Dict[str, float],
    b: Dict[str, float],
) -> Dict[str, float]:
    """Blend two genomes (uniform crossover)."""
    child: Dict[str, float] = {}
    for k in set(a.keys()) | set(b.keys()):
        va = a.get(k, DEFAULT_WEIGHTS.get(k, 0.1 if k not in EVOLVEABLE_PENALTY_KEYS else -0.05))
        vb = b.get(k, DEFAULT_WEIGHTS.get(k, 0.1 if k not in EVOLVEABLE_PENALTY_KEYS else -0.05))
        child[k] = (va + vb) / 2.0
    return child


def _compute_diversity(couplets: List[CoupletIndividual]) -> float:
    """Jaccard diversity of line1+line2 texts."""
    if len(couplets) < 2:
        return 1.0
    texts = [f"{c.line1} {c.line2}" for c in couplets]
    words_sets = [set(t.lower().split()) for t in texts]
    total_union = set()
    total_intersection = set(words_sets[0]) if words_sets else set()
    for ws in words_sets:
        total_union |= ws
        total_intersection &= ws
    if not total_union:
        return 1.0
    return 1.0 - len(total_intersection) / len(total_union)


def _nonsense_rate(couplets: List[CoupletIndividual]) -> float:
    """
    Heuristic nonsense rate: high rhyme_family_repetition_penalty in scores
    or very low lexical_validity / ngram_fluency.
    """
    if not couplets:
        return 0.0
    bad = 0
    for c in couplets:
        s = c.scores or {}
        lex = s.get("lexical_validity", 0.5)
        ngram = s.get("ngram_fluency", 0.5)
        rf_rep = s.get("rhyme_family_repetition_penalty", 0.0)
        if lex < 0.5 or ngram < 0.4 or rf_rep > 0.5:
            bad += 1
    return bad / len(couplets)


def _repetition_rate(couplets: List[CoupletIndividual]) -> float:
    """Average repetition_penalty from scores."""
    if not couplets:
        return 0.0
    scores = [c.scores or {} for c in couplets]
    reps = [s.get("repetition_penalty", 0.0) for s in scores]
    return sum(reps) / len(reps) if reps else 0.0


def compute_meta_fitness(
    couplets: List[CoupletIndividual],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Meta-fitness for a weight set based on its outputs.
    Uses signals that are harder to game than raw fitness.
    """
    if not couplets:
        return 0.0
    avg_fluency = sum((c.scores or {}).get("fluency", 0.5) for c in couplets) / len(couplets)
    avg_semantic = sum((c.scores or {}).get("semantic", 0.5) for c in couplets) / len(couplets)
    avg_lexical = sum((c.scores or {}).get("lexical_validity", 0.5) for c in couplets) / len(couplets)
    avg_ngram = sum((c.scores or {}).get("ngram_fluency", 0.5) for c in couplets) / len(couplets)
    avg_end_rhyme = sum((c.scores or {}).get("end_rhyme", 0.0) for c in couplets) / len(couplets)
    avg_internal = sum((c.scores or {}).get("internal_rhyme", 0.0) for c in couplets) / len(couplets)
    diversity = _compute_diversity(couplets)
    nonsense = _nonsense_rate(couplets)
    repetition = _repetition_rate(couplets)

    meta = (
        0.25 * avg_fluency
        + 0.15 * avg_semantic
        + 0.15 * avg_lexical
        + 0.10 * avg_ngram
        + 0.10 * avg_end_rhyme
        + 0.05 * avg_internal
        + 0.15 * diversity
        - 0.25 * nonsense
        - 0.15 * repetition
    )
    return max(0.0, min(1.0, meta))


def evaluate_weight_set(
    genome: Dict[str, float],
    benchmark_prompts: List[Tuple[str, ...]],
    inner_population_size: int,
    inner_generations: int,
    top_n_per_prompt: int,
    corpus_path: Optional[Path] = None,
    corpus_lines: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[float, List[CoupletIndividual]]:
    """
    Run inner lyric evolution on each benchmark prompt with the given weights.
    Return (meta_fitness, all_top_couplets).
    """
    if seed is not None:
        random.seed(seed)
    weights = genome_to_weights(genome)
    all_tops: List[CoupletIndividual] = []

    for prompt in benchmark_prompts:
        theme_keywords = list(prompt)
        kw_set = set(w.lower() for w in theme_keywords)

        raw = create_mixed_population(
            corpus_path=corpus_path,
            theme_keywords=theme_keywords,
            size=inner_population_size,
        )
        population: List[CoupletIndividual] = []
        for ind in raw:
            analyze_individual(ind)
            if passes_constraints(ind, None):
                population.append(ind)

        for _ in range(5):
            if len(population) >= inner_population_size:
                break
            extra = create_mixed_population(
                corpus_path=corpus_path,
                theme_keywords=theme_keywords,
                size=inner_population_size,
            )
            for ind in extra:
                if len(population) >= inner_population_size:
                    break
                analyze_individual(ind)
                if passes_constraints(ind, None):
                    population.append(ind)
        population = population[:inner_population_size]

        cfg = EvolutionConfig(
            population_size=inner_population_size,
            num_elites=5,
            random_immigrants_per_gen=5,
            fitness_weights=weights,
            corpus_lines=corpus_lines,
        )

        population = evolve(
            population,
            generations=inner_generations,
            config=cfg,
            prompt_keywords=kw_set,
            immigrant_generator=lambda n: create_mixed_population(
                corpus_path=corpus_path,
                theme_keywords=theme_keywords,
                size=n,
            ),
        )

        tops = population[:top_n_per_prompt]
        all_tops.extend(tops)

    meta = compute_meta_fitness(all_tops, weights)
    return meta, all_tops


def evolve_weights(
    population_size: int = 12,
    generations: int = 8,
    benchmark_prompts: Optional[List[Tuple[str, ...]]] = None,
    inner_population_size: int = 40,
    inner_generations: int = 8,
    top_n_per_prompt: int = 4,
    corpus_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> List[Tuple[Dict[str, float], float]]:
    """
    Outer-loop evolution of fitness weights.
    Returns list of (best_genome, meta_fitness) for top weight sets.
    """
    prompts = benchmark_prompts or DEFAULT_BENCHMARK_PROMPTS
    corpus_lines = None
    if corpus_path and corpus_path.exists():
        corpus_lines = load_corpus_lines(corpus_path)
        if corpus_lines:
            corpus_lines = corpus_lines[:2000]

    # Initial population of genomes
    if seed is not None:
        random.seed(seed)
    genomes: List[Tuple[Dict[str, float], float]] = []
    for _ in range(population_size):
        g = random_genome()
        meta, _ = evaluate_weight_set(
            g,
            prompts,
            inner_population_size,
            inner_generations,
            top_n_per_prompt,
            corpus_path=corpus_path,
            corpus_lines=corpus_lines,
            seed=None,
        )
        genomes.append((g, meta))

    genomes.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Weight tuner: {population_size} genomes, {generations} outer generations")
    logger.info(f"Benchmark: {len(prompts)} prompts, inner {inner_population_size} pop x {inner_generations} gen")

    for gen in range(generations - 1):
        # Elites
        num_elites = max(2, population_size // 4)
        next_gen: List[Tuple[Dict[str, float], float]] = [
            (copy.deepcopy(g), m) for g, m in genomes[:num_elites]
        ]

        # Offspring: crossover + mutate
        while len(next_gen) < population_size:
            a, _ = genomes[random.randint(0, num_elites)]
            b, _ = genomes[random.randint(0, min(num_elites + 2, len(genomes) - 1))]
            child = crossover_genomes(a, b)
            child = mutate_genome(child, mutation_rate=0.25)
            meta, _ = evaluate_weight_set(
                child,
                prompts,
                inner_population_size,
                inner_generations,
                top_n_per_prompt,
                corpus_path=corpus_path,
                corpus_lines=corpus_lines,
                seed=None,
            )
            next_gen.append((child, meta))

        next_gen.sort(key=lambda x: x[1], reverse=True)
        genomes = next_gen
        best_meta = genomes[0][1]
        logger.info(f"  Outer Gen {gen + 2}/{generations} | best_meta={best_meta:.4f}")

    return genomes
