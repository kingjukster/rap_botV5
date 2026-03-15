"""
evo_rhyme/verse_weight_tuner.py

Outer-loop evolutionary tuner for verse (4-line) scoring weights.
Evolves a compact weight genome grouped as: Structure, Meaning, Flavor, Penalty.
Meta-fitness rewards coherence, fluency, readability and penalizes structure-only
optimization and harsh penalties on garbled/cliché/repetition.
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evo_rhyme.fitness import VERSE_DEFAULT_WEIGHTS, compute_verse_fitness

logger = logging.getLogger(__name__)

# Group definitions for interpretable weight evolution (per-term within groups)
STRUCTURE_KEYS: List[str] = [
    "rhyme_scheme_score",
    "internal_rhyme",
    "rhyme_chain_density",
    "global_rhyme_chain_score",
    "internal_chain_score",
    "rhyme_graph_density",
    "rhyme_graph_cluster_coeff",
    "rhyme_graph_chain_length",
    "syllable_balance",
    "flow_alignment",
    "flow_continuity_score",
]
MEANING_KEYS: List[str] = [
    "semantic",
    "coherence",
    "style_adherence",
    "prompt_adherence",
    "lexical_validity",
    "lm_fluency",
    "fluency",
]
FLAVOR_KEYS: List[str] = [
    "punchline",
    "novelty",
]
PENALTY_KEYS: List[str] = [
    "identical_line_penalty",
    "template_penalty",
    "repetition_penalty",
    "near_duplicate_penalty",
    "filler_line_penalty",
    "line_phrase_penalty",
    "corpus_overlap_penalty",
    "garbled_line_penalty",
    "cliche_penalty",
    "structural_repetition_penalty",
    "cross_verse_repetition_penalty",
]

EVOLVEABLE_POSITIVE_KEYS: List[str] = (
    STRUCTURE_KEYS + MEANING_KEYS + FLAVOR_KEYS
)
EVOLVEABLE_PENALTY_KEYS: List[str] = PENALTY_KEYS
EVOLVEABLE_KEYS: List[str] = EVOLVEABLE_POSITIVE_KEYS + EVOLVEABLE_PENALTY_KEYS


def genome_to_weights(genome: Dict[str, float]) -> Dict[str, float]:
    """
    Convert genome dict to full verse fitness weights.
    Fills missing keys from VERSE_DEFAULT_WEIGHTS. Normalizes positive weights
    so they sum to 1.0; penalties stay negative and are not rescaled.
    """
    weights = dict(VERSE_DEFAULT_WEIGHTS)
    for k in EVOLVEABLE_KEYS:
        if k in genome:
            v = genome[k]
            if k in EVOLVEABLE_PENALTY_KEYS:
                weights[k] = max(-0.5, min(-0.01, v))
            else:
                weights[k] = max(0.01, min(0.5, v))
    # Normalize positive weights to sum to 1.0
    positive_total = sum(
        weights[k] for k in EVOLVEABLE_POSITIVE_KEYS if k in weights and weights[k] > 0
    )
    if positive_total > 0:
        scale = 1.0 / positive_total
        for k in EVOLVEABLE_POSITIVE_KEYS:
            if k in weights and weights[k] > 0:
                weights[k] *= scale
    return weights


def weights_to_genome(weights: Dict[str, float]) -> Dict[str, float]:
    """Extract evolveable keys from a full weights dict."""
    genome: Dict[str, float] = {}
    for k in EVOLVEABLE_KEYS:
        if k in weights:
            genome[k] = weights[k]
    return genome


def random_genome() -> Dict[str, float]:
    """Random genome from VERSE_DEFAULT_WEIGHTS with small multiplicative noise."""
    genome = weights_to_genome(VERSE_DEFAULT_WEIGHTS)
    for k in genome:
        if k in EVOLVEABLE_PENALTY_KEYS:
            genome[k] = genome[k] * (0.7 + 0.6 * random.random())
        else:
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
    """Blend two genomes (uniform crossover / average)."""
    child: Dict[str, float] = {}
    for k in set(a.keys()) | set(b.keys()):
        default = VERSE_DEFAULT_WEIGHTS.get(
            k, -0.1 if k in EVOLVEABLE_PENALTY_KEYS else 0.1
        )
        va = a.get(k, default)
        vb = b.get(k, default)
        child[k] = (va + vb) / 2.0
    return child


# ---------------------------------------------------------------------------
# Meta-fitness and weight-set evaluation
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK_THEMES: List[List[str]] = [
    ["flow", "show"],
    ["pressure", "mask"],
    ["pain", "growth"],
]


def _mean_score(verses: List[Any], key: str) -> float:
    """Mean of score key across verses that have .scores."""
    if not verses:
        return 0.0
    vals = []
    for v in verses:
        sc = getattr(v, "scores", None) or {}
        if key in sc:
            vals.append(float(sc[key]))
    return sum(vals) / len(vals) if vals else 0.0


def compute_meta_fitness_verse(
    verses: List[Any],
    _weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Proxy meta-fitness for a set of verses produced under a candidate weight set.
    Rewards coherence, fluency, readability; penalizes garbled/cliché/repetition.
    Optionally penalizes structure-only optimization (high structure, low coherence).
    """
    if not verses:
        return 0.0
    coh = _mean_score(verses, "coherence")
    flu = _mean_score(verses, "fluency")
    lm_flu = _mean_score(verses, "lm_fluency")
    punch = _mean_score(verses, "punchline")
    nov = _mean_score(verses, "novelty")
    garbled = _mean_score(verses, "garbled_line_penalty")
    cliche = _mean_score(verses, "cliche_penalty")
    struct_rep = _mean_score(verses, "structural_repetition_penalty")
    # Structure-heavy score (rhyme/chain) vs meaning
    structure_mean = (
        _mean_score(verses, "rhyme_scheme_score")
        + _mean_score(verses, "rhyme_chain_density")
        + _mean_score(verses, "global_rhyme_chain_score")
    ) / 3.0
    # Penalize when structure is high but coherence is low (technically impressive but awkward)
    structure_over_coherence_penalty = 0.0
    if structure_mean > 0.6 and coh < 0.4:
        structure_over_coherence_penalty = 0.2

    meta = (
        0.25 * coh
        + 0.15 * flu
        + 0.15 * lm_flu
        + 0.10 * punch
        + 0.10 * nov
        - 0.15 * garbled
        - 0.10 * cliche
        - 0.10 * struct_rep
        - structure_over_coherence_penalty
    )
    return max(0.0, min(1.0, meta))


def evaluate_weight_set_verse(
    genome: Dict[str, float],
    benchmark_themes: List[List[str]],
    inner_population_size: int,
    inner_generations: int,
    top_n_per_theme: int,
    corpus_path: Optional[Path] = None,
    corpus_lines: Optional[List[str]] = None,
    corpus_vocab: Optional[set] = None,
    seed: Optional[int] = None,
    use_emitters: bool = False,
) -> Tuple[float, List[Any]]:
    """
    Run inner verse QD on each benchmark theme with the given weights.
    Return (meta_fitness, all_top_verses).
    """
    if seed is not None:
        random.seed(seed)
    from evo_rhyme.individual import VerseIndividual, analyze_verse_individual
    from evo_rhyme.constraints import passes_verse_constraints
    from evo_rhyme.verse_evolution import evolve_verse_qd, evolve_verse_qd_emitters, QDEvolutionConfig
    from evo_rhyme.population import create_initial_verse_population

    weights = genome_to_weights(genome)
    all_tops: List[Any] = []

    for theme_keywords in benchmark_themes:
        kw_set = set(w.lower() for w in theme_keywords)
        raw = create_initial_verse_population(
            theme_keywords=theme_keywords,
            size=inner_population_size,
            corpus_path=corpus_path,
            init_mode="mixed",
            scheme="AABB",
            num_lines=4,
        )
        population: List[VerseIndividual] = []
        for ind in raw:
            analyze_verse_individual(ind)
            if passes_verse_constraints(ind, None):
                population.append(ind)
        for _ in range(5):
            if len(population) >= inner_population_size:
                break
            extra = create_initial_verse_population(
                theme_keywords=theme_keywords,
                size=inner_population_size,
                corpus_path=corpus_path,
                init_mode="mixed",
                scheme="AABB",
                num_lines=4,
            )
            for ind in extra:
                if len(population) >= inner_population_size:
                    break
                analyze_verse_individual(ind)
                if passes_verse_constraints(ind, None):
                    population.append(ind)
        population = population[:inner_population_size]
        if not population:
            continue

        cfg = QDEvolutionConfig(
            population_size=inner_population_size,
            num_generations=inner_generations,
            num_elites=5,
            random_immigrants_per_gen=max(5, inner_population_size // 10),
            rhyme_scheme="AABB",
            theme_keywords=theme_keywords,
            num_lines=4,
            fitness_weights=weights,
            corpus_vocab=corpus_vocab,
            use_embeddings=False,
            fast_mode=True,
            graph_top_k=12,
            max_expensive_scoring_candidates=min(20, inner_population_size),
            use_emitters=use_emitters,
            enable_style_genome=False,
            enable_prompt_genome=False,
        )

        if use_emitters:
            archive, _ = evolve_verse_qd_emitters(
                population=population,
                config=cfg,
                immigrant_generator=None,
            )
        else:
            archive, _ = evolve_verse_qd(
                population=population,
                config=cfg,
                immigrant_generator=None,
            )
        tops = archive.top_k(top_n_per_theme)
        all_tops.extend(tops)

    meta = compute_meta_fitness_verse(all_tops, weights)
    return meta, all_tops


def evolve_verse_weights(
    population_size: int = 12,
    generations: int = 8,
    benchmark_themes: Optional[List[List[str]]] = None,
    inner_population_size: int = 50,
    inner_generations: int = 8,
    top_n_per_theme: int = 5,
    corpus_path: Optional[Path] = None,
    corpus_lines: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    use_emitters: bool = False,
    seed_weights_path: Optional[Path] = None,
) -> List[Tuple[Dict[str, float], float]]:
    """
    Outer-loop evolution of verse scoring weights.
    Returns list of (genome, meta_fitness) for top weight sets, best first.
    """
    themes = benchmark_themes or DEFAULT_BENCHMARK_THEMES
    corpus_vocab = None
    if corpus_lines and len(corpus_lines) >= 2:
        import re
        word_re = re.compile(r"[A-Za-z']+")
        corpus_vocab = set()
        for line in corpus_lines[:2000]:
            corpus_vocab.update(w.lower() for w in word_re.findall(line.lower()))

    if seed is not None:
        random.seed(seed)

    genomes: List[Tuple[Dict[str, float], float]] = []
    if seed_weights_path and seed_weights_path.exists():
        with seed_weights_path.open("r", encoding="utf-8") as f:
            import json
            data = json.load(f)
        base_weights = data.get("weights", data)
        base_genome = weights_to_genome(base_weights)
        for i in range(population_size):
            g = mutate_genome(copy.deepcopy(base_genome), mutation_rate=0.2) if i > 0 else copy.deepcopy(base_genome)
            meta, _ = evaluate_weight_set_verse(
                g, themes, inner_population_size, inner_generations, top_n_per_theme,
                corpus_path=corpus_path, corpus_lines=corpus_lines, corpus_vocab=corpus_vocab,
                seed=None, use_emitters=use_emitters,
            )
            genomes.append((g, meta))
    else:
        for _ in range(population_size):
            g = random_genome()
            meta, _ = evaluate_weight_set_verse(
                g, themes, inner_population_size, inner_generations, top_n_per_theme,
                corpus_path=corpus_path, corpus_lines=corpus_lines, corpus_vocab=corpus_vocab,
                seed=None, use_emitters=use_emitters,
            )
            genomes.append((g, meta))

    genomes.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        "Verse weight tuner: %d genomes, %d outer generations; benchmark: %d themes, inner %d pop x %d gen",
        population_size, generations, len(themes), inner_population_size, inner_generations,
    )

    for gen in range(generations - 1):
        num_elites = max(2, population_size // 4)
        next_gen: List[Tuple[Dict[str, float], float]] = [
            (copy.deepcopy(g), m) for g, m in genomes[:num_elites]
        ]
        while len(next_gen) < population_size:
            a, _ = genomes[random.randint(0, num_elites)]
            b, _ = genomes[random.randint(0, min(num_elites + 2, len(genomes) - 1))]
            child = crossover_genomes(a, b)
            child = mutate_genome(child, mutation_rate=0.25)
            meta, _ = evaluate_weight_set_verse(
                child, themes, inner_population_size, inner_generations, top_n_per_theme,
                corpus_path=corpus_path, corpus_lines=corpus_lines, corpus_vocab=corpus_vocab,
                seed=None, use_emitters=use_emitters,
            )
            next_gen.append((child, meta))
        next_gen.sort(key=lambda x: x[1], reverse=True)
        genomes = next_gen
        best_meta = genomes[0][1]
        logger.info("  Outer Gen %d/%d | best_meta=%.4f", gen + 2, generations, best_meta)

    return genomes
