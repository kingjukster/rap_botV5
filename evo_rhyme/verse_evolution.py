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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from evo_rhyme.constraints import passes_verse_constraints
from evo_rhyme.archive import (
    MAPElitesArchive,
    default_verse_dimensions,
    create_verse_archive,
    compact_style_dimensions,
    style_chain_dimensions,
    ultra_compact_dimensions,
)
from evo_rhyme.line_archive import ScoredLine, LineArchive
from evo_rhyme.line_evolution import LineEvolutionConfig, evolve_lines, score_line
from evo_rhyme.verse_builder import build_verse_batch
from evo_rhyme.repro import append_jsonl, run_provenance_dict
from evo_rhyme.scoring.novelty import NoveltyArchive, embed_texts, compute_verse_novelty
from evo_rhyme.fitness import (
    OBJECTIVE_KEYS,
    VERSE_DEFAULT_WEIGHTS,
    compute_parent_improvement,
    compute_verse_fitness,
    score_vector,
    score_verse,
    score_verses_batch,
    verse_score_cache_snapshot,
)
from evo_rhyme.selection import (
    compute_population_objectives,
    crowding_distance,
    pareto_elitism,
    pareto_rank,
    pareto_tournament_select,
)
from evo_rhyme.individual import (
    CoupletIndividual,
    VerseIndividual,
    analyze_individual,
    analyze_verse_individual,
)
from evo_rhyme.mutation import MUTATION_WEIGHTS, mutate
from evo_rhyme.phonetics import tokenize_line, syllable_count_line
from evo_rhyme.template_grammar import get_template_pool

logger = logging.getLogger(__name__)


def _get_active_tracer():
    """Return the active OperatorTracer (if any) without creating import cycles."""
    try:
        from evo_rhyme.operator_telemetry import get_operator_tracer
        return get_operator_tracer()
    except Exception:
        return None


def _write_verse_lineage(run_id: int, child_id: int, parents, gen: int) -> None:
    """Best-effort DB lineage insert for verse QD offspring."""
    if run_id <= 0 or child_id <= 0:
        return
    try:
        from evo_rhyme import db as _db
        if not _db.db_enabled():
            return
        seen = set()
        for p in (parents if isinstance(parents, (list, tuple)) else [parents]):
            pid = (p.metadata or {}).get("db_id") if hasattr(p, "metadata") else None
            if pid and pid > 0 and pid not in seen:
                seen.add(pid)
                _db.insert_lineage(child_id, pid, "crossover+mutate", gen)
    except Exception as e:
        logger.debug("verse lineage write failed: %s", e)


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
    use_niching: bool = True
    corpus_lines: Optional[list] = None


@dataclass
class QDEvolutionConfig:
    """Configuration for quality-diversity verse evolution."""
    population_size: int = 120
    num_generations: int = 100
    num_elites: int = 5
    tournament_k: int = 4
    random_immigrants_per_gen: int = 20
    rhyme_scheme: str = "AABB"
    theme_keywords: List[str] = field(default_factory=list)
    num_lines: int = 4

    lm_mutation_budget_per_gen: int = 20

    archive_dims: Optional[List] = None

    min_fluency: float = 0.4
    min_lm_fluency: float = 0.4
    min_coherence: float = 0.30
    min_semantic: float = 0.0
    min_rhyme_scheme_score: float = 0.10

    max_offspring_attempts: int = 200
    crossover_rate: float = 0.6

    output_dir: Optional[str] = None
    run_id: Optional[int] = None  # DB run ID when RAPBOT_USE_DB=1
    initial_archive: Optional[Any] = None  # Pre-loaded archive for resume

    min_syllables: int = 6
    max_syllables: int = 18

    corpus_vocab: Optional[set] = None
    use_embeddings: bool = False
    embedding_weight: float = 0.40

    use_emitters: bool = True
    novelty_weight: float = 0.3
    allowed_schemes: List[str] = field(default_factory=lambda: ["AABB", "ABAB", "ABBA", "ABCB", "AABA", "AAAA"])
    archive_mode: str = "compact_style"
    curriculum_switch_gen: int = 20
    coverage_target: Optional[float] = None  # e.g. 0.5 for 50% coverage; enables adaptive emitter boost

    # Archive: break ties with embedding novelty when fitness is equal (optional)
    archive_novelty_tiebreak: bool = False
    # Crossover: prefer semantically similar parent pairs (reduces mixed-topic crossover)
    semantic_crossover_pairing: bool = False

    # Runtime-efficiency controls
    fast_mode: bool = True
    graph_top_k: int = 24
    max_expensive_scoring_candidates: int = 40
    graph_edge_mode: str = "phonetic"

    # Prompt/style genome controls
    enable_style_genome: bool = True
    enable_prompt_genome: bool = True
    prompt_llm_fraction: float = 0.0
    prompt_dedup_enabled: bool = True
    proposer_config: Optional[Dict[str, Any]] = None

    # Scoring weight override (for evolutionary weight tuning)
    fitness_weights: Optional[Dict[str, float]] = None

    # Two-tier line evolution
    line_population_size: int = 1500
    line_generations_per_verse_gen: int = 3
    line_lm_seed_count: int = 80
    composed_offspring_ratio: float = 0.5
    max_lines_per_rhyme_group: int = 200
    line_lm_mutation_budget: int = 15

    # Optional instrumentation probes
    enable_controllability_probes: bool = False
    controllability_probe_every: int = 5
    controllability_probe_batch_size: int = 8

    # Hierarchical evolution: structural mutations at 4-bar level
    use_structural_mutations: bool = False

    # Optional per-operator mutation weights (keys match MUTATION_WEIGHTS); None = use defaults
    mutation_weights: Optional[Dict[str, float]] = None

    # Early stop: stop after K generations without best-fitness improvement (disabled if None)
    early_stop_stagnant_gens: Optional[int] = None
    early_stop_min_delta: float = 0.005

    # Tighten quality floors when best fitness stagnates (anti-plateau curriculum)
    curriculum_tighten_on_stagnation: bool = True
    curriculum_tighten_after_stagnant_gens: int = 6
    curriculum_tighten_fluency_step: float = 0.03
    curriculum_tighten_coherence_step: float = 0.03
    curriculum_tighten_fluency_cap: float = 0.55
    curriculum_tighten_coherence_cap: float = 0.42


def _cull_near_duplicates(
    population: List[VerseIndividual],
    max_pop: int,
) -> List[VerseIndividual]:
    """Remove near-duplicate verses, keeping highest-fitness copy.
    Uses first-3-words-per-line fingerprint as a fast proxy."""
    seen_fps: set = set()
    result: list = []
    for ind in sorted(population, key=lambda x: x.fitness or 0.0, reverse=True):
        fp = tuple(" ".join(l.lower().split()[:3]) for l in ind.lines)
        if fp not in seen_fps:
            seen_fps.add(fp)
            result.append(ind)
    return result[:max_pop]


def verse_crossover(
    parent1: VerseIndividual,
    parent2: VerseIndividual,
    config: Optional[Any] = None,
) -> VerseIndividual:
    """
    Crossover two verse parents. Three strategies:
    1. Half swap: lines 1-2 from A, 3-4 from B (preserves rhyming couplets)
    2. Single-line swap: replace one line from A with the same-position line from B
    3. Phrase-slice: swap phrase slices when structure matches
    """
    from evo_rhyme.operator_telemetry import record_operator_event

    cfg = config or {}
    try_phrase_slice = cfg.get("phrase_slice", False)

    if try_phrase_slice and random.random() < 0.2:
        child = _verse_phrase_slice_crossover(parent1, parent2)
        if child is not None:
            record_operator_event(
                scope="verse", operator_kind="crossover",
                operator_name="phrase_slice", succeeded=True,
            )
            return child

    scheme = cfg.get("scheme", "AABB") if isinstance(cfg, dict) else "AABB"

    # Rhyme-scheme-aware crossover groups
    # AABB: lines (0,1) and (2,3) rhyme together
    # ABAB: lines (0,2) and (1,3) rhyme together
    # ABBA: lines (0,3) and (1,2) rhyme together
    # ABCB: lines (1,3) rhyme together, 0 and 2 are independent
    if scheme == "ABAB":
        rhyme_pairs = [(0, 2), (1, 3)]
    elif scheme == "ABBA":
        rhyme_pairs = [(0, 3), (1, 2)]
    elif scheme == "ABCB":
        rhyme_pairs = [(1, 3)]
    else:
        rhyme_pairs = [(0, 1), (2, 3)]

    op_name: str = "half_swap"
    r = random.random()
    if r < 0.45:
        # Scheme-aware swap: keep rhyme-paired lines from same parent
        if random.random() < 0.5:
            lines = list(parent1.lines)
            for a, b in rhyme_pairs[len(rhyme_pairs) // 2:]:
                lines[a] = parent2.lines[a]
                lines[b] = parent2.lines[b]
        else:
            lines = list(parent2.lines)
            for a, b in rhyme_pairs[:len(rhyme_pairs) // 2 + 1]:
                lines[a] = parent1.lines[a]
                lines[b] = parent1.lines[b]
        op_name = "scheme_aware_swap"
    elif r < 0.65:
        # Half swap (preserves rhyming couplets within AABB)
        if random.random() < 0.5:
            lines = parent1.lines[:2] + parent2.lines[2:]
        else:
            lines = parent2.lines[:2] + parent1.lines[2:]
    elif r < 0.85:
        # Single-line swap (minimal disruption, prefer non-rhyming line)
        op_name = "single_line_swap"
        idx = random.randint(0, 3)
        lines = list(parent1.lines)
        lines[idx] = parent2.lines[idx]
    else:
        # Best-of-each: pick best-scoring parent's line at each position
        op_name = "best_of_each"
        lines = []
        for i in range(4):
            s1 = parent1.fitness or 0
            s2 = parent2.fitness or 0
            if s1 >= s2:
                lines.append(parent1.lines[i] if random.random() < 0.7 else parent2.lines[i])
            else:
                lines.append(parent2.lines[i] if random.random() < 0.7 else parent1.lines[i])

    record_operator_event(
        scope="verse", operator_kind="crossover",
        operator_name=op_name, succeeded=True,
    )
    return VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata={},
    )


def select_crossover_parents(
    parents: List[VerseIndividual],
    *,
    semantic_pairing: bool,
) -> Tuple[VerseIndividual, VerseIndividual]:
    """Pick two parents for crossover; optionally prefer semantically similar pair."""
    if len(parents) < 2:
        p = parents[0]
        return p, p
    if not semantic_pairing:
        p1 = random.choice(parents)
        pool = [x for x in parents if x is not p1] or parents
        return p1, random.choice(pool)
    try:
        import numpy as np

        texts = [" ".join(p.lines) for p in parents]
        embs = embed_texts(texts)
        idx1 = random.randint(0, len(parents) - 1)
        p1 = parents[idx1]
        e1 = embs[idx1]
        n1 = float(np.linalg.norm(e1)) + 1e-9
        best_j: Optional[int] = None
        best_sim = -2.0
        for j in range(len(parents)):
            if j == idx1:
                continue
            e2 = embs[j]
            sim = float(np.dot(e1, e2) / (n1 * (float(np.linalg.norm(e2)) + 1e-9)))
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is not None:
            return p1, parents[best_j]
    except Exception:
        logger.debug("select_crossover_parents semantic_pairing failed", exc_info=True)
    p1 = random.choice(parents)
    pool = [x for x in parents if x is not p1] or parents
    return p1, random.choice(pool)


def verse_archive_add_batch(
    archive: MAPElitesArchive,
    individuals: List[VerseIndividual],
    *,
    min_coherence: float,
) -> int:
    """Insert into MAP-Elites archive, skipping individuals below coherence floor."""
    n = 0
    for ind in individuals:
        if min_coherence > 0:
            coh = (ind.scores or {}).get("coherence")
            if coh is not None and float(coh) < float(min_coherence):
                continue
        if archive.add(ind):
            n += 1
    return n


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


def _lm_verse_rewrite(
    individual: VerseIndividual,
    config: Optional[Any] = None,
    lm_budget: Optional[Dict[str, int]] = None,
) -> Optional[VerseIndividual]:
    """Full 4-line LM rewrite preserving rhyme scheme and theme."""
    try:
        from evo_rhyme.mutation import get_rewriter
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        theme_keywords = []
        if config and isinstance(config, dict):
            theme_keywords = config.get("theme_keywords", [])
        theme = ", ".join(theme_keywords) if theme_keywords else "hip-hop"
        verse_text = "\n".join(individual.lines)

        prompt = (
            f'Rewrite this entire rap verse with fresh imagery and structure.\n'
            f'Keep the AABB rhyme scheme (lines 1-2 rhyme, lines 3-4 rhyme).\n'
            f'Theme: {theme}\n'
            f'Each line should be 8-16 syllables.\n\n'
            f'Original verse:\n{verse_text}\n\n'
            f'Write ONE rewritten verse (4 lines). Output ONLY the 4 lines.'
        )

        from evo_rhyme.lm_rewriter import _cache_key
        key = _cache_key(verse_text, "verse_rewrite", (theme,))
        raw = rewriter._call_lm(prompt, key)
        if lm_budget:
            lm_budget["remaining"] = lm_budget.get("remaining", 0) - 1

        new_lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        if len(new_lines) != 4:
            return None

        return VerseIndividual(
            lines=new_lines,
            features=None,
            scores=None,
            fitness=None,
            metadata={"origin": "lm_verse_rewrite"},
        )
    except Exception:
        logger.warning("_lm_verse_rewrite failed", exc_info=True)
        return None


def _lm_repair_couplet(
    couplet: CoupletIndividual,
    config: Optional[Any] = None,
    lm_budget: Optional[Dict[str, int]] = None,
) -> Optional[CoupletIndividual]:
    """Repair a constraint-violating couplet via LM."""
    try:
        from evo_rhyme.mutation import get_rewriter
        rewriter = get_rewriter(config if isinstance(config, dict) else None)
        min_syl = config.get("min_syllables", 6) if isinstance(config, dict) else 6
        max_syl = config.get("max_syllables", 18) if isinstance(config, dict) else 18

        tokens2 = tokenize_line(couplet.line2)
        rhyme_target = tokens2[-1] if tokens2 else "night"

        repaired_lines = rewriter.repair(
            couplet.line1, rhyme_target, (min_syl, max_syl),
        )
        if lm_budget:
            lm_budget["remaining"] = lm_budget.get("remaining", 0) - 1

        if repaired_lines:
            from evo_rhyme.individual import CoupletIndividual as CI
            return CI(line1=repaired_lines[0], line2=couplet.line2)
        return None
    except Exception:
        logger.warning("_lm_repair_couplet failed", exc_info=True)
        return None


def verse_mutate(
    individual: VerseIndividual,
    config: Optional[Any] = None,
    weights: Optional[Dict[str, float]] = None,
    constraint_config: Optional[Any] = None,
    lm_budget: Optional[Dict[str, int]] = None,
) -> VerseIndividual:
    """Mutate verse: 80% single couplet, 15% both couplets, 5% verse LM rewrite.
    When use_structural_mutations=True in config: 15% swap_couplets, 10% rewrite_transition."""
    from evo_rhyme.constraints import passes_constraints
    from evo_rhyme.structural_mutations import swap_couplets, rewrite_transition_line
    from evo_rhyme.operator_telemetry import record_operator_event

    cfg = config if isinstance(config, dict) else {}
    use_structural = cfg.get("use_structural_mutations", False)

    r = random.random()

    # Structural mutations at 4-bar level (when enabled)
    if use_structural and len(individual.lines) == 4:
        if r < 0.15:
            record_operator_event(
                scope="verse", operator_kind="mutation",
                operator_name="swap_couplets", succeeded=True,
            )
            return swap_couplets(individual)
        if r < 0.25 and lm_budget and lm_budget.get("remaining", 0) > 0:
            result = rewrite_transition_line(individual, boundary_idx=1, lm_budget=lm_budget)
            if result is not None:
                record_operator_event(
                    scope="verse", operator_kind="mutation",
                    operator_name="rewrite_transition", succeeded=True,
                )
                return result
        if r < 0.25:
            r = random.random()  # fall through to couplet mutation

    # 5% chance: whole-verse LM rewrite
    if r < 0.05 and lm_budget and lm_budget.get("remaining", 0) > 0:
        result = _lm_verse_rewrite(individual, config, lm_budget)
        if result is not None:
            record_operator_event(
                scope="verse", operator_kind="mutation",
                operator_name="lm_verse_rewrite", succeeded=True,
            )
            return result

    # 20% chance: mutate both couplets, 15% chance: multi-point (2 passes)
    multi_point = r < 0.15
    if r < 0.20:
        pair_indices = [0, 1]
    else:
        pair_indices = [random.randint(0, 1)]

    lines = list(individual.lines)
    for pair_idx in pair_indices:
        line1 = lines[pair_idx * 2]
        line2 = lines[pair_idx * 2 + 1]
        couplet = CoupletIndividual(line1=line1, line2=line2)
        mutated = mutate(couplet, config, weights, lm_budget=lm_budget)
        if multi_point:
            mutated = mutate(mutated, config, weights, lm_budget=lm_budget)
        if passes_constraints(mutated, constraint_config):
            lines[pair_idx * 2] = mutated.line1
            lines[pair_idx * 2 + 1] = mutated.line2
        else:
            # Try LM repair instead of discarding
            if lm_budget and lm_budget.get("remaining", 0) > 0:
                repaired = _lm_repair_couplet(mutated, config, lm_budget)
                if repaired and passes_constraints(repaired, constraint_config):
                    lines[pair_idx * 2] = repaired.line1
                    lines[pair_idx * 2 + 1] = repaired.line2

    mutated = VerseIndividual(
        lines=lines,
        features=None,
        scores=None,
        fitness=None,
        metadata=dict(individual.metadata),
    )

    from evo_rhyme.fitness import _score_verse_garbled_line_penalty
    if _score_verse_garbled_line_penalty(mutated) > 0.25:
        return individual

    return mutated


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
    target_size = cfg.population_size
    scheme = cfg.rhyme_scheme or "AABB"

    semantic_scorer: Optional[Any] = None
    if cfg.use_embeddings:
        try:
            from evo_rhyme.siamese_scorer import SiameseRhymeScorer, SIAMESE_MODEL_DIR
            semantic_scorer = SiameseRhymeScorer(str(SIAMESE_MODEL_DIR), device="cuda")
            logger.info("Loaded SiameseRhymeScorer for verse semantic scoring (CUDA)")
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
        _tracer = _get_active_tracer()
        if _tracer is not None:
            _tracer.gen = gen

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
                corpus_lines=cfg.corpus_lines,
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

        # Elites -- cap at 30% of surviving population to prevent premature convergence
        num_elites = min(cfg.num_elites, max(1, len(population) * 3 // 10))
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

        # Scale immigrants when population is small (< half target)
        immigrants_this_gen = cfg.random_immigrants_per_gen
        if len(population) < cfg.population_size // 2:
            immigrants_this_gen = cfg.random_immigrants_per_gen * 2

        # Offspring
        attempts = 0
        accepted = 0
        max_attempts = (target_size - num_elites) * cfg.max_offspring_attempts
        while len(next_pop) < target_size - immigrants_this_gen and attempts < max_attempts:
            attempts += 1
            p1 = _tournament_select_verse(population, cfg.tournament_k)
            p2 = _tournament_select_verse(population, cfg.tournament_k)
            child = verse_crossover(p1, p2, crossover_config)
            child = verse_mutate(child, mutation_config, mut_weights, constraint_config=cfg.constraint_config)
            if passes_verse_constraints(child, cfg.constraint_config):
                accepted += 1
                next_pop.append(child)

        if immigrant_generator and immigrants_this_gen > 0:
            immigrants = immigrant_generator(immigrants_this_gen)
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


def evolve_from_seed_verse(
    seed_lines: List[str],
    generations: int = 5,
    population_size: int = 25,
    objective_weights: Optional[Dict[str, float]] = None,
    scheme: str = "AABB",
    prompt_keywords: Optional[Set[str]] = None,
    edit_aggressiveness: float = 0.5,
) -> List[VerseIndividual]:
    """
    Improve an existing verse by evolving from a seed. Population is built by
    mutating the seed; no random immigrants. Returns population sorted by fitness (best first).

    edit_aggressiveness: 0 = minimal mutation, 1 = more aggressive (not yet wired to mutation rate).
    """
    if len(seed_lines) != 4:
        raise ValueError("seed_lines must be 4 lines")
    seed = VerseIndividual(
        lines=list(seed_lines),
        features=None,
        scores=None,
        fitness=None,
        metadata={"origin": "seed"},
    )
    mutation_config = {
        "theme_keywords": list(prompt_keywords or []),
        "min_syllables": 6,
        "max_syllables": 18,
    }
    population: List[VerseIndividual] = [seed]
    for _ in range(population_size - 1):
        mutated = verse_mutate(seed, mutation_config, MUTATION_WEIGHTS, constraint_config=None)
        population.append(mutated)
    cfg = VerseEvolutionConfig(
        population_size=population_size,
        num_elites=min(3, population_size // 5),
        random_immigrants_per_gen=0,
        fitness_weights=objective_weights or VERSE_DEFAULT_WEIGHTS,
        rhyme_scheme=scheme,
        output_dir=None,
    )
    return evolve_verse_population(
        population,
        generations,
        config=cfg,
        prompt_keywords=prompt_keywords,
        immigrant_generator=None,
    )


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
            "archive_mode": config.archive_mode,
            "fast_mode": config.fast_mode,
            "graph_top_k": config.graph_top_k,
            "max_expensive_scoring_candidates": config.max_expensive_scoring_candidates,
            "enable_style_genome": config.enable_style_genome,
            "enable_prompt_genome": config.enable_prompt_genome,
            "prompt_llm_fraction": config.prompt_llm_fraction,
        }
        seed_info = getattr(config, "seed_info", None)
        if seed_info:
            data["seed_info"] = seed_info
        if extra:
            data.update(extra)
        data["provenance"] = run_provenance_dict()
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
        row = {
            "gen": gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
        }
        self.score_history.append(row)
        try:
            append_jsonl(self.run_dir / "generation_metrics.jsonl", dict(row))
        except Exception as e:
            logger.warning("generation_metrics.jsonl append failed: %s", e)
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
        man_path = self.run_dir / "run_manifest.json"
        with man_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "provenance": run_provenance_dict(),
                    "score_history_rows": len(self.score_history),
                },
                f,
                indent=2,
            )


# ---------------------------------------------------------------------------
# Quality-Diversity (MAP-Elites + Pareto) evolution
# ---------------------------------------------------------------------------


@dataclass
class VerseQDRunLogger:
    """Write QD verse run artifacts to run_dir.
    When run_dir is None but run_id is set, only DB persistence is performed
    (used by control experiments with --db but without --runs-dir).
    """

    run_dir: Optional[Path] = None
    score_history: List[Dict[str, Any]] = field(default_factory=list)
    run_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(
        self,
        config: QDEvolutionConfig,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.run_dir is None:
            return
        data: Dict[str, Any] = {
            "population_size": config.population_size,
            "num_generations": config.num_generations,
            "num_elites": config.num_elites,
            "tournament_k": config.tournament_k,
            "random_immigrants_per_gen": config.random_immigrants_per_gen,
            "rhyme_scheme": config.rhyme_scheme,
            "theme_keywords": config.theme_keywords,
            "num_lines": config.num_lines,
            "lm_mutation_budget_per_gen": config.lm_mutation_budget_per_gen,
            "min_fluency": config.min_fluency,
            "min_semantic": config.min_semantic,
            "crossover_rate": config.crossover_rate,
            "max_offspring_attempts": config.max_offspring_attempts,
            "min_syllables": config.min_syllables,
            "max_syllables": config.max_syllables,
            "use_embeddings": config.use_embeddings,
            "embedding_weight": config.embedding_weight,
            "archive_mode": config.archive_mode,
            "curriculum_switch_gen": config.curriculum_switch_gen,
            "graph_edge_mode": config.graph_edge_mode,
            "graph_top_k": config.graph_top_k,
            "max_expensive_scoring_candidates": config.max_expensive_scoring_candidates,
            "prompt_llm_fraction": config.prompt_llm_fraction,
            "enable_controllability_probes": config.enable_controllability_probes,
        }
        seed_info = getattr(config, "seed_info", None)
        if seed_info:
            data["seed_info"] = seed_info
        if extra:
            data.update(extra)
        data["provenance"] = run_provenance_dict()
        path = self.run_dir / "config.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def log_generation(
        self,
        gen: int,
        archive_coverage: float,
        best_fitness: float,
        mean_fitness: float,
        occupied_niches: int,
        acceptance_rate: Optional[float] = None,
        runtime: Optional[Dict[str, Any]] = None,
        archive: Optional[MAPElitesArchive] = None,
    ) -> None:
        row = {
            "generation": gen,
            "archive_coverage": archive_coverage,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "occupied_niches": occupied_niches,
            "acceptance_rate": acceptance_rate,
        }
        if runtime:
            row.update(runtime)
        if archive is not None:
            try:
                row["occupancy_diversity"] = archive.occupancy_diversity_stats()
            except Exception:
                logger.debug("occupancy_diversity_stats failed", exc_info=True)
        self.score_history.append(row)
        if self.run_dir is not None:
            try:
                append_jsonl(self.run_dir / "generation_metrics.jsonl", dict(row))
            except Exception as e:
                logger.warning("generation_metrics.jsonl append failed: %s", e)
        if self.run_id is not None and self.run_id > 0:
            try:
                from evo_rhyme import db as _db
                if _db.db_enabled():
                    _db.insert_generation(
                        self.run_id, gen, best_fitness, mean_fitness,
                        archive_coverage, acceptance_rate,
                        {"occupied_niches": occupied_niches, "runtime": runtime},
                    )
                    # Insert top candidates each gen so runs page shows progress even if run crashes later
                    if archive is not None:
                        scheme = "AABB"
                        for ind in archive.top_k(5):
                            if (ind.metadata or {}).get("db_id"):
                                continue
                            ct = f"verse{len(ind.lines)}"
                            cid = _db.insert_candidate(
                                self.run_id, gen, ct, scheme,
                                ind.lines, ind.fitness or 0.0, ind.scores,
                            )
                            if cid and cid > 0:
                                ind.metadata = ind.metadata or {}
                                ind.metadata["db_id"] = cid
            except Exception as e:
                logger.warning("DB log_generation failed: %s", e)

    def flush(self, archive: MAPElitesArchive) -> None:
        top = archive.top_k(50)
        if self.run_dir is not None:
            csv_path = self.run_dir / "score_history.csv"
            if self.score_history:
                with csv_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=sorted({k for row in self.score_history for k in row.keys()}),
                    )
                    writer.writeheader()
                    writer.writerows(self.score_history)
            archive_path = self.run_dir / "archive.json"
            with archive_path.open("w", encoding="utf-8") as f:
                json.dump(archive.to_json(), f, indent=2)
            top_path = self.run_dir / "top_candidates.json"
            candidates = [
                {"lines": ind.lines, "fitness": ind.fitness, "scores": ind.scores}
                for ind in top
            ]
            with top_path.open("w", encoding="utf-8") as f:
                json.dump(candidates, f, indent=2)
            man_path = self.run_dir / "run_manifest.json"
            with man_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provenance": run_provenance_dict(),
                        "score_history_rows": len(self.score_history),
                        "archive_occupied": archive.occupied_niches(),
                        "archive_total_niches": archive.total_niches(),
                    },
                    f,
                    indent=2,
                )
        if self.run_id is not None and self.run_id > 0:
            try:
                from evo_rhyme import db as _db
                if _db.db_enabled():
                    if self.score_history:
                        last_gen = max(r.get("generation", 0) for r in self.score_history)
                        scheme = "AABB"
                        for ind in top:
                            ct = f"verse{len(ind.lines)}"
                            _db.insert_candidate(
                                self.run_id, last_gen, ct, scheme,
                                ind.lines, ind.fitness or 0.0, ind.scores,
                            )
                    # Sync full archive to archive_cells (batched for efficiency)
                    cells = [
                        ("_".join(str(c) for c in coord), ind.lines, ind.fitness or 0.0, ind.scores)
                        for coord, ind in archive.best_per_niche().items()
                    ]
                    _db.upsert_archive_cells_batch(self.run_id, cells)
            except Exception as e:
                logger.warning("DB insert candidates/archive failed: %s", e)


def evolve_verse_qd(
    population: List[VerseIndividual],
    config: QDEvolutionConfig,
    immigrant_generator: Optional[Callable[[], VerseIndividual]] = None,
    on_generation: Optional[Callable[[int, MAPElitesArchive, List[VerseIndividual]], None]] = None,
) -> Tuple[MAPElitesArchive, List[VerseIndividual]]:
    """Run quality-diversity evolution on verse population.

    Uses MAP-Elites archive for diversity preservation and Pareto-based
    selection for multi-objective optimization.

    Args:
        population: Initial population of VerseIndividuals.
        config: QD evolution configuration.
        immigrant_generator: Optional callable that produces random immigrants.
        on_generation: Optional callback(gen_idx, archive, population) for logging.

    Returns:
        Tuple of (archive, final_population).
    """
    scheme = config.rhyme_scheme or "AABB"
    weights = config.fitness_weights or VERSE_DEFAULT_WEIGHTS
    mut_weights = config.mutation_weights if getattr(config, "mutation_weights", None) else MUTATION_WEIGHTS

    theme_keywords = config.theme_keywords or []
    kw = set(w.lower() for w in theme_keywords) if theme_keywords else None
    theme_string = " ".join(theme_keywords) if theme_keywords else None

    mutation_config: Dict[str, Any] = {
        "theme_keywords": list(theme_keywords),
        "min_syllables": config.min_syllables,
        "max_syllables": config.max_syllables,
        "use_structural_mutations": getattr(config, "use_structural_mutations", False),
    }
    if config.corpus_vocab:
        mutation_config["corpus_vocab"] = config.corpus_vocab

    constraint_config: Dict[str, Any] = {
        "min_syllables": config.min_syllables,
        "max_syllables": config.max_syllables,
    }
    if theme_keywords:
        constraint_config["prompt_keywords"] = list(theme_keywords)
    crossover_config: Dict[str, Any] = {}

    semantic_scorer: Optional[Any] = None
    if config.use_embeddings:
        try:
            from evo_rhyme.siamese_scorer import SiameseRhymeScorer, SIAMESE_MODEL_DIR
            semantic_scorer = SiameseRhymeScorer(str(SIAMESE_MODEL_DIR), device="cuda")
            logger.info("Loaded SiameseRhymeScorer for QD verse semantic scoring (CUDA)")
        except Exception as e:
            logger.warning("Failed to load SiameseRhymeScorer: %s", e)

    run_logger: Optional[VerseQDRunLogger] = None
    if config.output_dir or (config.run_id is not None and config.run_id > 0):
        run_dir = Path(config.output_dir) if config.output_dir else None
        run_logger = VerseQDRunLogger(run_dir=run_dir, run_id=config.run_id)
        run_logger.write_config(config)

    dims = config.archive_dims
    if dims is None:
        mode = getattr(config, "archive_mode", "default")
        if mode == "style_chain":
            dims = style_chain_dimensions()
        elif mode == "compact_style":
            dims = compact_style_dimensions()
        elif mode == "ultra_compact":
            dims = ultra_compact_dimensions()
        elif mode == "curriculum_compact":
            dims = default_verse_dimensions()
    archive = create_verse_archive(
        dims,
        novelty_tiebreak=getattr(config, "archive_novelty_tiebreak", False),
    )

    # Novelty archives
    verse_novelty_archive = NoveltyArchive(max_size=5000, k_nearest=10)
    line_novelty_archive = NoveltyArchive(max_size=10000, k_nearest=15)

    # Line evolution config
    line_evo_config = LineEvolutionConfig(
        population_size=config.line_population_size,
        num_generations=config.line_generations_per_verse_gen,
        lm_seed_count=config.line_lm_seed_count,
        lm_mutation_budget=config.line_lm_mutation_budget,
        max_per_rhyme_group=config.max_lines_per_rhyme_group,
        theme_keywords=list(theme_keywords),
        min_syllables=config.min_syllables,
        max_syllables=config.max_syllables,
        use_embeddings=config.use_embeddings,
        embedding_weight=config.embedding_weight,
    )
    line_archive: Optional[LineArchive] = None

    # Template pool for evolved template selection
    try:
        template_pool = get_template_pool()
        logger.info("Initialized template pool with %d templates", template_pool.size())
    except Exception:
        template_pool = None
        logger.warning("Template pool initialization failed, using static templates")

    run_id_for_lineage = getattr(config, "run_id", None) or 0

    _stagnation_counter = 0
    _prev_best_fitness = -1.0
    _stagnation_threshold = 3
    _stagnation_boost_gens = 0
    _early_stop_best = -1.0
    _early_stop_stagnant = 0
    es_delta = float(getattr(config, "early_stop_min_delta", 0.005) or 0.005)
    es_k = getattr(config, "early_stop_stagnant_gens", None)

    for gen in range(config.num_generations):
        # Keep the operator tracer aware of current generation (if active)
        _tracer = _get_active_tracer()
        if _tracer is not None:
            _tracer.gen = gen

        if (
            config.archive_mode == "curriculum_compact"
            and gen == max(1, int(config.curriculum_switch_gen))
        ):
            old_entries = list(archive.best_per_niche().values())
            archive = create_verse_archive(
                compact_style_dimensions(),
                novelty_tiebreak=getattr(config, "archive_novelty_tiebreak", False),
            )
            archive.add_batch(old_entries)
            logger.info(
                "Curriculum switch: default -> compact_style at gen %d (%d niches)",
                gen,
                archive.total_niches(),
            )

        # 1. Analyze & batch-score all individuals
        for ind in population:
            analyze_verse_individual(ind)
        pop_scores = score_verses_batch(
            population,
            scheme=scheme,
            prompt_keywords=kw,
            theme_string=theme_string,
            semantic_scorer=semantic_scorer if config.use_embeddings else None,
            embedding_weight=config.embedding_weight,
            graph_top_k=config.graph_top_k,
            expensive_top_k=config.max_expensive_scoring_candidates,
            fast_mode=config.fast_mode,
            graph_edge_mode=config.graph_edge_mode,
        )
        for ind, sc in zip(population, pop_scores):
            ind.scores = sc
            ind.fitness = compute_verse_fitness(sc, weights)

        # --- Line evolution stage (incremental: reuse archive across generations) ---
        population_sorted = sorted(population, key=lambda i: i.fitness or 0.0, reverse=True)
        initial_scored_lines: List[ScoredLine] = []
        for ind in population_sorted:
            for line_text in ind.lines:
                try:
                    sl = score_line(
                        line_text, kw, theme_string, semantic_scorer,
                        embedding_weight=config.embedding_weight,
                    )
                    initial_scored_lines.append(sl)
                except Exception:
                    pass

        incremental_config = LineEvolutionConfig(
            population_size=line_evo_config.population_size,
            num_generations=line_evo_config.num_generations if gen == 0 else max(1, line_evo_config.num_generations // 3),
            lm_seed_count=line_evo_config.lm_seed_count if gen == 0 else 0,
            lm_mutation_budget=line_evo_config.lm_mutation_budget,
            max_per_rhyme_group=line_evo_config.max_per_rhyme_group,
            theme_keywords=line_evo_config.theme_keywords,
            min_syllables=line_evo_config.min_syllables,
            max_syllables=line_evo_config.max_syllables,
            use_embeddings=line_evo_config.use_embeddings,
            embedding_weight=line_evo_config.embedding_weight,
        )

        try:
            line_archive = evolve_lines(
                incremental_config,
                initial_lines=initial_scored_lines,
                semantic_scorer=semantic_scorer,
                novelty_archive=line_novelty_archive,
                existing_archive=line_archive,
            )
        except Exception:
            logger.warning("Line evolution failed, using previous archive", exc_info=True)

        # 2. Add all to archive
        improved = verse_archive_add_batch(
            archive, population, min_coherence=config.min_coherence,
        )

        # 3. Extract objectives for Pareto selection
        objectives = compute_population_objectives(population, score_vector)
        ranks = pareto_rank(population, objectives)

        all_crowding = [0.0] * len(population)
        for rank_val in set(ranks):
            front = [i for i, r in enumerate(ranks) if r == rank_val]
            if len(front) > 1:
                cd = crowding_distance(objectives, front)
                for idx, c in zip(front, cd):
                    all_crowding[idx] = c

        # 4. Select diverse parents from archive
        archive_parents = archive.sample_parents(config.population_size // 2)

        # 5. Select parents from population via Pareto tournament
        tournament_parents = pareto_tournament_select(
            population, ranks, all_crowding,
            n=config.population_size // 2,
            k=config.tournament_k,
        )

        parent_pool = archive_parents + tournament_parents

        # 6. Generate offspring (two-pass: generate candidates, then batch-score)
        # LM budget reserved exclusively for repair (Phase 3) -- not used in mutation
        lm_repair_budget: Dict[str, int] = {"remaining": config.lm_mutation_budget_per_gen}
        target = max(0, config.population_size - config.num_elites - config.random_immigrants_per_gen)

        # --- Phase 1: Generate all candidate offspring (fast: no GPU scoring) ---
        candidates: List[VerseIndividual] = []

        # 6a. Composed offspring from line archive (no LM)
        if line_archive is not None and line_archive.size() > 0 and target > 0:
            composed_target = max(1, int(target * config.composed_offspring_ratio))
            try:
                raw_composed = build_verse_batch(
                    line_archive, composed_target, scheme,
                    optimize_transitions=True,
                    lm_budget=None,
                )
                for v in raw_composed:
                    if passes_verse_constraints(v, constraint_config):
                        analyze_verse_individual(v)
                        candidates.append(v)
            except Exception:
                logger.warning("Verse building from archive failed", exc_info=True)

        # 6b. Evolved offspring via crossover/mutation (no LM -- pure evolutionary)
        evolved_target = target - len(candidates)
        attempts = 0
        evolved_candidates: List[VerseIndividual] = []

        while len(evolved_candidates) < evolved_target and attempts < config.max_offspring_attempts:
            attempts += 1
            if len(parent_pool) >= 2:
                p1, p2 = select_crossover_parents(
                    parent_pool,
                    semantic_pairing=getattr(config, "semantic_crossover_pairing", False),
                )
            else:
                p1 = parent_pool[0]
                p2 = parent_pool[0]

            if random.random() < config.crossover_rate:
                child = verse_crossover(p1, p2, crossover_config)
            else:
                child = VerseIndividual(lines=list(p1.lines))

            child = verse_mutate(
                child, mutation_config, mut_weights,
                constraint_config=constraint_config,
                lm_budget=None,
            )
            if _stagnation_boost_gens > 0 and random.random() < 0.5:
                child = verse_mutate(
                    child, mutation_config, mut_weights,
                    constraint_config=constraint_config,
                    lm_budget=None,
                )

            if passes_verse_constraints(child, constraint_config):
                analyze_verse_individual(child)
                child.metadata = child.metadata or {}
                p1_fit = p1.fitness or 0.0
                p2_fit = p2.fitness or 0.0
                child.metadata["_parent_fitness"] = max(p1_fit, p2_fit)
                p1_db = (p1.metadata or {}).get("db_id")
                p2_db = (p2.metadata or {}).get("db_id")
                if p1_db or p2_db:
                    child.metadata["_parent_db_ids"] = [
                        pid for pid in (p1_db, p2_db) if pid and pid > 0
                    ]
                evolved_candidates.append(child)

        candidates.extend(evolved_candidates)
        evolved_acceptance_rate = (len(evolved_candidates) / max(1, attempts)) if evolved_target > 0 else 0.0

        # --- Phase 2: Batch-score all candidates (one GPU pass) ---
        if candidates:
            batch_scores = score_verses_batch(
                candidates,
                scheme=scheme,
                prompt_keywords=kw,
                theme_string=theme_string,
                semantic_scorer=semantic_scorer if config.use_embeddings else None,
                embedding_weight=config.embedding_weight,
                graph_top_k=config.graph_top_k,
                expensive_top_k=config.max_expensive_scoring_candidates,
                fast_mode=config.fast_mode,
                graph_edge_mode=config.graph_edge_mode,
            )
            for cand, sc in zip(candidates, batch_scores):
                cand.scores = sc
        else:
            batch_scores = []

        # --- Phase 3: LM repair for low-fluency candidates (all LM budget reserved for this) ---
        if config.min_lm_fluency > 0:
            for i, cand in enumerate(candidates):
                lm_f = cand.scores.get("lm_fluency", 0.0)
                if lm_f < config.min_lm_fluency and lm_repair_budget.get("remaining", 0) > 0:
                    lines = list(cand.lines)
                    repaired_lines = list(lines)
                    did_repair = False
                    for pair_idx in range(len(lines) // 2):
                        cpl = CoupletIndividual(
                            line1=lines[pair_idx * 2],
                            line2=lines[pair_idx * 2 + 1],
                        )
                        repaired = _lm_repair_couplet(cpl, mutation_config, lm_repair_budget)
                        if repaired:
                            repaired_lines[pair_idx * 2] = repaired.line1
                            repaired_lines[pair_idx * 2 + 1] = repaired.line2
                            did_repair = True
                    if did_repair:
                        new_cand = VerseIndividual(
                            lines=repaired_lines, features=None, scores=None,
                            fitness=None, metadata=dict(cand.metadata),
                        )
                        if passes_verse_constraints(new_cand, constraint_config):
                            analyze_verse_individual(new_cand)
                            candidates[i] = new_cand

            repaired_indices = [
                i for i, c in enumerate(candidates) if c.scores is None
            ]
            if repaired_indices:
                repaired_cands = [candidates[i] for i in repaired_indices]
                repaired_scores = score_verses_batch(
                    repaired_cands,
                    scheme=scheme,
                    prompt_keywords=kw,
                    theme_string=theme_string,
                    semantic_scorer=semantic_scorer if config.use_embeddings else None,
                    embedding_weight=config.embedding_weight,
                    graph_top_k=config.graph_top_k,
                    expensive_top_k=config.max_expensive_scoring_candidates,
                    fast_mode=config.fast_mode,
                    graph_edge_mode=config.graph_edge_mode,
                )
                for idx, sc in zip(repaired_indices, repaired_scores):
                    candidates[idx].scores = sc

        # --- Phase 4: Filter by quality thresholds ---
        filtered: List[VerseIndividual] = []
        min_rhyme = getattr(config, "min_rhyme_scheme_score", 0.0)
        for cand in candidates:
            sc = cand.scores or {}
            if config.min_fluency > 0 and sc.get("fluency", 0.0) < config.min_fluency:
                continue
            if config.min_lm_fluency > 0 and sc.get("lm_fluency", 0.0) < config.min_lm_fluency:
                continue
            if config.min_coherence > 0 and sc.get("coherence", 0.0) < config.min_coherence:
                continue
            if min_rhyme > 0 and sc.get("rhyme_scheme_score", 0.0) < min_rhyme:
                continue
            if sc.get("garbled_line_penalty", 0.0) > 0.25:
                continue
            filtered.append(cand)

        # --- Phase 5: Batch novelty + fitness (single embed_texts call) ---
        try:
            if filtered:
                verse_texts = [" ".join(c.lines) for c in filtered]
                verse_embs = embed_texts(verse_texts)
                novelties = verse_novelty_archive.compute_novelty_batch(verse_embs)
                for cand, nov in zip(filtered, novelties):
                    cand.scores["novelty"] = float(nov)
                verse_novelty_archive.add_batch(verse_embs)
        except Exception:
            for cand in filtered:
                if cand.scores is not None:
                    cand.scores.setdefault("novelty", 0.5)

        for cand in filtered:
            parent_fit = (cand.metadata or {}).get("_parent_fitness")
            if parent_fit is not None:
                cand.scores["parent_improvement"] = compute_parent_improvement(
                    cand.scores, parent_fit, weights,
                )
            else:
                cand.scores.setdefault("parent_improvement", 0.5)
            cand.fitness = compute_verse_fitness(cand.scores, weights)

        all_offspring = sorted(filtered, key=lambda c: c.fitness or 0.0, reverse=True)[:target]

        # 6f. Persist offspring to DB + write lineage rows
        if run_id_for_lineage and run_id_for_lineage > 0:
            try:
                from evo_rhyme import db as _db
                if _db.db_enabled():
                    for cand in all_offspring:
                        ct = f"verse{len(cand.lines)}"
                        cid = _db.insert_candidate(
                            run_id_for_lineage, gen, ct, scheme,
                            cand.lines, cand.fitness or 0.0, cand.scores,
                        )
                        if cid and cid > 0:
                            cand.metadata = cand.metadata or {}
                            cand.metadata["db_id"] = cid
                            parent_ids = cand.metadata.pop("_parent_db_ids", None)
                            if parent_ids:
                                for pid in parent_ids:
                                    _db.insert_lineage(cid, pid, "crossover+mutate", gen)
            except Exception as e:
                logger.debug("verse QD offspring DB/lineage failed: %s", e)

        # 7. Elites from archive (best per niche, diverse)
        elites = archive.top_k(config.num_elites)

        # 8. Random immigrants (boosted during stagnation)
        immigrant_count = config.random_immigrants_per_gen
        if _stagnation_boost_gens > 0:
            immigrant_count = config.random_immigrants_per_gen * 2
            _stagnation_boost_gens -= 1
            logger.info("Stagnation boost active: immigrants=%d (2x), boost_gens_left=%d",
                        immigrant_count, _stagnation_boost_gens)
        immigrants: List[VerseIndividual] = []
        if immigrant_generator:
            for _ in range(immigrant_count):
                try:
                    imm = immigrant_generator()
                    if imm and passes_verse_constraints(imm, constraint_config):
                        immigrants.append(imm)
                except Exception:
                    pass

        # 9. Assemble next generation + diversity cull
        population = elites + all_offspring + immigrants
        population = _cull_near_duplicates(population, config.population_size)

        while len(population) < config.population_size and archive.occupied_niches() > 0:
            population.extend(archive.sample_parents(1))

        # Template pool evolution
        if template_pool is not None:
            # Record fitness for templates used in this generation's population
            for ind in population:
                if hasattr(ind, 'template_ids') and ind.template_ids and ind.fitness:
                    for tid in ind.template_ids:
                        if tid:
                            template_pool.record_fitness(tid, ind.fitness)
            # Evolve templates every 3 generations
            if gen > 0 and gen % 3 == 0:
                template_pool.evolve(mutation_rate=0.2, crossover_rate=0.1)
                logger.info(
                    "Evolved template pool: %d templates, top fitness=%.3f",
                    template_pool.size(),
                    template_pool.top_k(1)[0].avg_fitness if template_pool.top_k(1) else 0.0,
                )

        # 10. Logging + stagnation detection
        best_fit = max((ind.fitness or 0.0 for ind in population), default=0.0)
        mean_fit = (
            sum(ind.fitness or 0.0 for ind in population) / max(1, len(population))
        )

        if best_fit > _prev_best_fitness + 1e-4:
            _stagnation_counter = 0
            _prev_best_fitness = best_fit
        else:
            _stagnation_counter += 1
            if _stagnation_counter >= _stagnation_threshold and _stagnation_boost_gens <= 0:
                _stagnation_boost_gens = 2
                _stagnation_counter = 0
                logger.info("Stagnation detected at gen %d (best=%.4f unchanged for %d gens) -- activating boost",
                            gen, best_fit, _stagnation_threshold)

        if best_fit > _early_stop_best + es_delta:
            _early_stop_best = best_fit
            _early_stop_stagnant = 0
        else:
            _early_stop_stagnant += 1

        if getattr(config, "curriculum_tighten_on_stagnation", False):
            thr_t = max(1, int(getattr(config, "curriculum_tighten_after_stagnant_gens", 6)))
            if _early_stop_stagnant >= thr_t:
                nf_cap = float(getattr(config, "curriculum_tighten_fluency_cap", 0.55))
                nc_cap = float(getattr(config, "curriculum_tighten_coherence_cap", 0.42))
                nf_step = float(getattr(config, "curriculum_tighten_fluency_step", 0.03))
                nc_step = float(getattr(config, "curriculum_tighten_coherence_step", 0.03))
                new_f = min(nf_cap, float(config.min_fluency) + nf_step)
                new_c = min(nc_cap, float(config.min_coherence) + nc_step)
                if new_f > float(config.min_fluency) + 1e-6 or new_c > float(config.min_coherence) + 1e-6:
                    logger.info(
                        "Curriculum tighten at gen %d: min_fluency %.3f->%.3f min_coherence %.3f->%.3f (stagnant=%d)",
                        gen, config.min_fluency, new_f, config.min_coherence, new_c, _early_stop_stagnant,
                    )
                    config.min_fluency = new_f
                    config.min_coherence = new_c
                _early_stop_stagnant = 0

        if run_logger:
            run_logger.log_generation(
                gen, archive.coverage(), best_fit, mean_fit, archive.occupied_niches(),
                acceptance_rate=evolved_acceptance_rate,
                archive=archive,
            )

        if on_generation:
            on_generation(gen, archive, population)

        if (
            config.enable_controllability_probes
            and gen > 0
            and gen % max(1, config.controllability_probe_every) == 0
        ):
            _log_style_controllability_probe(
                archive=archive,
                sample_size=max(1, config.controllability_probe_batch_size),
            )

        logger.info(
            "Gen %d: pop=%d archive_coverage=%.1f%% archive_best=%.3f "
            "lm_budget_remaining=%d improved=%d",
            gen, len(population), archive.coverage() * 100,
            best_fit, lm_repair_budget.get("remaining", 0), improved,
        )
        if gen % 10 == 0 and gen > 0:
            try:
                from evo_rhyme.verse_fitness_breakdown import audit_population_fitness
                audit_population_fitness(population, weights, log_results=True)
            except Exception:
                logger.debug("Fitness audit skipped", exc_info=True)
        try:
            cs = verse_score_cache_snapshot()
            logger.info(
                "VerseScoreCache: hit_rate=%.3f size=%s hits=%s misses=%s evictions=%s expired=%s ttl_s=%s",
                float(cs.get("hit_rate", 0.0)),
                cs.get("size"),
                cs.get("hits"),
                cs.get("misses"),
                cs.get("evictions"),
                cs.get("expired"),
                cs.get("ttl_seconds"),
            )
        except Exception:
            pass

        if es_k is not None and int(es_k) > 0 and _early_stop_stagnant >= int(es_k):
            setattr(config, "early_stop_reason", f"stagnation_{int(es_k)}_gens_delta_{es_delta}")
            setattr(config, "early_stop_gen", int(gen))
            logger.info("Early stop at gen %d: %s", gen, config.early_stop_reason)
            break

    if run_logger:
        run_logger.flush(archive)

    return archive, population


def _log_style_controllability_probe(archive: MAPElitesArchive, sample_size: int = 8) -> None:
    """Cheap proxy for style genome controllability from archive occupancy."""
    parents = archive.sample_parents(sample_size)
    if not parents:
        return
    tone_counts: Dict[str, int] = {}
    narr_counts: Dict[str, int] = {}
    meta_counts: Dict[str, int] = {}
    for p in parents:
        labels = (p.metadata or {}).get("style_genome_labels", {})
        tone = labels.get("tone", "unknown")
        narr = labels.get("narrativity", "unknown")
        meta = labels.get("metaphor_density", "unknown")
        tone_counts[tone] = tone_counts.get(tone, 0) + 1
        narr_counts[narr] = narr_counts.get(narr, 0) + 1
        meta_counts[meta] = meta_counts.get(meta, 0) + 1
    logger.info(
        "Controllability probe | tone=%s narr=%s metaphor=%s",
        tone_counts,
        narr_counts,
        meta_counts,
    )


# ---------------------------------------------------------------------------
# Emitter-based MAP-Elites evolution
# ---------------------------------------------------------------------------


def evolve_verse_qd_emitters(
    population: List[VerseIndividual],
    config: QDEvolutionConfig,
    immigrant_generator: Optional[Callable[[], VerseIndividual]] = None,
    on_generation: Optional[Callable[[int, MAPElitesArchive, List[VerseIndividual]], None]] = None,
) -> Tuple[MAPElitesArchive, List[VerseIndividual]]:
    """Emitter-based MAP-Elites QD evolution.

    Uses specialized emitters coordinated by an adaptive scheduler
    coordinated by an adaptive scheduler for better coverage.
    """
    from evo_rhyme.emitters import (
        create_emitters, run_emitter_generation,
    )

    scheme = config.rhyme_scheme or "AABB"
    weights = config.fitness_weights or VERSE_DEFAULT_WEIGHTS
    theme_keywords = config.theme_keywords or []
    kw = set(w.lower() for w in theme_keywords) if theme_keywords else None
    theme_string = " ".join(theme_keywords) if theme_keywords else None

    semantic_scorer = None
    if config.use_embeddings:
        try:
            from evo_rhyme.siamese_scorer import SiameseRhymeScorer, SIAMESE_MODEL_DIR
            semantic_scorer = SiameseRhymeScorer(str(SIAMESE_MODEL_DIR), device="cuda")
        except Exception as e:
            logger.warning("Failed to load SiameseRhymeScorer: %s", e)

    run_logger = None
    run_id = getattr(config, "run_id", None)
    if config.output_dir or (run_id is not None and run_id > 0):
        run_dir = Path(config.output_dir) if config.output_dir else None
        run_logger = VerseQDRunLogger(run_dir=run_dir, run_id=run_id)
        run_logger.write_config(config)

    dims = config.archive_dims
    if dims is None:
        mode = getattr(config, "archive_mode", "default")
        if mode == "style_chain":
            dims = style_chain_dimensions()
        elif mode == "compact_style":
            dims = compact_style_dimensions()
        elif mode == "ultra_compact":
            dims = ultra_compact_dimensions()
        elif mode == "curriculum_compact":
            dims = default_verse_dimensions()
    archive = config.initial_archive if getattr(config, "initial_archive", None) else create_verse_archive(
        dims,
        novelty_tiebreak=getattr(config, "archive_novelty_tiebreak", False),
    )

    for ind in population:
        analyze_verse_individual(ind)
        if config.enable_style_genome or config.enable_prompt_genome:
            try:
                from evo_rhyme.emitters import _attach_genomes  # type: ignore
                _attach_genomes(ind)
            except Exception:
                pass
    pop_scores = score_verses_batch(
        population, scheme=scheme, prompt_keywords=kw, theme_string=theme_string,
        semantic_scorer=semantic_scorer if config.use_embeddings else None,
        embedding_weight=config.embedding_weight,
        graph_top_k=config.graph_top_k,
        expensive_top_k=config.max_expensive_scoring_candidates,
        fast_mode=config.fast_mode,
        graph_edge_mode=config.graph_edge_mode,
    )
    for ind, sc in zip(population, pop_scores):
        ind.scores = sc
        ind.fitness = compute_verse_fitness(sc, weights)
    verse_archive_add_batch(
        archive, population, min_coherence=config.min_coherence,
    )

    logger.info(
        "Initial archive: %d/%d niches (%.1f%% coverage), best=%.3f",
        archive.occupied_niches(), archive.total_niches(),
        archive.coverage() * 100,
        max((ind.fitness or 0 for ind in population), default=0),
    )

    emitter_config = {
        "theme_keywords": list(theme_keywords),
        "corpus_path": None,
        "min_syllables": config.min_syllables,
        "max_syllables": config.max_syllables,
        "use_structural_mutations": getattr(config, "use_structural_mutations", False),
        "embedding_neighbor_k": 12,
        "embedding_min_cosine": 0.58,
        "schemes": config.allowed_schemes if hasattr(config, 'allowed_schemes') else ["AABB", "ABAB", "ABBA", "ABCB"],
        "crossover_rate": config.crossover_rate,
        "enable_style_genome": config.enable_style_genome,
        "enable_prompt_genome": config.enable_prompt_genome,
        "prompt_llm_fraction": config.prompt_llm_fraction if config.lm_mutation_budget_per_gen > 0 else 0.0,
        "prompt_dedup_enabled": config.prompt_dedup_enabled,
        "proposer_config": config.proposer_config if config.lm_mutation_budget_per_gen > 0 else {},
        "niche_targeting_sample": 600,
        "niche_targeting_queue": 400,
        "coverage_target": config.coverage_target,
        "coverage_boost_threshold": 0.35,
        "semantic_crossover_pairing": getattr(config, "semantic_crossover_pairing", False),
        "lm_mutation_budget_per_gen": config.lm_mutation_budget_per_gen,
    }
    if config.corpus_vocab:
        emitter_config["corpus_vocab"] = config.corpus_vocab
    emitters, scheduler = create_emitters(emitter_config)

    _early_stop_best = -1.0
    _early_stop_stagnant = 0
    es_delta = float(getattr(config, "early_stop_min_delta", 0.005) or 0.005)
    es_k = getattr(config, "early_stop_stagnant_gens", None)

    def batch_score(candidates):
        for c in candidates:
            if c.features is None:
                analyze_verse_individual(c)
        return score_verses_batch(
            candidates, scheme=scheme, prompt_keywords=kw, theme_string=theme_string,
            semantic_scorer=semantic_scorer if config.use_embeddings else None,
            embedding_weight=config.embedding_weight,
            graph_top_k=config.graph_top_k,
            expensive_top_k=config.max_expensive_scoring_candidates,
            fast_mode=config.fast_mode,
            graph_edge_mode=config.graph_edge_mode,
        )

    def fitness_fn(scores):
        return compute_verse_fitness(scores, weights)

    for gen in range(config.num_generations):
        _tracer = _get_active_tracer()
        if _tracer is not None:
            _tracer.gen = gen

        if (
            config.archive_mode == "curriculum_compact"
            and gen == max(1, int(config.curriculum_switch_gen))
        ):
            old_entries = list(archive.best_per_niche().values())
            archive = create_verse_archive(
                compact_style_dimensions(),
                novelty_tiebreak=getattr(config, "archive_novelty_tiebreak", False),
            )
            archive.add_batch(old_entries)
            logger.info(
                "Curriculum switch: default -> compact_style at gen %d (%d niches)",
                gen,
                archive.total_niches(),
            )

        gen_start = time.perf_counter()
        total_budget = config.population_size

        gen_stats = run_emitter_generation(
            archive=archive,
            emitters=emitters,
            scheduler=scheduler,
            total_budget=total_budget,
            generation=gen,
            score_fn=batch_score,
            fitness_fn=fitness_fn,
            novelty_weight=config.novelty_weight if hasattr(config, 'novelty_weight') else 0.3,
            min_coherence=config.min_coherence,
        )

        population = list(archive.top_k(min(config.population_size, archive.occupied_niches())))
        while len(population) < config.population_size and archive.occupied_niches() > 0:
            population.extend(archive.sample_parents(1))

        best_fit = max((ind.fitness or 0.0 for ind in population), default=0.0)
        mean_fit = sum(ind.fitness or 0.0 for ind in population) / max(1, len(population))
        elapsed_s = max(1e-6, time.perf_counter() - gen_start)
        runtime = {
            "gen_wall_time_s": elapsed_s,
            "candidates_generated": gen_stats.get("total_candidates", 0),
            "inserted": gen_stats.get("inserted", 0),
            "niches_per_sec": gen_stats.get("new_niches", 0) / elapsed_s,
            "candidates_per_sec": gen_stats.get("total_candidates", 0) / elapsed_s,
        }
        total_cand = int(gen_stats.get("total_candidates", 0) or 0)
        inserted = int(gen_stats.get("inserted", 0) or 0)
        acceptance_rate = (inserted / max(1, total_cand)) if total_cand > 0 else 0.0

        if run_logger:
            run_logger.log_generation(
                gen, archive.coverage(), best_fit, mean_fit, archive.occupied_niches(),
                acceptance_rate=acceptance_rate,
                runtime=runtime,
                archive=archive,
            )

        if on_generation:
            on_generation(gen, archive, population)

        if (
            config.enable_controllability_probes
            and gen > 0
            and gen % max(1, config.controllability_probe_every) == 0
        ):
            _log_style_controllability_probe(
                archive=archive,
                sample_size=max(1, config.controllability_probe_batch_size),
            )

        sched_weights = {k: f"{v:.2f}" for k, v in scheduler._weights.items()}
        logger.info(
            "Gen %d: coverage=%.1f%% (%d/%d) best=%.3f inserted=%d new_niches=%d cand/s=%.1f niche/s=%.2f weights=%s",
            gen,
            archive.coverage() * 100,
            archive.occupied_niches(),
            archive.total_niches(),
            best_fit,
            gen_stats.get("inserted", 0),
            gen_stats.get("new_niches", 0),
            runtime["candidates_per_sec"],
            runtime["niches_per_sec"],
            sched_weights,
        )
        if gen % 10 == 0 and gen > 0:
            try:
                from evo_rhyme.verse_fitness_breakdown import audit_population_fitness
                audit_population_fitness(population, weights, log_results=True)
            except Exception:
                logger.debug("Fitness audit skipped", exc_info=True)

        if best_fit > _early_stop_best + es_delta:
            _early_stop_best = best_fit
            _early_stop_stagnant = 0
        else:
            _early_stop_stagnant += 1

        if getattr(config, "curriculum_tighten_on_stagnation", False):
            thr_t = max(1, int(getattr(config, "curriculum_tighten_after_stagnant_gens", 6)))
            if _early_stop_stagnant >= thr_t:
                nf_cap = float(getattr(config, "curriculum_tighten_fluency_cap", 0.55))
                nc_cap = float(getattr(config, "curriculum_tighten_coherence_cap", 0.42))
                nf_step = float(getattr(config, "curriculum_tighten_fluency_step", 0.03))
                nc_step = float(getattr(config, "curriculum_tighten_coherence_step", 0.03))
                new_f = min(nf_cap, float(config.min_fluency) + nf_step)
                new_c = min(nc_cap, float(config.min_coherence) + nc_step)
                if new_f > float(config.min_fluency) + 1e-6 or new_c > float(config.min_coherence) + 1e-6:
                    logger.info(
                        "Curriculum tighten at gen %d: min_fluency %.3f->%.3f min_coherence %.3f->%.3f (stagnant=%d)",
                        gen, config.min_fluency, new_f, config.min_coherence, new_c, _early_stop_stagnant,
                    )
                    config.min_fluency = new_f
                    config.min_coherence = new_c
                _early_stop_stagnant = 0

        if es_k is not None and int(es_k) > 0 and _early_stop_stagnant >= int(es_k):
            setattr(config, "early_stop_reason", f"stagnation_{int(es_k)}_gens_delta_{es_delta}")
            setattr(config, "early_stop_gen", int(gen))
            logger.info("Early stop at gen %d: %s", gen, config.early_stop_reason)
            break

    if run_logger:
        run_logger.flush(archive)

    return archive, population
