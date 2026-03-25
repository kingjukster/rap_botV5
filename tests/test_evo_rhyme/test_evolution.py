"""Tests for evo_rhyme.evolution: Pareto ranking, tournament selection, diversity, elites, logger."""

from pathlib import Path
from typing import Set

import pytest

from evo_rhyme.constraints import ConstraintConfig
from evo_rhyme.evolution import (
    EvolutionConfig,
    EvolutionRunLogger,
    OBJECTIVE_KEYS,
    _compute_crowding_by_rank,
    _compute_diversity,
    _crowding_distance_for_indices,
    _dominates,
    _effective_constraint_config,
    _end_words_from_population,
    _objectives_from_scores,
    _compute_pareto_ranks,
    _select_elites_niching,
    _top_rhyme_tails,
    _tournament_select,
    _tournament_select_multiobjective,
)
from evo_rhyme.individual import CoupletIndividual


# ---------------------------------------------------------------------------
# EvolutionConfig
# ---------------------------------------------------------------------------


def test_evolution_config_defaults():
    cfg = EvolutionConfig()
    assert cfg.population_size == 100
    assert cfg.num_elites == 5
    assert cfg.tournament_k == 3
    assert cfg.multiobjective is False
    assert cfg.output_dir is None


def test_evolution_config_custom():
    cfg = EvolutionConfig(population_size=50, num_elites=3, multiobjective=True)
    assert cfg.population_size == 50
    assert cfg.num_elites == 3
    assert cfg.multiobjective is True


# ---------------------------------------------------------------------------
# Objectives and Pareto
# ---------------------------------------------------------------------------


def test_objectives_from_scores_empty():
    assert _objectives_from_scores(None) == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_objectives_from_scores_full():
    scores = {
        "end_rhyme": 0.8,
        "fluency": 0.9,
        "semantic": 0.5,
        "corpus_overlap_penalty": 0.2,
        "rhyme_family_novelty": 0.7,
    }
    obj = _objectives_from_scores(scores)
    assert obj[0] == 0.8
    assert obj[1] == 0.9
    assert obj[2] == 0.5
    assert obj[3] == pytest.approx(0.8)  # 1 - corpus_overlap
    assert obj[4] == 0.7


def test_dominates_same_length():
    assert _dominates([1.0, 1.0], [0.5, 0.5]) is True
    assert _dominates([0.5, 0.5], [1.0, 1.0]) is False
    assert _dominates([1.0, 0.5], [1.0, 0.5]) is False  # equal
    assert _dominates([1.0, 0.6], [1.0, 0.5]) is True


def test_dominates_different_length_returns_false():
    assert _dominates([1.0, 1.0], [1.0]) is False


def test_compute_pareto_ranks_two_non_dominating():
    ind1 = CoupletIndividual(line1="a", line2="b")
    ind1.fitness = 0.5
    ind1.scores = {"end_rhyme": 1.0, "fluency": 0.0}
    ind2 = CoupletIndividual(line1="c", line2="d")
    ind2.fitness = 0.5
    ind2.scores = {"end_rhyme": 0.0, "fluency": 1.0}
    pop = [ind1, ind2]
    ranks = _compute_pareto_ranks(pop)
    assert ranks[0] == 0
    assert ranks[1] == 0


def test_compute_pareto_ranks_one_dominates():
    ind1 = CoupletIndividual(line1="a", line2="b")
    ind1.scores = {"end_rhyme": 1.0, "fluency": 1.0}
    ind2 = CoupletIndividual(line1="c", line2="d")
    ind2.scores = {"end_rhyme": 0.0, "fluency": 0.0}
    pop = [ind1, ind2]
    ranks = _compute_pareto_ranks(pop)
    assert ranks[0] == 0
    assert ranks[1] == 1


def test_crowding_distance_small_front():
    ind1 = CoupletIndividual(line1="a", line2="b")
    ind1.scores = {"end_rhyme": 0.5, "fluency": 0.5}
    ind2 = CoupletIndividual(line1="c", line2="d")
    ind2.scores = {"end_rhyme": 0.5, "fluency": 0.5}
    pop = [ind1, ind2]
    dist = _crowding_distance_for_indices(pop, [0, 1])
    assert dist[0] == float("inf")
    assert dist[1] == float("inf")


def test_crowding_distance_three_individuals():
    inds = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
        CoupletIndividual(line1="e", line2="f"),
    ]
    for i, ind in enumerate(inds):
        ind.scores = {"end_rhyme": float(i) * 0.5, "fluency": 0.5}
    dist = _crowding_distance_for_indices(inds, [0, 1, 2])
    assert dist[0] == float("inf")
    assert dist[2] == float("inf")
    assert 0 <= dist[1] < float("inf")


# ---------------------------------------------------------------------------
# Tournament and helpers
# ---------------------------------------------------------------------------


def test_tournament_select_empty_raises():
    with pytest.raises(ValueError, match="Empty population"):
        _tournament_select([], 3)


def test_tournament_select_single():
    ind = CoupletIndividual(line1="a", line2="b")
    ind.fitness = 0.9
    winner = _tournament_select([ind], 1)
    assert winner is ind


def test_tournament_select_best_wins():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
        CoupletIndividual(line1="e", line2="f"),
    ]
    pop[0].fitness = 0.3
    pop[1].fitness = 0.9
    pop[2].fitness = 0.5
    # With enough trials we should get the best sometimes; deterministic with fixed seed
    import random
    random.seed(42)
    winner = _tournament_select(pop, 3)
    assert winner.fitness == 0.9


def test_end_words_from_population():
    pop = [
        CoupletIndividual(line1="first line one", line2="second line two"),
        CoupletIndividual(line1="hello world", line2="foo bar"),
    ]
    words = _end_words_from_population(pop)
    assert "one" in words
    assert "two" in words
    assert "world" in words
    assert "bar" in words
    assert len(words) == 4


def test_compute_diversity():
    pop = [
        CoupletIndividual(line1="mean scene", line2="green"),  # similar tails
        CoupletIndividual(line1="flow go", line2="show"),
    ]
    for ind in pop:
        ind.fitness = 0.5
    div = _compute_diversity(pop)
    assert 0 <= div <= 1


def test_top_rhyme_tails():
    pop = [
        CoupletIndividual(line1="mean", line2="scene"),
        CoupletIndividual(line1="green", line2="queen"),
        CoupletIndividual(line1="mean", line2="clean"),
    ]
    top = _top_rhyme_tails(pop, k=2)
    assert len(top) <= 2
    assert all(isinstance(t, tuple) and len(t) == 2 for t in top)


# ---------------------------------------------------------------------------
# Elites niching
# ---------------------------------------------------------------------------


def test_select_elites_niching():
    # Sorted by fitness best first
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
        CoupletIndividual(line1="e", line2="f"),
        CoupletIndividual(line1="g", line2="h"),
    ]
    pop[0].fitness = 0.9
    pop[1].fitness = 0.8
    pop[2].fitness = 0.7
    pop[3].fitness = 0.6
    elites = _select_elites_niching(pop, k=2)
    assert len(elites) == 2
    assert elites[0].fitness == 0.9


# ---------------------------------------------------------------------------
# Effective constraint config
# ---------------------------------------------------------------------------


def test_effective_constraint_config_none():
    assert _effective_constraint_config(None, None) is None


def test_effective_constraint_config_no_theme_required():
    cfg = EvolutionConfig(constraint_config=ConstraintConfig())
    result = _effective_constraint_config(cfg, None)
    assert result is cfg.constraint_config


def test_effective_constraint_config_theme_required():
    cfg = EvolutionConfig(require_theme_presence=True, constraint_config=None)
    kw: Set[str] = {"flow", "money"}
    result = _effective_constraint_config(cfg, kw)
    assert isinstance(result, dict)
    assert result["require_theme_presence"] is True
    assert "flow" in result["prompt_keywords"]


# ---------------------------------------------------------------------------
# EvolutionRunLogger
# ---------------------------------------------------------------------------


def test_evolution_run_logger_write_config(tmp_path):
    logger = EvolutionRunLogger(run_dir=tmp_path)
    cfg = EvolutionConfig(population_size=40, num_elites=4)
    logger.write_config(cfg)
    config_file = tmp_path / "config.json"
    assert config_file.exists()
    import json
    data = json.loads(config_file.read_text())
    assert data["population_size"] == 40
    assert data["num_elites"] == 4
    assert "provenance" in data
    assert "written_at_utc" in data["provenance"]


def test_evolution_run_logger_log_generation_and_flush(tmp_path):
    logger = EvolutionRunLogger(run_dir=tmp_path)
    ind = CoupletIndividual(line1="a", line2="b")
    ind.fitness = 0.8
    ind.scores = {}
    logger.log_generation(0, 0.8, 0.5, 0.3, 0.2, [ind] * 5)
    logger.flush()
    assert (tmp_path / "score_history.csv").exists()
    assert (tmp_path / "top_candidates.json").exists()
    assert (tmp_path / "generation_metrics.jsonl").exists()
    assert (tmp_path / "run_manifest.json").exists()


def test_tournament_select_multiobjective():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
    ]
    pop[0].fitness = 0.5
    pop[1].fitness = 0.5
    ranks = {0: 0, 1: 1}
    crowding = {0: 1.0, 1: 0.5}
    import random
    random.seed(99)
    winner = _tournament_select_multiobjective(pop, 2, ranks, crowding)
    assert winner in pop


# ---------------------------------------------------------------------------
# evolve() one generation
# ---------------------------------------------------------------------------


def test_evolve_one_generation_tiny_population():
    """Run evolve() for 1 generation with 4 individuals to cover main loop."""
    from evo_rhyme.evolution import evolve

    pop = [
        CoupletIndividual(line1="The burns of life bring many concerns", line2="We learn from pain as the whole world turns"),
        CoupletIndividual(line1="I got the flow when I step in the spot", line2="You know I rock it hard when I hit the block"),
        CoupletIndividual(line1="Diamonds on my wrist they shining bright", line2="Running through the city every night"),
        CoupletIndividual(line1="From the bottom to the top we rise", line2="Looking at the world through different eyes"),
    ]
    config = EvolutionConfig(population_size=4, num_elites=1, tournament_k=2, max_offspring_attempts=5)
    result = evolve(pop, generations=1, config=config, prompt_keywords=None, immigrant_generator=None)
    assert len(result) == 4
    assert all(hasattr(ind, "fitness") for ind in result)
    assert result[0].fitness >= result[-1].fitness


def test_evolve_two_generations_with_output_dir(tmp_path):
    """evolve() with output_dir writes config and flushes run_logger."""
    from evo_rhyme.evolution import evolve

    pop = [
        CoupletIndividual(line1="The burns of life bring many concerns", line2="We learn from pain as the whole world turns"),
        CoupletIndividual(line1="I got the flow when I step in the spot", line2="You know I rock it hard when I hit the block"),
    ]
    config = EvolutionConfig(
        population_size=2,
        num_elites=1,
        tournament_k=2,
        max_offspring_attempts=3,
        output_dir=tmp_path,
    )
    result = evolve(pop, generations=2, config=config, prompt_keywords=None, immigrant_generator=None)
    assert len(result) == 2
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "score_history.csv").exists() or (tmp_path / "top_candidates.json").exists()


def test_evolve_multiobjective_tiny_population():
    """evolve_multiobjective with 2 individuals, 1 generation (Pareto + crowding in loop)."""
    from evo_rhyme.evolution import evolve

    pop = [
        CoupletIndividual(line1="The burns of life bring many concerns", line2="We learn from pain as the whole world turns"),
        CoupletIndividual(line1="I got the flow when I step in the spot", line2="You know I rock it hard when I hit the block"),
    ]
    config = EvolutionConfig(
        population_size=2,
        num_elites=1,
        tournament_k=2,
        multiobjective=True,
        max_offspring_attempts=3,
    )
    result = evolve(pop, generations=1, config=config, prompt_keywords=None, immigrant_generator=None)
    assert len(result) == 2
    assert all(hasattr(ind, "fitness") for ind in result)


def test_evolve_with_prompt_keywords():
    """evolve with prompt_keywords uses theme for fitness (theme_string path)."""
    from evo_rhyme.evolution import evolve

    pop = [
        CoupletIndividual(line1="The burns of life bring many concerns", line2="We learn from pain as the whole world turns"),
        CoupletIndividual(line1="I got the flow when I step in the spot", line2="You know I rock it hard when I hit the block"),
    ]
    config = EvolutionConfig(
        population_size=2,
        num_elites=1,
        tournament_k=2,
        max_offspring_attempts=3,
    )
    result = evolve(pop, generations=1, config=config, prompt_keywords={"pressure", "flow"}, immigrant_generator=None)
    assert len(result) == 2
    assert all(hasattr(ind, "fitness") for ind in result)


def test_evolve_immigrant_generator_called(tmp_path):
    """When random_immigrants_per_gen > 0 and immigrant_generator provided, it is called."""
    from evo_rhyme.evolution import evolve

    calls = []

    def immigrant_gen(size: int):
        calls.append(size)
        return [
            CoupletIndividual(line1="Immigrant line one here", line2="Immigrant line two there")
            for _ in range(size)
        ]

    pop = [
        CoupletIndividual(line1="The burns of life bring many concerns", line2="We learn from pain as the whole world turns"),
        CoupletIndividual(line1="I got the flow when I step in the spot", line2="You know I rock it hard when I hit the block"),
    ]
    config = EvolutionConfig(
        population_size=4,
        num_elites=1,
        tournament_k=2,
        max_offspring_attempts=3,
        random_immigrants_per_gen=1,
    )
    result = evolve(pop, generations=2, config=config, prompt_keywords=None, immigrant_generator=immigrant_gen)
    assert len(calls) >= 1
    assert calls[0] == 1
    assert len(result) >= 2
    assert all(hasattr(ind, "fitness") for ind in result)
