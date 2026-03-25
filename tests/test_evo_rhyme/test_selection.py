"""Tests for evo_rhyme.selection: elitism, tournament, immigrants, Pareto rank, crowding."""

from typing import Set

import pytest

from evo_rhyme.individual import CoupletIndividual
from evo_rhyme.selection import (
    compute_population_objectives,
    crowding_distance,
    elitism,
    epsilon_dominates,
    inject_random_immigrants,
    pareto_elitism,
    pareto_rank,
    pareto_tournament_select,
    tournament_select,
    _dominates,
)


def test_elitism_empty():
    assert elitism([], 3) == []


def test_elitism_n_zero():
    ind = CoupletIndividual(line1="a", line2="b")
    ind.fitness = 0.9
    assert elitism([ind], 0) == []


def test_elitism_top_n():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
        CoupletIndividual(line1="e", line2="f"),
    ]
    pop[0].fitness = 0.3
    pop[1].fitness = 0.9
    pop[2].fitness = 0.5
    top = elitism(pop, 2)
    assert len(top) == 2
    assert top[0].fitness == 0.9
    assert top[1].fitness == 0.5


def test_elitism_custom_key():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
    ]
    pop[0].metadata["score"] = 10
    pop[1].metadata["score"] = 20
    top = elitism(pop, 1, key=lambda ind: ind.metadata.get("score", 0))
    assert len(top) == 1
    assert top[0].metadata["score"] == 20


def test_tournament_select_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        tournament_select([])


def test_tournament_select_single():
    ind = CoupletIndividual(line1="a", line2="b")
    ind.fitness = 0.8
    assert tournament_select([ind], k=1) is ind


def test_tournament_select_best_wins():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
        CoupletIndividual(line1="e", line2="f"),
    ]
    pop[0].fitness = 0.2
    pop[1].fitness = 0.9
    pop[2].fitness = 0.4
    import random
    random.seed(123)
    winner = tournament_select(pop, k=3)
    assert winner.fitness == 0.9


def test_inject_random_immigrants():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
    ]
    pop[0].fitness = 0.5

    def generator(theme_keywords: Set[str] | None = None):
        return CoupletIndividual(line1="new", line2="verse")

    new_pop = inject_random_immigrants(pop, generator, count=2)
    assert len(new_pop) == 3
    assert new_pop[0] is pop[0]
    assert new_pop[1].line1 == "new"
    assert new_pop[2].line1 == "new"


def test_inject_random_immigrants_generator_no_kwargs():
    pop = [CoupletIndividual(line1="a", line2="b")]

    def generator():
        return CoupletIndividual(line1="x", line2="y")

    new_pop = inject_random_immigrants(pop, generator, count=1)
    assert len(new_pop) == 2
    assert new_pop[1].line1 == "x"


def test_dominates():
    assert _dominates([1.0, 1.0], [0.5, 0.5]) is True
    assert _dominates([0.5, 0.5], [1.0, 1.0]) is False
    assert _dominates([1.0, 0.5], [1.0, 0.5]) is False


def test_pareto_rank_empty():
    assert pareto_rank([], []) == []


def test_pareto_rank_two_non_dominating():
    objectives = [[1.0, 0.0], [0.0, 1.0]]
    pop = [CoupletIndividual(line1="a", line2="b"), CoupletIndividual(line1="c", line2="d")]
    ranks = pareto_rank(pop, objectives)
    assert ranks == [0, 0]


def test_pareto_rank_one_dominates():
    objectives = [[1.0, 1.0], [0.0, 0.0]]
    pop = [CoupletIndividual(line1="a", line2="b"), CoupletIndividual(line1="c", line2="d")]
    ranks = pareto_rank(pop, objectives)
    assert ranks[0] == 0
    assert ranks[1] == 1


def test_crowding_distance_empty_front():
    assert crowding_distance([[1.0, 0.5]], []) == []


def test_crowding_distance_small_front():
    dist = crowding_distance([[0.0, 0.5], [1.0, 0.5]], [0, 1])
    assert len(dist) == 2
    assert dist[0] == float("inf")
    assert dist[1] == float("inf")


def test_crowding_distance_three():
    objectives = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    dist = crowding_distance(objectives, [0, 1, 2])
    assert len(dist) == 3
    assert dist[0] == float("inf")
    assert dist[2] == float("inf")
    assert dist[1] < float("inf") and dist[1] > 0


def test_pareto_tournament_select():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
        CoupletIndividual(line1="e", line2="f"),
    ]
    objectives = [[1.0, 0.5], [0.5, 1.0], [0.3, 0.3]]
    ranks = pareto_rank(pop, objectives)
    dist = crowding_distance(objectives, list(range(3)))
    selected = pareto_tournament_select(pop, ranks, dist, n=2, k=2)
    assert len(selected) == 2
    assert all(ind in pop for ind in selected)


def test_pareto_elitism():
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
    ]
    ranks = [1, 0]
    crowding = [0.5, 1.0]
    top = pareto_elitism(pop, ranks, crowding, n=1)
    assert len(top) == 1
    assert top[0] == pop[1]
    assert pareto_elitism(pop, ranks, crowding, n=0) == []
    assert pareto_elitism([], [], [], n=1) == []


def test_compute_population_objectives():
    from evo_rhyme.fitness import score_vector, OBJECTIVE_KEYS
    pop = [
        CoupletIndividual(line1="a", line2="b"),
        CoupletIndividual(line1="c", line2="d"),
    ]
    pop[0].scores = {"end_rhyme": 0.8, "fluency": 0.9}
    pop[1].scores = None
    vecs = compute_population_objectives(pop, lambda s: score_vector(s))
    assert len(vecs) == 2
    assert len(vecs[0]) == len(OBJECTIVE_KEYS)
    assert vecs[1] == [0.0] * len(OBJECTIVE_KEYS)


def test_epsilon_dominates():
    assert epsilon_dominates([0.6, 0.6], [0.5, 0.5], eps=0.02)
    assert not epsilon_dominates([0.5, 0.5], [0.6, 0.6], eps=0.02)
