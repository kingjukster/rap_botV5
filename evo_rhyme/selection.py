"""
evo_rhyme/selection.py

Selection operators for evolutionary rhyme: elitism, tournament selection,
random immigrant injection, and multi-objective Pareto-based selection.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from evo_rhyme.individual import CoupletIndividual


def elitism(
    population: List[CoupletIndividual],
    n: int,
    key: Optional[Callable[[CoupletIndividual], float]] = None,
) -> List[CoupletIndividual]:
    """
    Keep top N individuals by fitness.
    key: optional callable to extract fitness (default: individual.fitness).
    """
    if n <= 0 or not population:
        return []

    def _fitness(ind: CoupletIndividual) -> float:
        if key is not None:
            return key(ind)
        f = ind.fitness
        return f if f is not None else float("-inf")

    sorted_pop = sorted(population, key=_fitness, reverse=True)
    return sorted_pop[:n]


def tournament_select(
    population: List[CoupletIndividual],
    k: int = 4,
    key: Optional[Callable[[CoupletIndividual], float]] = None,
) -> CoupletIndividual:
    """
    Tournament selection: pick k random individuals, return the best.
    key: optional callable to extract fitness (default: individual.fitness).
    """
    if not population:
        raise ValueError("tournament_select requires non-empty population")

    def _fitness(ind: CoupletIndividual) -> float:
        if key is not None:
            return key(ind)
        f = ind.fitness
        return f if f is not None else float("-inf")

    pool = random.choices(population, k=min(k, len(population)))
    return max(pool, key=_fitness)


def inject_random_immigrants(
    population: List[CoupletIndividual],
    generator: Callable[..., CoupletIndividual],
    count: int,
    theme_keywords: Optional[Set[str]] = None,
) -> List[CoupletIndividual]:
    """
    Add count random immigrants to the population.
    generator: callable that returns CoupletIndividual (e.g. evo_rhyme.generator.generate_couplet).
    theme_keywords: optional set passed to generator if it accepts it.
    """
    immigrants: List[CoupletIndividual] = []
    for _ in range(count):
        try:
            ind = generator(theme_keywords=theme_keywords)
        except TypeError:
            ind = generator()
        immigrants.append(ind)
    return list(population) + immigrants


# ---------------------------------------------------------------------------
# Multi-objective (Pareto) selection operators
# ---------------------------------------------------------------------------


def pareto_rank(
    population: List[Any],
    objectives: List[List[float]],
) -> List[int]:
    """Assign Pareto front rank to each individual.

    Args:
        population: List of individuals (not modified).
        objectives: List of objective vectors, one per individual.
            Each vector has the same length. All objectives are MAXIMIZED.

    Returns:
        List of ints, same length as population. Rank 0 = Pareto front
        (non-dominated), rank 1 = second front, etc.

    Algorithm: Fast non-dominated sorting (NSGA-II style).
    Individual A dominates B iff A is >= B on all objectives and > B on at
    least one.
    """
    n = len(population)
    if n == 0:
        return []

    dominated_set: List[List[int]] = [[] for _ in range(n)]
    domination_count: List[int] = [0] * n
    ranks: List[int] = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            i_dom_j = _dominates(objectives[i], objectives[j])
            j_dom_i = _dominates(objectives[j], objectives[i])
            if i_dom_j:
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif j_dom_i:
                dominated_set[j].append(i)
                domination_count[i] += 1

    current_front = [i for i in range(n) if domination_count[i] == 0]
    rank = 0

    while current_front:
        next_front: List[int] = []
        for i in current_front:
            ranks[i] = rank
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        rank += 1

    return ranks


def _dominates(a: List[float], b: List[float]) -> bool:
    """Return True if *a* dominates *b* (all >= and at least one >)."""
    dominated = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            dominated = True
    return dominated


def crowding_distance(
    objectives: List[List[float]],
    front_indices: List[int],
) -> List[float]:
    """Compute crowding distance for individuals in a Pareto front.

    Args:
        objectives: Full list of objective vectors for all individuals.
        front_indices: Indices of individuals in this front.

    Returns:
        List of crowding distances, one per individual in *front_indices*
        (same order as *front_indices*).

    Boundary individuals (min/max on any objective) get infinite distance.
    """
    size = len(front_indices)
    if size == 0:
        return []
    if size <= 2:
        return [math.inf] * size

    num_objectives = len(objectives[front_indices[0]])
    distances: List[float] = [0.0] * size

    idx_map = {idx: pos for pos, idx in enumerate(front_indices)}

    for m in range(num_objectives):
        sorted_positions = sorted(range(size), key=lambda p: objectives[front_indices[p]][m])

        obj_min = objectives[front_indices[sorted_positions[0]]][m]
        obj_max = objectives[front_indices[sorted_positions[-1]]][m]
        span = obj_max - obj_min

        distances[sorted_positions[0]] = math.inf
        distances[sorted_positions[-1]] = math.inf

        if span == 0.0:
            continue

        for i in range(1, size - 1):
            prev_val = objectives[front_indices[sorted_positions[i - 1]]][m]
            next_val = objectives[front_indices[sorted_positions[i + 1]]][m]
            distances[sorted_positions[i]] += (next_val - prev_val) / span

    return distances


def pareto_tournament_select(
    population: List[Any],
    ranks: List[int],
    crowding: List[float],
    n: int,
    k: int = 3,
) -> List[Any]:
    """Tournament selection using Pareto rank and crowding distance.

    Args:
        population: List of individuals.
        ranks: Pareto rank per individual (lower = better).
        crowding: Crowding distance per individual (higher = better).
        n: Number of parents to select.
        k: Tournament size.

    Returns:
        List of *n* selected individuals.

    Selection criterion: prefer lower rank. On tie, prefer higher crowding
    distance.
    """
    if not population:
        raise ValueError("pareto_tournament_select requires non-empty population")

    pop_size = len(population)
    selected: List[Any] = []

    for _ in range(n):
        candidates = random.sample(range(pop_size), min(k, pop_size))
        best = candidates[0]
        for c in candidates[1:]:
            if (ranks[c], -crowding[c]) < (ranks[best], -crowding[best]):
                best = c
        selected.append(population[best])

    return selected


def pareto_elitism(
    population: List[Any],
    ranks: List[int],
    crowding: List[float],
    n: int,
) -> List[Any]:
    """Select top *n* individuals by Pareto rank, then crowding distance.

    Fills from rank 0 first. If a rank has more individuals than remaining
    slots, keeps those with highest crowding distance.
    """
    if n <= 0 or not population:
        return []

    indexed = sorted(
        range(len(population)),
        key=lambda i: (ranks[i], -crowding[i]),
    )
    return [population[i] for i in indexed[:n]]


def compute_population_objectives(
    population: List[Any],
    score_vector_fn: Callable[[Dict[str, float]], List[float]],
) -> List[List[float]]:
    """Extract objective vectors from a population.

    Args:
        population: List of individuals with a *.scores* attribute (dict).
        score_vector_fn: Function that converts a scores dict to an objective
            vector.

    Returns:
        List of objective vectors.
    """
    result: List[List[float]] = []
    for ind in population:
        if ind.scores is not None:
            result.append(score_vector_fn(ind.scores))
        else:
            result.append([0.0] * len(score_vector_fn({})))
    return result
