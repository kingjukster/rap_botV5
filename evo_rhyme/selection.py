"""
evo_rhyme/selection.py

Selection operators for evolutionary rhyme: elitism, tournament selection,
and random immigrant injection.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Set

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
    import random

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
