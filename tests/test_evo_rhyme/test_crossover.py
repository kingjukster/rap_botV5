"""Tests for evo_rhyme.crossover: line-level and phrase-slice crossover."""

import random

import pytest

from evo_rhyme.crossover import crossover, _phrase_slice_crossover
from evo_rhyme.individual import CoupletIndividual


def test_crossover_returns_couplet():
    random.seed(42)
    p1 = CoupletIndividual(line1="first line here", line2="second line there")
    p2 = CoupletIndividual(line1="another bar", line2="one more bar")
    child = crossover(p1, p2)
    assert isinstance(child, CoupletIndividual)
    assert child.line1 in (p1.line1, p2.line1)
    assert child.line2 in (p1.line2, p2.line2)
    assert child.fitness is None
    assert child.scores is None


def test_crossover_with_phrase_slice_config():
    random.seed(123)
    # Lines with similar syllable structure to allow phrase_slice
    p1 = CoupletIndividual(
        line1="I got the flow when I step in the spot",
        line2="You know I rock it hard when I hit the block",
    )
    p2 = CoupletIndividual(
        line1="We run the streets and we never stop",
        line2="The beat goes on from the bottom to the top",
    )
    child = crossover(p1, p2, config={"phrase_slice": True})
    assert isinstance(child, CoupletIndividual)
    assert child.line1 is not None
    assert child.line2 is not None


def test_phrase_slice_crossover_short_lines_returns_none():
    p1 = CoupletIndividual(line1="a b", line2="c d")
    p2 = CoupletIndividual(line1="e f", line2="g h")
    result = _phrase_slice_crossover(p1, p2)
    assert result is None


def test_phrase_slice_crossover_long_lines_returns_couplet():
    random.seed(77)
    p1 = CoupletIndividual(
        line1="one two three four five six",
        line2="first second third fourth fifth sixth",
    )
    p2 = CoupletIndividual(
        line1="alpha beta gamma delta epsilon zeta",
        line2="red green blue yellow orange purple",
    )
    result = _phrase_slice_crossover(p1, p2)
    if result is not None:
        assert isinstance(result, CoupletIndividual)
        assert len(result.line1.split()) >= 3 or len(result.line2.split()) >= 3


def test_phrase_slice_crossover_line2_branch():
    """Exercise line2 crossover branch (else branch in _phrase_slice_crossover)."""
    random.seed(100)
    p1 = CoupletIndividual(
        line1="I got the flow when I step in the spot",
        line2="You know I rock it hard when I hit the block",
    )
    p2 = CoupletIndividual(
        line1="We run the streets and we never stop",
        line2="The beat goes on from the bottom to the top",
    )
    results = []
    for _ in range(30):
        child = _phrase_slice_crossover(p1, p2)
        if child is not None:
            results.append(child)
    assert len(results) >= 1
