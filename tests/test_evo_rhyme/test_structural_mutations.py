"""Tests for evo_rhyme.structural_mutations: swap_couplets, rewrite_transition_line."""

import pytest

from evo_rhyme.individual import VerseIndividual
from evo_rhyme.structural_mutations import (
    swap_couplets,
    couplet_swap_crossover,
    rewrite_transition_line,
)


def test_swap_couplets():
    """swap_couplets swaps lines 1-2 with lines 3-4."""
    ind = VerseIndividual(
        lines=["line1", "line2", "line3", "line4"],
        metadata={"origin": "test"},
    )
    result = swap_couplets(ind)
    assert result.lines == ["line3", "line4", "line1", "line2"]
    assert result.metadata.get("mutation") == "swap_couplets"


def test_swap_couplets_short_verse_unchanged():
    """swap_couplets returns unchanged for verses with < 4 lines."""
    ind = VerseIndividual(lines=["a", "b", "c"])
    result = swap_couplets(ind)
    assert result.lines == ["a", "b", "c"]


def test_couplet_swap_crossover():
    """couplet_swap_crossover produces child from both parents."""
    p1 = VerseIndividual(lines=["p1a", "p1b", "p1c", "p1d"])
    p2 = VerseIndividual(lines=["p2a", "p2b", "p2c", "p2d"])
    child = couplet_swap_crossover(p1, p2)
    assert len(child.lines) == 4
    assert (
        (child.lines[:2] == p1.lines[:2] and child.lines[2:] == p2.lines[2:])
        or (child.lines[:2] == p2.lines[:2] and child.lines[2:] == p1.lines[2:])
    )


def test_rewrite_transition_no_budget_returns_none():
    """rewrite_transition_line returns None when lm_budget exhausted."""
    ind = VerseIndividual(lines=["line1", "line2", "line3", "line4"])
    result = rewrite_transition_line(ind, boundary_idx=1, lm_budget={"remaining": 0})
    assert result is None
