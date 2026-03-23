"""Tests for evo_rhyme.block_builder: build_4bar_from_couplets, build_4bar_batch."""

import random

import pytest

from evo_rhyme.block_builder import build_4bar_batch, build_4bar_from_couplets
from evo_rhyme.couplet_archive import CoupletArchive, ScoredCouplet
from evo_rhyme.individual import VerseIndividual


def _make_couplet_archive_with_two_groups():
    """Archive with two rhyme groups for AABB 4-bar assembly."""
    archive = CoupletArchive(max_per_group=10)
    # Group A
    archive.add(ScoredCouplet.from_couplet(
        "The burns of life bring many concerns",
        "We learn from pain as the whole world turns",
        fitness=0.85,
    ))
    archive.add(ScoredCouplet.from_couplet(
        "I got the flow when I step in the spot",
        "You know I rock it hard when I hit the block",
        fitness=0.80,
    ))
    # Group B (different rhyme)
    archive.add(ScoredCouplet.from_couplet(
        "Diamonds on my wrist they shining bright",
        "Running through the city every night",
        fitness=0.78,
    ))
    archive.add(ScoredCouplet.from_couplet(
        "From the bottom to the top we rise",
        "Looking at the world through different eyes",
        fitness=0.75,
    ))
    return archive


def test_build_4bar_from_couplets_aabb():
    """build_4bar_from_couplets produces 4-line verse for AABB."""
    random.seed(42)
    archive = _make_couplet_archive_with_two_groups()
    verse = build_4bar_from_couplets(archive, scheme="AABB")
    assert verse is not None
    assert isinstance(verse, VerseIndividual)
    assert len(verse.lines) == 4
    assert verse.metadata.get("origin") == "couplet_assembly"


def test_build_4bar_from_couplets_abab():
    """build_4bar_from_couplets works with ABAB scheme."""
    random.seed(7)
    archive = _make_couplet_archive_with_two_groups()
    verse = build_4bar_from_couplets(archive, scheme="ABAB")
    assert verse is not None
    assert len(verse.lines) == 4


def test_build_4bar_from_couplets_insufficient_returns_none():
    """Returns None when archive has insufficient variety."""
    archive = CoupletArchive()
    archive.add(ScoredCouplet.from_couplet("only one", "couplet here", fitness=0.8))
    verse = build_4bar_from_couplets(archive, scheme="AABB")
    assert verse is None


def test_build_4bar_batch():
    """build_4bar_batch returns multiple verses."""
    random.seed(5)
    archive = _make_couplet_archive_with_two_groups()
    verses = build_4bar_batch(archive, count=3, scheme="AABB")
    assert len(verses) >= 1
    assert all(isinstance(v, VerseIndividual) for v in verses)
    assert all(len(v.lines) == 4 for v in verses)
