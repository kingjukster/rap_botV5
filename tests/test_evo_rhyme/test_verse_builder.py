"""Tests for evo_rhyme.verse_builder: build_verse_from_archive, build_verse_batch. Area 8."""

import random

import pytest

from evo_rhyme.individual import VerseIndividual
from evo_rhyme.line_archive import LineArchive, ScoredLine
from evo_rhyme.verse_builder import (
    build_verse_batch,
    build_verse_from_archive,
)


def _make_archive_with_two_groups():
    """Archive with two rhyme groups (A and B), each with 2+ lines for AABB."""
    archive = LineArchive(max_per_group=10)
    # Group 1: rhyme tail "ER1 N Z" (concerns, turns)
    archive.add(ScoredLine("The burns of life bring many concerns", "ER1 N Z", "concerns", 8, composite=0.8))
    archive.add(ScoredLine("We learn from pain as the whole world turns", "ER1 N Z", "turns", 8, composite=0.7))
    # Group 2: rhyme tail "AH1 R K" (dark, mark)
    archive.add(ScoredLine("I fight through pressure every night inside the dark", "AH1 R K", "dark", 10, composite=0.8))
    archive.add(ScoredLine("I carve a lane through the pain and leave a mark", "AH1 R K", "mark", 10, composite=0.7))
    return archive


def test_build_verse_from_archive_aabb():
    random.seed(42)
    archive = _make_archive_with_two_groups()
    verse = build_verse_from_archive(archive, scheme="AABB", num_lines=4)
    assert verse is not None
    assert isinstance(verse, VerseIndividual)
    assert len(verse.lines) == 4
    assert verse.metadata.get("origin") == "line_archive_assembly"


def test_build_verse_from_archive_insufficient_groups_returns_none():
    archive = LineArchive()
    archive.add(ScoredLine("only one group line one", "X", "one", 5, composite=0.5))
    archive.add(ScoredLine("only one group line two", "X", "two", 5, composite=0.5))
    verse = build_verse_from_archive(archive, scheme="AABB", num_lines=4)
    assert verse is None


def test_build_verse_from_archive_abab():
    """ABAB scheme produces interleaved rhyme groups."""
    random.seed(7)
    archive = _make_archive_with_two_groups()
    verse = build_verse_from_archive(archive, scheme="ABAB", num_lines=4)
    assert verse is not None
    assert isinstance(verse, VerseIndividual)
    assert len(verse.lines) == 4
    assert verse.metadata.get("origin") == "line_archive_assembly"


def test_build_verse_from_archive_used_pairs_avoids_reuse():
    """used_pairs prevents reusing the same rhyme group pair."""
    random.seed(0)
    archive = _make_archive_with_two_groups()
    verse1 = build_verse_from_archive(archive, scheme="AABB", num_lines=4, used_pairs=set())
    assert verse1 is not None
    used = {frozenset(["ER1 N Z", "AH1 R K"])}
    verse2 = build_verse_from_archive(archive, scheme="AABB", num_lines=4, used_pairs=used)
    assert verse2 is not None


def test_build_verse_from_archive_num_lines_extends_scheme():
    """When num_lines > len(scheme), scheme is extended (e.g. AABB -> AABBAABB for 8)."""
    archive = LineArchive(max_per_group=10)
    for tail, line_texts in [
        ("ER1 N Z", ["line one a", "line two a", "line three a", "line four a"]),
        ("AH1 R K", ["line one b", "line two b", "line three b", "line four b"]),
    ]:
        for t in line_texts:
            archive.add(ScoredLine(t, tail, t.split()[-1], 5, composite=0.6))
    random.seed(1)
    verse = build_verse_from_archive(archive, scheme="AABB", num_lines=8)
    assert verse is not None
    assert len(verse.lines) == 8


def test_build_verse_batch():
    """build_verse_batch returns multiple verses and tracks used_pairs."""
    random.seed(2)
    archive = _make_archive_with_two_groups()
    verses = build_verse_batch(archive, count=2, scheme="AABB", num_lines=4, optimize_transitions=False)
    assert len(verses) == 2
    assert all(isinstance(v, VerseIndividual) for v in verses)
    assert all(len(v.lines) == 4 for v in verses)


def test_build_verse_batch_with_optimize_transitions():
    """build_verse_batch with optimize_transitions=True runs without error (LM may be no-op)."""
    random.seed(3)
    archive = _make_archive_with_two_groups()
    verses = build_verse_batch(
        archive, count=1, scheme="AABB", num_lines=4,
        optimize_transitions=True, lm_budget={"remaining": 0},
    )
    assert len(verses) == 1
    assert len(verses[0].lines) == 4
