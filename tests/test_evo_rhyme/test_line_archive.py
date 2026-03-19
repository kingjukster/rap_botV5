"""Tests for evo_rhyme.line_archive: ScoredLine, compute_line_composite, LineArchive. Area 8."""

import pytest

from evo_rhyme.line_archive import (
    LINE_COMPOSITE_WEIGHTS,
    LineArchive,
    ScoredLine,
    compute_line_composite,
)


def test_scored_line_from_text():
    line = ScoredLine.from_text("The burns of life bring many concerns")
    assert line.text == "The burns of life bring many concerns"
    assert line.end_word == "concerns"
    assert line.syllables >= 1
    assert line.end_tail != ""


def test_scored_line_dataclass():
    line = ScoredLine(
        text="hello world",
        end_tail="ER1 L D",
        end_word="world",
        syllables=2,
        fluency=0.8,
        composite=0.5,
    )
    assert line.fluency == 0.8
    assert line.composite == 0.5


def test_compute_line_composite():
    line = ScoredLine(
        text="x",
        end_tail="X",
        end_word="x",
        syllables=1,
        fluency=0.5,
        lm_fluency=0.5,
        semantic=0.5,
        novelty=0.5,
        chain_potential=0.5,
    )
    score = compute_line_composite(line)
    assert score >= 0
    assert score <= 1.0


def test_compute_line_composite_custom_weights():
    line = ScoredLine(
        text="x",
        end_tail="X",
        end_word="x",
        syllables=1,
        fluency=1.0,
    )
    score = compute_line_composite(line, weights={"fluency": 1.0})
    assert score == 1.0


def test_line_archive_add_and_get_group():
    archive = LineArchive(max_per_group=10)
    line1 = ScoredLine("first line here", "AY1 N", "here", 3, composite=0.8)
    line2 = ScoredLine("second line there", "EH1 R", "there", 3, composite=0.6)
    assert archive.add(line1) is True
    assert archive.add(line2) is True
    assert archive.add(line1) is False
    assert len(archive.get_group("AY1 N")) == 1
    assert len(archive.get_group("EH1 R")) == 1
    assert archive.size() == 2


def test_line_archive_eligible_tails():
    archive = LineArchive()
    line1 = ScoredLine("a one", "X", "one", 1, composite=0.5)
    line2 = ScoredLine("a two", "X", "two", 1, composite=0.5)
    line3 = ScoredLine("b one", "Y", "one", 1, composite=0.5)
    archive.add(line1)
    archive.add(line2)
    archive.add(line3)
    eligible = archive.eligible_tails(min_size=2)
    assert "X" in eligible
    assert "Y" not in eligible


def test_line_archive_sample_rhyming_pair():
    archive = LineArchive()
    line1 = ScoredLine("first rhyme", "AY1 M", "rhyme", 2, composite=0.8)
    line2 = ScoredLine("second rhyme", "AY1 M", "rhyme", 2, composite=0.7)
    archive.add(line1)
    archive.add(line2)
    pair = archive.sample_rhyming_pair(tail="AY1 M")
    assert pair is not None
    assert len(pair) == 2
    assert pair[0].end_tail == "AY1 M"
    assert pair[1].end_tail == "AY1 M"


def test_line_archive_add_evicts_when_full_and_better():
    """When group is full, add evicts lowest if new line has higher composite."""
    archive = LineArchive(max_per_group=2)
    low = ScoredLine("low score line", "X", "line", 3, composite=0.3)
    mid = ScoredLine("mid score line", "X", "line", 3, composite=0.5)
    high = ScoredLine("high score line", "X", "line", 3, composite=0.9)
    assert archive.add(low) is True
    assert archive.add(mid) is True
    assert archive.add(high) is True
    group = archive.get_group("X")
    assert len(group) == 2
    composites = [l.composite for l in group]
    assert 0.9 in composites
    assert 0.5 in composites
    assert 0.3 not in composites


def test_line_archive_add_batch():
    """add_batch returns count of lines actually added."""
    archive = LineArchive(max_per_group=5)
    lines = [
        ScoredLine("one a", "A", "a", 1, composite=0.5),
        ScoredLine("two a", "A", "a", 1, composite=0.6),
        ScoredLine("three b", "B", "b", 1, composite=0.4),
    ]
    n = archive.add_batch(lines)
    assert n == 3
    assert archive.size() == 3


def test_line_archive_sample_rhyming_pair_no_tail():
    """sample_rhyming_pair with tail=None picks a random eligible group."""
    archive = LineArchive()
    for i in range(3):
        archive.add(ScoredLine(f"line {i} rhyme", "AY1 M", "rhyme", 2, composite=0.5 + i * 0.1))
    pair = archive.sample_rhyming_pair(tail=None, min_group_size=2)
    assert pair is not None
    assert len(pair) == 2


def test_line_archive_top_k_global():
    """top_k_global returns top k lines across all groups by composite."""
    archive = LineArchive()
    archive.add(ScoredLine("low", "A", "low", 1, composite=0.3))
    archive.add(ScoredLine("high", "B", "high", 1, composite=0.9))
    archive.add(ScoredLine("mid", "A", "mid", 1, composite=0.6))
    top = archive.top_k_global(2)
    assert len(top) == 2
    assert top[0].composite >= top[1].composite
    assert top[0].text == "high"
