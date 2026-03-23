"""Tests for evo_rhyme.couplet_archive: ScoredCouplet, CoupletArchive, from_results_json."""

import json
import random
from pathlib import Path

import pytest

from evo_rhyme.couplet_archive import (
    CoupletArchive,
    ScoredCouplet,
)


def test_scored_couplet_from_couplet():
    """ScoredCouplet.from_couplet computes end tails from lines."""
    sc = ScoredCouplet.from_couplet(
        line1="The burns of life bring many concerns",
        line2="We learn from pain as the whole world turns",
        fitness=0.75,
        scores={"end_rhyme": 0.9, "fluency": 0.7},
    )
    assert sc.line1 == "The burns of life bring many concerns"
    assert sc.line2 == "We learn from pain as the whole world turns"
    assert sc.fitness == 0.75
    assert sc.scores.get("end_rhyme") == 0.9
    assert sc.end_tail_1 != "__unknown__" or sc.end_tail_2 != "__unknown__"


def test_couplet_archive_add_and_top_k():
    """CoupletArchive.add and top_k work correctly."""
    archive = CoupletArchive(max_per_group=10)
    c1 = ScoredCouplet.from_couplet("line a one", "line a two", fitness=0.8)
    c2 = ScoredCouplet.from_couplet("line b one", "line b two", fitness=0.6)
    c3 = ScoredCouplet.from_couplet("line c one", "line c two", fitness=0.9)
    archive.add(c1)
    archive.add(c2)
    archive.add(c3)
    assert archive.size() == 3
    top = archive.top_k(2)
    assert len(top) == 2
    assert top[0].fitness >= top[1].fitness


def test_couplet_archive_eligible_pairs():
    """eligible_pairs returns pairs of group keys for 4-bar assembly."""
    archive = CoupletArchive(max_per_group=10)
    archive.add(ScoredCouplet.from_couplet("a1", "a2", fitness=0.8))
    archive.add(ScoredCouplet.from_couplet("a3", "a4", fitness=0.7))
    archive.add(ScoredCouplet.from_couplet("b1", "b2", fitness=0.75))
    archive.add(ScoredCouplet.from_couplet("b3", "b4", fitness=0.65))
    pairs = archive.eligible_pairs(min_per_group=1)
    assert len(pairs) >= 1


def test_couplet_archive_from_results_json(tmp_path):
    """from_results_json loads CoupletArchive from run_couplet_evolution output."""
    data = {
        "config": {"theme": "test"},
        "candidates": [
            {"line1": "first line here", "line2": "second line there", "fitness": 0.8, "scores": {}},
            {"line1": "another bar one", "line2": "another bar two", "fitness": 0.7, "scores": {}},
        ],
    }
    path = tmp_path / "couplet_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    archive = CoupletArchive.from_results_json(path)
    assert archive.size() == 2
    assert archive.group_count() >= 1


def test_couplet_archive_sample():
    """sample returns fitness-proportional couplets."""
    random.seed(42)
    archive = CoupletArchive(max_per_group=10)
    for i in range(5):
        archive.add(ScoredCouplet.from_couplet(f"line{i}a", f"line{i}b", fitness=0.5 + i * 0.1))
    samples = archive.sample(k=3)
    assert len(samples) == 3
    assert all(isinstance(s, ScoredCouplet) for s in samples)
