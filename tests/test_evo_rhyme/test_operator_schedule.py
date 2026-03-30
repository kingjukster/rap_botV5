"""Tests for operator weight blending from event counts."""

from __future__ import annotations

import pytest

from evo_rhyme.operator_schedule import blend_mutation_weights_from_counts


def test_blend_mutation_weights_preserves_sum():
    base = {"a": 0.5, "b": 0.3, "c": 0.2}
    counts = {"a": 100, "b": 10, "c": 10}
    out = blend_mutation_weights_from_counts(base, counts)
    assert out is not None
    assert pytest.approx(sum(out.values())) == sum(base.values())
    assert out["a"] > out["c"]


def test_blend_mutation_weights_returns_none_on_empty_counts():
    assert blend_mutation_weights_from_counts({"a": 1.0}, {}) is None


def test_diversify_cross_run_candidates_caps_per_run():
    from evo_rhyme.db import _diversify_cross_run_candidates

    rows = [
        {"run_id": 1, "lines": ["a", "b", "c", "d"], "fitness": 0.9},
        {"run_id": 1, "lines": ["e", "f", "g", "h"], "fitness": 0.89},
        {"run_id": 1, "lines": ["i", "j", "k", "l"], "fitness": 0.88},
        {"run_id": 2, "lines": ["a", "b", "c", "d"], "fitness": 0.87},
        {"run_id": 3, "lines": ["m", "n", "o", "p"], "fitness": 0.5},
    ]
    out = _diversify_cross_run_candidates(rows, limit=4, max_per_source_run=2)
    assert len(out) <= 4
    assert sum(1 for r in out if r["run_id"] == 1) <= 2
    assert out[0]["fitness"] >= out[-1]["fitness"]
