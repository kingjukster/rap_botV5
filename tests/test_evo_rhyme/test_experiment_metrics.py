"""Tests for evo_rhyme.experiment_metrics: fitness vector and aggregate_outcome."""

from __future__ import annotations

import pytest
from evo_rhyme.experiment_metrics import (
    FITNESS_VECTOR_KEYS,
    FITNESS_VECTOR_SCHEMA_VERSION,
    aggregate_outcome,
    fitness_vector_from_scores,
)


def test_fitness_vector_schema_version():
    assert FITNESS_VECTOR_SCHEMA_VERSION == "v1"
    assert set(FITNESS_VECTOR_KEYS) == {"rhyme", "flow", "semantic", "novelty", "punchline"}


def test_fitness_vector_from_scores_empty():
    out = fitness_vector_from_scores(None)
    assert set(out.keys()) == set(FITNESS_VECTOR_KEYS)
    assert all(v == 0.0 for v in out.values())


def test_fitness_vector_from_scores_couplet_style():
    scores = {
        "end_rhyme": 0.8,
        "internal_rhyme": 0.7,
        "syllable_balance": 0.9,
        "stress_alignment": 0.6,
        "semantic": 0.5,
        "coherence": 0.6,
        "novelty": 0.4,
        "punchline": 0.3,
    }
    out = fitness_vector_from_scores(scores)
    assert out["rhyme"] > 0
    assert out["flow"] > 0
    assert out["semantic"] > 0
    assert out["novelty"] >= 0
    assert out["punchline"] == 0.3
    assert all(0 <= v <= 1 for v in out.values())


def test_aggregate_outcome_empty():
    out = aggregate_outcome([], mode="best")
    assert out["fitness"] == 0.0
    assert set(out["fitness_vector"].keys()) == set(FITNESS_VECTOR_KEYS)


def test_aggregate_outcome_best():
    candidates = [
        {"fitness": 0.5, "scores": {"end_rhyme": 0.5, "punchline": 0.2}},
        {"fitness": 0.9, "scores": {"end_rhyme": 0.9, "punchline": 0.8}},
    ]
    out = aggregate_outcome(candidates, mode="best")
    assert out["fitness"] == 0.9
    assert out["fitness_vector"]["punchline"] == 0.8


def test_aggregate_outcome_top_k_mean():
    candidates = [
        {"fitness": 0.4, "scores": {}},
        {"fitness": 0.6, "scores": {}},
        {"fitness": 0.8, "scores": {}},
    ]
    out = aggregate_outcome(candidates, mode="top_k_mean", top_k=2)
    assert out["fitness"] == pytest.approx(0.7)  # (0.8 + 0.6) / 2
