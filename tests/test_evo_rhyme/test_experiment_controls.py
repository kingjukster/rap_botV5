"""Tests for evo_rhyme.experiment_controls: registry, flatten, snapshot builders."""

from __future__ import annotations

import pytest
from evo_rhyme.experiment_controls import (
    CONTROL_REGISTRY,
    LAYER_META,
    LAYER_VERSE,
    build_control_snapshot_from_couplet_args,
    build_control_snapshot_from_qd_args,
    flatten_controls,
    get_controls_by_layer,
)


def test_control_registry_has_expected_keys():
    assert "population" in CONTROL_REGISTRY
    assert "elites" in CONTROL_REGISTRY
    assert "scheme" in CONTROL_REGISTRY
    assert "init" in CONTROL_REGISTRY
    spec = CONTROL_REGISTRY["population"]
    assert spec.layer == LAYER_META
    assert spec.type == "numeric"


def test_flatten_controls_deterministic():
    snap = {"b": 2, "a": 1, "c": "x"}
    out = flatten_controls(snap)
    assert list(out.keys()) == sorted(snap.keys())
    assert out["a"] == 1
    assert out["b"] == 2
    assert out["c"] == "x"


def test_flatten_controls_handles_nested_dict():
    snap = {"a": 1, "weights": {"x": 0.5}}
    out = flatten_controls(snap)
    assert out["a"] == 1
    assert out["weights"] == {"x": 0.5}


def test_build_control_snapshot_from_couplet_args():
    class Args:
        theme = "pressure,mask"
        population = 100
        generations = 20
        elites = 5
        immigrants = 10
        init = "mixed"
        use_embeddings = False
        embedding_weight = 0.5
        multiobjective = False
        use_niching = False
        min_fluency = 0.6
        min_semantic = 0.25
        min_lexical = 0.0
        min_ngram = None
        lm_fluency = False
        lm_fluency_weight = 0.5
        style_weight = 0.1
        require_theme = False

    defaults = {"population": 100, "generations": 10, "tournament_k": 3}
    snap = build_control_snapshot_from_couplet_args(Args(), defaults)
    assert snap["runner"] == "couplet"
    assert snap["theme"] == "pressure,mask"
    assert snap["population"] == 100
    assert snap["generations"] == 20
    assert snap["init"] == "mixed"


def test_build_control_snapshot_from_qd_args():
    class Args:
        theme = "crown"
        population = 120
        generations = 30
        elites = 5
        immigrants = 20
        scheme = "AABB"
        num_lines = 4
        init = "lm"
        lm_budget = 25
        use_embeddings = False
        embedding_weight = 0.4
        proposer_model = "gpt-4o-mini"
        proposer_backend = "openai"
        min_fluency = 0.3
        min_semantic = 0.0
        line_pop = 1500
        line_gens = 3
        line_seeds = 80
        line_lm_budget = 15
        compose_ratio = 0.5
        emitter_strategy = "multi"
        archive_mode = "compact_style"
        style_genome = True
        prompt_genome = True
        novelty_weight = 0.3
        fast_mode = True
        graph_top_k = 24
        expensive_top_k = 40
        graph_edge_mode = "phonetic"
        curriculum_switch_gen = 20
        prompt_llm_fraction = 0.2

    defaults = {}
    snap = build_control_snapshot_from_qd_args(Args(), defaults)
    assert snap["runner"] == "qd"
    assert snap["scheme"] == "AABB"
    assert snap["num_lines"] == 4
    assert snap["lm_budget"] == 25


def test_get_controls_by_layer():
    meta = get_controls_by_layer(LAYER_META)
    assert "population" in meta
    assert "elites" in meta
    verse = get_controls_by_layer(LAYER_VERSE)
    assert "scheme" in verse
