"""Tests for runtime learned-policy loading and epsilon exploration helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from evo_rhyme.policy_runtime import (
    apply_controls_to_args,
    load_learned_policy,
    maybe_epsilon_perturb,
    resolve_policy_controls,
)


def test_load_learned_policy_reads_recommended_controls(tmp_path: Path):
    policy = tmp_path / "learned_policy.json"
    policy.write_text('{"recommended_controls":{"population":123,"elites":7}}', encoding="utf-8")
    controls, source = load_learned_policy(str(policy), tmp_path)
    assert controls["population"] == 123
    assert controls["elites"] == 7
    assert str(policy) in source


def test_apply_controls_to_args_skips_protected():
    args = Namespace(population=100, elites=5, theme="dark")
    updates = apply_controls_to_args(
        args,
        {"population": 120, "elites": 3, "theme": "light"},
        protected_keys={"theme"},
    )
    assert updates == 2
    assert args.population == 120
    assert args.elites == 3
    assert args.theme == "dark"


def test_maybe_epsilon_perturb_with_epsilon_zero_returns_false():
    args = Namespace(population=100, generations=20, embedding_weight=0.5)
    changed = maybe_epsilon_perturb(args, epsilon=0.0, protected_keys=set())
    assert changed is False


def test_resolve_policy_controls_top_k_sampling(tmp_path: Path):
    policy = tmp_path / "learned_policy.json"
    policy.write_text(
        '{"policy_version":"v42","policy_hash":"abc123","top_configs":['
        '{"controls":{"population":120},"score":0.9},'
        '{"controls":{"population":80},"score":0.6}'
        ']}',
        encoding="utf-8",
    )
    controls, meta = resolve_policy_controls(str(policy), tmp_path)
    assert "population" in controls
    assert meta["policy_version"] == "v42"
    assert meta["policy_hash"] == "abc123"
    assert meta["sampled_policy_rank"] in (1, 2)

