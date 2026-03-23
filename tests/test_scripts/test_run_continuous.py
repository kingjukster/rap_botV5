"""Tests for scripts/run_continuous policy and config loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_sample_elite_config_no_policy():
    """Elite replay returns None when policy file does not exist."""
    from scripts.run_continuous import _sample_elite_config

    result = _sample_elite_config(ROOT / "nonexistent_policy.json")
    assert result is None


def test_sample_elite_config_empty_policy(tmp_path):
    """Elite replay returns None when policy has no viable top_configs."""
    import json

    from scripts.run_continuous import _sample_elite_config

    policy_path = tmp_path / "learned_policy.json"
    policy_path.write_text(json.dumps({"top_configs": [], "recommended_controls": {}}), encoding="utf-8")
    assert _sample_elite_config(policy_path) is None


def test_sample_elite_config_all_zero_scores(tmp_path):
    """Elite replay returns None when all top_configs have score <= 0."""
    import json

    from scripts.run_continuous import _sample_elite_config

    policy_path = tmp_path / "learned_policy.json"
    policy_path.write_text(
        json.dumps({
            "top_configs": [
                {"controls": {"theme": "x", "population": 80}, "score": 0.0},
                {"controls": {"theme": "y", "population": 60}, "score": -0.1},
            ],
        }),
        encoding="utf-8",
    )
    assert _sample_elite_config(policy_path) is None


def test_load_configs_elite_replay_metadata():
    """Config loads elite_replay_fraction and epsilon_decay from YAML."""
    config_path = ROOT / "config" / "continuous_runs.yaml"
    if not config_path.exists():
        pytest.skip("config/continuous_runs.yaml not found")
    from scripts.run_continuous import load_configs

    _, metadata = load_configs(config_path)
    assert "elite_replay_fraction" in metadata
    assert metadata["elite_replay_fraction"] == 0.3
    assert "epsilon_decay_factor" in metadata
    assert metadata["epsilon_decay_factor"] == 0.98
    assert "epsilon_decay_cap" in metadata


def test_load_configs_with_seeds():
    """Configs are expanded per seed when seeds are in YAML."""
    from scripts.run_continuous import load_configs

    config_path = ROOT / "config" / "continuous_runs.yaml"
    if not config_path.exists():
        pytest.skip("config/continuous_runs.yaml not found")
    configs, metadata = load_configs(config_path)
    assert metadata["seeds"] == [1, 2, 3]
    assert metadata["policy_mode"] == "learned"
    # 13 base configs × 3 seeds = 39
    assert len(configs) >= 13
    assert len(configs) == 13 * 3  # one per seed
    # Each config has seed
    for c in configs:
        assert "seed" in c
        assert c["seed"] in (1, 2, 3)
    # Population bumped to 80 for most arms
    theme_configs = [c for c in configs if c["arm"] == "theme_pressure"]
    assert len(theme_configs) == 3  # one per seed
    assert all(c["population"] == 80 for c in theme_configs)


def test_load_configs_no_seeds():
    """Configs without seeds are not expanded."""
    import yaml

    from scripts.run_continuous import load_configs

    path = Path(tempfile.gettempdir()) / "rapbot_test_no_seeds.yaml"
    path.write_text(
        yaml.safe_dump({
            "runs": [
                {"arm": "test", "theme": "x", "population": 60, "scheme": "AABB", "init": "mixed"},
            ],
        }),
        encoding="utf-8",
    )
    try:
        configs, metadata = load_configs(path)
        assert len(configs) == 1
        assert "seed" not in configs[0]
        assert metadata["seeds"] == []
    finally:
        path.unlink(missing_ok=True)
