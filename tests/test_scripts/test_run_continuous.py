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
    import yaml

    from scripts.run_continuous import load_configs

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _, metadata = load_configs(config_path)
    assert "elite_replay_fraction" in metadata
    assert metadata["elite_replay_fraction"] == float(raw.get("elite_replay_fraction", 0.3))
    assert "epsilon_decay_factor" in metadata
    assert metadata["epsilon_decay_factor"] == float(raw.get("epsilon_decay_factor", 0.98))
    assert "epsilon_decay_cap" in metadata
    assert metadata["default_seed_from_archive"] == int(raw.get("seed_from_archive", 0))
    assert "elite_replay_schemes" in metadata
    assert metadata["elite_replay_schemes"] == []


def test_load_configs_with_seeds():
    """Configs are expanded per seed when seeds are in YAML."""
    import yaml

    from scripts.run_continuous import load_configs

    config_path = ROOT / "config" / "continuous_runs.yaml"
    if not config_path.exists():
        pytest.skip("config/continuous_runs.yaml not found")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = raw.get("runs") or []
    n_base = len(runs)
    configs, metadata = load_configs(config_path)
    assert metadata["seeds"] == [1, 2, 3]
    assert metadata["policy_mode"] == raw.get("policy_mode", "learned")
    assert len(configs) == n_base * 3  # one per seed
    # Each config has seed
    for c in configs:
        assert "seed" in c
        assert c["seed"] in (1, 2, 3)
    tp_row = next((r for r in runs if r.get("arm") == "theme_pressure"), None)
    assert tp_row is not None
    exp_pop = int(tp_row.get("population", 60))
    theme_configs = [c for c in configs if c["arm"] == "theme_pressure"]
    assert len(theme_configs) == 3  # one per seed
    assert all(c["population"] == exp_pop for c in theme_configs)


def test_parallel_arg_default():
    """--parallel defaults to 1."""
    from scripts.run_continuous import parse_args

    # Must patch sys.argv since parse_args reads it
    import sys
    orig = sys.argv
    sys.argv = ["run_continuous.py"]
    try:
        args = parse_args()
        assert getattr(args, "parallel", 1) == 1
    finally:
        sys.argv = orig


def test_parallel_arg_parsed():
    """--parallel N is parsed correctly."""
    from scripts.run_continuous import parse_args

    import sys
    orig = sys.argv
    sys.argv = ["run_continuous.py", "--parallel", "3"]
    try:
        args = parse_args()
        assert args.parallel == 3
    finally:
        sys.argv = orig


def test_plateau_restart_args_default():
    """Adaptive plateau restart args have safe defaults."""
    from scripts.run_continuous import parse_args

    import sys
    orig = sys.argv
    sys.argv = ["run_continuous.py"]
    try:
        args = parse_args()
        assert args.restart_on_plateau is True
        assert args.plateau_window == 5
        assert args.plateau_min_improvement == 0.005
        assert args.plateau_restart_runs == 2
    finally:
        sys.argv = orig


def test_make_plateau_restart_cfg_is_deterministic():
    """Restart cfg mutation is deterministic and preserves key controls."""
    from scripts.run_continuous import _make_plateau_restart_cfg

    base = {
        "arm": "theme_pressure",
        "theme": "pressure,mask,survival",
        "population": 80,
        "generations": 20,
        "scheme": "ABAB",
        "init": "random",
        "immigrants": 25,
        "seed_from_archive": 1,
        "seed": 7,
    }
    out = _make_plateau_restart_cfg(base, run_num=42)
    assert out["config_source"] == "plateau_restart"
    assert out["arm"] == "theme_pressure_restart"
    assert out["scheme"] == "AABB"
    assert out["init"] == "mixed"
    assert out["generations"] == 15
    assert out["immigrants"] == 35
    assert out["seed_from_archive"] == 3
    assert out["seed"] == 100049


def test_make_plateau_restart_cfg_assigns_seed_when_missing():
    """Restart cfg assigns deterministic seed if base config had none."""
    from scripts.run_continuous import _make_plateau_restart_cfg

    base = {
        "arm": "theme_crown",
        "theme": "crown,empire,power",
        "population": 80,
        "generations": 20,
        "scheme": "AABB",
        "init": "mixed",
        "immigrants": 20,
        "seed_from_archive": 3,
    }
    out = _make_plateau_restart_cfg(base, run_num=11)
    assert out["seed"] == (11 * 9973) % 2147483647


def test_sample_elite_config_respects_scheme_filter(tmp_path):
    """Elite replay skips configs whose scheme is not in elite_replay_schemes."""
    import json

    from scripts.run_continuous import _sample_elite_config

    policy_path = tmp_path / "learned_policy.json"
    policy_path.write_text(
        json.dumps({
            "top_configs": [
                {
                    "controls": {"theme": "a", "scheme": "ABAB", "population": 80, "generations": 20, "init": "mixed"},
                    "score": 1.0,
                },
            ],
        }),
        encoding="utf-8",
    )
    assert (
        _sample_elite_config(
            policy_path,
            elite_replay_schemes=["AABB"],
        )
        is None
    )
    picked = _sample_elite_config(policy_path, elite_replay_schemes=["ABAB"])
    assert picked is not None
    assert picked["scheme"] == "ABAB"


def test_clamp_lm_budgets_disables_lm_when_continuous_defaults_zero():
    """DB/model must not enable LM when continuous YAML sets both budgets to 0."""
    from scripts.run_continuous import _clamp_lm_budgets_for_continuous

    assert _clamp_lm_budgets_for_continuous(50, 12, default_lm_budget=0, default_line_lm_budget=0) == (0, 0)
    assert _clamp_lm_budgets_for_continuous(0, 0, default_lm_budget=5, default_line_lm_budget=2) == (5, 2)
    assert _clamp_lm_budgets_for_continuous(10, 1, default_lm_budget=5, default_line_lm_budget=2) == (10, 2)


def test_build_evolution_cmd_passes_novelty_weight():
    from scripts.run_continuous import _build_evolution_cmd

    cfg = {
        "theme": "x,y",
        "population": 60,
        "generations": 20,
        "scheme": "AABB",
        "init": "mixed",
        "arm": "a1",
        "lm_budget": 0,
        "line_lm_budget": 0,
        "immigrants": 20,
        "novelty_weight": 0.45,
    }
    cmd, _env = _build_evolution_cmd(cfg, use_docker=False)
    assert "--novelty-weight" in cmd
    assert "0.45" in cmd


def test_build_evolution_cmd_passes_experiment_and_early_stop():
    from scripts.run_continuous import _build_evolution_cmd

    cfg = {
        "theme": "x,y",
        "population": 60,
        "generations": 20,
        "scheme": "AABB",
        "init": "mixed",
        "arm": "a1",
        "experiment_id": 99,
        "arm_id": 3,
        "early_stop_stagnant_gens": 12,
        "operator_db_weights": True,
        "lm_budget": 0,
        "line_lm_budget": 0,
        "immigrants": 20,
    }
    cmd, _env = _build_evolution_cmd(cfg, use_docker=False)
    assert "scripts/run_verse_qd.py" in cmd or "run_verse_qd.py" in " ".join(cmd)
    assert "--experiment-id" in cmd
    assert str(99) in cmd
    assert "--arm-id" in cmd
    assert "--early-stop-stagnant-gens" in cmd
    assert "12" in cmd
    assert "--operator-db-weights" in cmd


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
