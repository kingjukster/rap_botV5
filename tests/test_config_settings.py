"""Tests for config.settings: get_evolution_defaults, get_qd_defaults, key aliases, path mapping, precedence."""

from pathlib import Path

import pytest

# Ensure we can import config.settings (may need project root on path)
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_get_evolution_defaults_returns_dict_with_canonical_keys():
    from config.settings import get_evolution_defaults

    d = get_evolution_defaults()
    assert isinstance(d, dict)
    assert "population" in d
    assert "elites" in d
    assert "immigrants" in d
    assert "init" in d
    assert "generations" in d
    assert isinstance(d["population"], (int, float)) or d["population"] is None
    assert d["init"] in ("mixed", "random", "template", None)


def test_get_qd_defaults_returns_dict_with_canonical_keys():
    from config.settings import get_qd_defaults

    d = get_qd_defaults()
    assert isinstance(d, dict)
    assert "population" in d
    assert "scheme" in d
    assert "elites" in d
    assert "init" in d
    assert "generations" in d
    assert d["scheme"] in ("AABB", "ABAB", "ABBA", "AAAA", "ABCB", "AABA", None)


def test_evolution_section_alias_population_size_to_population(tmp_path):
    """YAML key population_size is mapped to canonical key population."""
    cfg = tmp_path / "ev.yaml"
    cfg.write_text(
        "evolution:\n"
        "  population_size: 99\n"
        "  num_elites: 7\n"
        "  random_immigrants_per_gen: 12\n"
        "  population_init: random\n"
    )
    from config.settings import get_evolution_defaults

    d = get_evolution_defaults(config_path=str(cfg))
    assert d["population"] == 99
    assert d["elites"] == 7
    assert d["immigrants"] == 12
    assert d["init"] == "random"


def test_path_alias_elite_corpus_maps_to_elite_corpus_path(tmp_path):
    """paths.elite_corpus in YAML is used for elite_corpus_path."""
    cfg = tmp_path / "paths.yaml"
    cfg.write_text(
        "paths:\n"
        "  elite_corpus: data/my_custom_corpus.txt\n"
    )
    from config.settings import load_settings

    settings = load_settings(config_path=str(cfg))
    assert "my_custom_corpus" in str(settings.elite_corpus_path)


def test_config_file_overrides_python_defaults(tmp_path):
    """Config file evolution section overrides EVOLUTION_SECTION_DEFAULTS."""
    cfg = tmp_path / "overrides.yaml"
    cfg.write_text(
        "evolution:\n"
        "  population: 42\n"
        "  generations: 99\n"
        "  init: template\n"
    )
    from config.settings import get_evolution_defaults

    d = get_evolution_defaults(config_path=str(cfg))
    assert d["population"] == 42
    assert d["generations"] == 99
    assert d["init"] == "template"


def test_qd_section_alias_num_elites_to_elites(tmp_path):
    """QD section YAML key num_elites is mapped to elites."""
    cfg = tmp_path / "qd.yaml"
    cfg.write_text(
        "qd:\n"
        "  population_size: 80\n"
        "  num_elites: 10\n"
        "  random_immigrants_per_gen: 25\n"
        "  population_init: mixed\n"
    )
    from config.settings import get_qd_defaults

    d = get_qd_defaults(config_path=str(cfg))
    assert d["population"] == 80
    assert d["elites"] == 10
    assert d["immigrants"] == 25
    assert d["init"] == "mixed"


def test_evolution_default_config_path_prefers_evolution_yaml():
    """When no RAPBOT_CONFIG, loader prefers config/evolution.yaml if it exists."""
    from config.settings import _repo_root

    base = _repo_root()
    evolution_yaml = base / "config" / "evolution.yaml"
    if not evolution_yaml.exists():
        pytest.skip("config/evolution.yaml not present")
    from config.settings import load_settings
    import os
    orig = os.environ.pop("RAPBOT_CONFIG", None)
    try:
        settings = load_settings()
        # Our evolution.yaml has evolution.population: 100, generations: 30, etc.
        assert "population" in settings.evolution
        assert settings.evolution.get("generations") == 30
    finally:
        if orig is not None:
            os.environ["RAPBOT_CONFIG"] = orig
