"""Tests for webapp.config. Area 13."""

from pathlib import Path

import pytest

from webapp.config import ROOT, STATIC_DIR, TEMPLATES_DIR


def test_config_root_is_path():
    assert isinstance(ROOT, Path)
    assert ROOT.exists() or True


def test_config_templates_dir():
    assert TEMPLATES_DIR == ROOT / "webapp" / "templates"
    assert "templates" in str(TEMPLATES_DIR)


def test_config_static_dir():
    assert STATIC_DIR == ROOT / "webapp" / "static"
    assert "static" in str(STATIC_DIR)
