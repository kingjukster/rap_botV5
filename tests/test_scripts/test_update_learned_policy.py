"""Tests for update_learned_policy.py scoring and CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "update_learned_policy",
        ROOT / "scripts" / "update_learned_policy.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_for_policy_completed():
    """Completed runs use raw fitness."""
    mod = _load_module()
    row = {"fitness": 0.8, "status": "completed"}
    assert mod._score_for_policy(row, 0.5) == 0.8


def test_score_for_policy_failed():
    """Failed runs get penalty subtracted."""
    mod = _load_module()
    row = {"fitness": 0.8, "status": "failed"}
    assert mod._score_for_policy(row, 0.5) == pytest.approx(0.3)


def test_score_for_policy_failed_clamped():
    """Failed run score is clamped at 0."""
    mod = _load_module()
    row = {"fitness": 0.2, "status": "failed"}
    assert mod._score_for_policy(row, 0.5) == 0.0


def test_score_for_policy_failed_penalty_zero():
    """Zero penalty means failed == fitness."""
    mod = _load_module()
    row = {"fitness": 0.7, "status": "failed"}
    assert mod._score_for_policy(row, 0.0) == 0.7


def test_score_for_policy_selection_pressure_linear():
    """selection_pressure=1.0 gives linear (legacy) behavior."""
    mod = _load_module()
    row = {"fitness": 0.8, "status": "completed"}
    assert mod._score_for_policy(row, 0.5, selection_pressure=1.0) == 0.8


def test_score_for_policy_selection_pressure_squared():
    """selection_pressure=2.0 strengthens top configs (fitness^2)."""
    mod = _load_module()
    row = {"fitness": 0.9, "status": "completed"}
    assert mod._score_for_policy(row, 0.5, selection_pressure=2.0) == pytest.approx(0.81)
    row_low = {"fitness": 0.7, "status": "completed"}
    assert mod._score_for_policy(row_low, 0.5, selection_pressure=2.0) == pytest.approx(0.49)


def test_score_for_policy_squared_failed():
    """Squared scoring with failed status applies penalty to base."""
    mod = _load_module()
    row = {"fitness": 0.8, "status": "failed"}
    # base = 0.8^2 = 0.64, score = max(0, 0.64 - 0.5) = 0.14
    assert mod._score_for_policy(row, 0.5, selection_pressure=2.0) == pytest.approx(0.14)


def test_default_failure_penalty():
    """DEFAULT_FAILURE_PENALTY is 0.8."""
    mod = _load_module()
    assert mod.DEFAULT_FAILURE_PENALTY == 0.8


def test_cli_help():
    """Script accepts --failure-penalty, --selection-pressure, and --min-runs."""
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "update_learned_policy.py"), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0
    assert "--failure-penalty" in rc.stdout
    assert "--selection-pressure" in rc.stdout
    assert "--min-runs" in rc.stdout
