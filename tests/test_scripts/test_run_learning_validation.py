"""Tests for scripts/run_learning_validation.py harness helpers."""

from __future__ import annotations

from pathlib import Path
import pytest

from scripts.run_learning_validation import (
    classify_delta,
    compute_policy_mode_verdict,
    phase_specs,
    run_analysis_for_run_ids,
)


def test_classify_delta_thresholds():
    assert classify_delta(0.02) == "improved"
    assert classify_delta(-0.02) == "regressed"
    assert classify_delta(0.005) == "neutral"
    assert classify_delta(-0.005) == "neutral"


def test_compute_policy_mode_verdict_low_confidence():
    report = {
        "by_control": {
            "policy_mode": {
                "mean_outcome_by_value": {
                    "static": {"fitness": 0.60},
                    "learned": {"fitness": 0.63},
                },
                "n_per_value": {"static": 3, "learned": 4},
            }
        }
    }
    verdict = compute_policy_mode_verdict(report, baseline="static", candidate="learned")
    assert verdict["policy_improvement"] == pytest.approx(0.03)
    assert verdict["verdict"] == "improved"
    assert verdict["low_confidence"] is True


def test_phase_specs_contains_required_phases():
    phases = phase_specs("standard")
    names = [p[0] for p in phases]
    assert "sanity" in names
    assert "ab_static_vs_learned" in names
    assert "ab_learned_vs_exploremix" in names


def test_run_analysis_for_run_ids_dry_run_returns_command(tmp_path: Path):
    res = run_analysis_for_run_ids([1, 2, 3], tmp_path, "aggregate", dry_run=True)
    cmd = res["command"]
    assert cmd["exit_code"] == 0
    argv = cmd["argv"]
    joined = " ".join(argv)
    assert "analyze_control_impact.py" in joined
    assert "--run-ids" in argv
    assert "1,2,3" in argv
