"""Tests for scripts/analyze_control_impact and evo_rhyme.experiment_analysis."""

from __future__ import annotations

import pytest
from evo_rhyme.experiment_analysis import analyze_control_impact
from scripts.analyze_control_impact import report_to_markdown


def test_analyze_control_impact_empty_rows():
    report = analyze_control_impact([])
    assert report["runs"] == 0
    assert "by_control" in report
    assert "correlations" in report


def test_analyze_control_impact_synthetic_two_arms():
    rows = [
        {"run_id": 1, "controls": {"elites": 3, "population": 100}, "fitness": 0.5, "fitness_vector": {"rhyme": 0.5, "flow": 0.5, "semantic": 0.5, "novelty": 0.5, "punchline": 0.5}},
        {"run_id": 2, "controls": {"elites": 3, "population": 100}, "fitness": 0.52, "fitness_vector": {"rhyme": 0.52, "flow": 0.5, "semantic": 0.5, "novelty": 0.5, "punchline": 0.5}},
        {"run_id": 3, "controls": {"elites": 10, "population": 100}, "fitness": 0.7, "fitness_vector": {"rhyme": 0.7, "flow": 0.7, "semantic": 0.6, "novelty": 0.6, "punchline": 0.6}},
        {"run_id": 4, "controls": {"elites": 10, "population": 100}, "fitness": 0.72, "fitness_vector": {"rhyme": 0.72, "flow": 0.7, "semantic": 0.6, "novelty": 0.6, "punchline": 0.6}},
    ]
    report = analyze_control_impact(rows, bootstrap_n=50)
    assert report["runs"] == 4
    assert "elites" in report["by_control"]
    elites_data = report["by_control"]["elites"]
    assert "values" in elites_data
    assert "n_per_value" in elites_data
    assert "mean_differences" in elites_data
    assert "cohens_d" in elites_data
    assert "ci_by_value" in elites_data
    assert set(elites_data["values"]) == {"3", "10"}
    assert elites_data["n_per_value"]["3"] == 2
    assert elites_data["n_per_value"]["10"] == 2


def test_analyze_control_impact_correlation_numeric():
    rows = [
        {"run_id": i, "controls": {"population": 50 + i * 10}, "fitness": 0.4 + i * 0.05, "fitness_vector": {"rhyme": 0.5, "flow": 0.5, "semantic": 0.5, "novelty": 0.5, "punchline": 0.5}}
        for i in range(5)
    ]
    report = analyze_control_impact(rows, bootstrap_n=30)
    assert "correlations" in report
    assert "population" in report["correlations"]
    assert "fitness" in report["correlations"]["population"]
    r = report["correlations"]["population"]["fitness"]
    assert -1 <= r <= 1


def test_analyze_control_impact_policy_version_performance():
    rows = [
        {"run_id": 1, "controls": {"policy_version": "v1"}, "fitness": 0.5, "fitness_vector": {"rhyme": 0.5, "flow": 0.4, "semantic": 0.4, "novelty": 0.4, "punchline": 0.4}},
        {"run_id": 2, "controls": {"policy_version": "v1"}, "fitness": 0.6, "fitness_vector": {"rhyme": 0.6, "flow": 0.5, "semantic": 0.5, "novelty": 0.5, "punchline": 0.5}},
        {"run_id": 3, "controls": {"policy_version": "v2"}, "fitness": 0.7, "fitness_vector": {"rhyme": 0.7, "flow": 0.6, "semantic": 0.6, "novelty": 0.6, "punchline": 0.6}},
    ]
    report = analyze_control_impact(rows, bootstrap_n=20)
    perf = report.get("policy_performance") or {}
    assert "v1" in perf
    assert "v2" in perf
    assert perf["v1"]["n_runs"] == 2
    assert perf["v2"]["n_runs"] == 1


def test_report_to_markdown_includes_policy_progression_table():
    report = {
        "runs": 3,
        "policy_performance": {
            "20260319_1015": {
                "n_runs": 2,
                "avg_fitness": 0.612,
                "fitness_vector_mean": {
                    "rhyme": 0.71,
                    "flow": 0.58,
                    "semantic": 0.54,
                    "novelty": 0.49,
                    "punchline": 0.43,
                },
            },
            "20260319_1430": {
                "n_runs": 1,
                "avg_fitness": 0.644,
                "fitness_vector_mean": {
                    "rhyme": 0.73,
                    "flow": 0.61,
                    "semantic": 0.57,
                    "novelty": 0.52,
                    "punchline": 0.47,
                },
            },
        },
        "by_control": {},
        "correlations": {},
    }
    md = report_to_markdown(report)
    assert "## Policy Progression" in md
    assert "| policy_version | runs | avg_fitness |" in md
    assert "20260319_1015" in md
    assert "20260319_1430" in md
    assert "Best policy so far" in md
    assert "low-confidence" in md
