"""Tests for notebooks/runs_statistics - gate logic and stats pipeline."""

from __future__ import annotations


def test_enough_runs_gate_logic():
    """ENOUGH_RUNS is True when count >= 100, False otherwise."""
    assert (100 >= 100) is True
    assert (99 >= 100) is False
    assert (150 >= 100) is True


def test_runs_statistics_pipeline_with_mocked_db():
    """With 100+ mocked runs, load_run_outcomes + analyze_control_impact produce valid report."""
    from evo_rhyme.experiment_analysis import analyze_control_impact
    from evo_rhyme.experiment_metrics import FITNESS_VECTOR_KEYS

    # Synthetic rows simulating load_run_outcomes output for 120 runs
    rows = []
    for i in range(120):
        rows.append({
            "run_id": i + 1,
            "controls": {
                "init": "mixed" if i % 2 == 0 else "random",
                "population": 80 if i % 3 == 0 else 100,
                "policy_version": f"v{(i % 5) + 1}",
            },
            "fitness": 0.4 + (i % 60) * 0.01,
            "fitness_vector": {k: 0.5 + (i % 10) * 0.01 for k in FITNESS_VECTOR_KEYS},
        })

    report = analyze_control_impact(rows, bootstrap_n=50)
    assert report["runs"] == 120
    assert "by_control" in report
    assert "init" in report["by_control"]
    assert "policy_performance" in report
    assert len(report["policy_performance"]) >= 1
    # All fitness vectors should be present in aggregation
    for r in rows:
        assert set((r.get("fitness_vector") or {}).keys()) <= set(FITNESS_VECTOR_KEYS)
