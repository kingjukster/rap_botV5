"""Tests for webapp.services.insight_service: run-level and global insight detection."""

import pytest

from webapp.services.insight_service import get_run_insights, get_global_insights


def _gens(best_values, avg_values=None, diversity=None, acceptance=None):
    """Helper: build generation dicts from best fitness list."""
    gens = []
    for i, best in enumerate(best_values):
        g = {"gen": i, "best_fitness": best}
        if avg_values and i < len(avg_values):
            g["avg_fitness"] = avg_values[i]
        if diversity and i < len(diversity):
            g["diversity"] = diversity[i]
        if acceptance and i < len(acceptance):
            g["acceptance_rate"] = acceptance[i]
        gens.append(g)
    return gens


def test_empty_generations_returns_empty():
    assert get_run_insights([]) == []


def test_healthy_run_returns_success():
    gens = _gens([0.5, 0.6, 0.7, 0.8])
    insights = get_run_insights(gens)
    assert len(insights) == 1
    assert insights[0]["level"] == "success"


def test_stagnation_detected():
    gens = _gens([0.5, 0.6, 0.7, 0.7, 0.7, 0.7])
    insights = get_run_insights(gens)
    levels = [i["level"] for i in insights]
    assert "warning" in levels
    titles = " ".join(i["title"] for i in insights)
    assert "Stagnation" in titles or "stagnation" in titles.lower()


def test_fitness_decline_detected():
    gens = _gens([0.5, 0.8, 0.7, 0.6])
    insights = get_run_insights(gens)
    titles = " ".join(i["title"] for i in insights)
    assert "dropped" in titles.lower() or "decline" in titles.lower()


def test_low_acceptance_detected():
    gens = _gens([0.5, 0.6, 0.7], acceptance=[0.01, 0.02, 0.01])
    insights = get_run_insights(gens)
    titles = " ".join(i["title"] for i in insights)
    assert "acceptance" in titles.lower()


def test_diversity_collapse_detected():
    gens = _gens([0.5, 0.6, 0.7, 0.8], diversity=[1.0, 0.9, 0.5, 0.3])
    insights = get_run_insights(gens)
    titles = " ".join(i["title"] for i in insights)
    assert "diversity" in titles.lower()


def test_early_convergence_detected():
    gens = _gens([0.3, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])
    insights = get_run_insights(gens)
    titles = " ".join(i["title"] for i in insights)
    assert "early" in titles.lower() or "convergence" in titles.lower()


def test_operator_dominance_detected():
    events = [{"operator": "swap_words"}] * 80 + [{"operator": "rephrase"}] * 10
    gens = _gens([0.5, 0.6, 0.7, 0.8])
    insights = get_run_insights(gens, operator_events=events)
    titles = " ".join(i["title"] for i in insights)
    assert "swap_words" in titles


def test_global_insights_high_stagnation():
    data = {"stagnation_runs": 10, "runs": {"total": 20, "failed": 2}, "operator_mix": [], "best_fitness": 0.8}
    insights = get_global_insights(data)
    titles = " ".join(i["title"] for i in insights)
    assert "stagnation" in titles.lower()


def test_global_insights_high_failure_rate():
    data = {"stagnation_runs": 0, "runs": {"total": 10, "failed": 5}, "operator_mix": [], "best_fitness": 0.5}
    insights = get_global_insights(data)
    titles = " ".join(i["title"] for i in insights)
    assert "failure" in titles.lower()


def test_global_insights_healthy():
    data = {"stagnation_runs": 1, "runs": {"total": 20, "failed": 1}, "operator_mix": [], "best_fitness": 0.9}
    insights = get_global_insights(data)
    assert insights[0]["level"] == "success"
