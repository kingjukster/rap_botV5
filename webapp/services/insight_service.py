"""Run insight detection: auto-detect patterns and return actionable insight cards."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STAGNATION_THRESHOLD = 3
LOW_ACCEPTANCE_THRESHOLD = 0.05
DIVERSITY_DROP_RATIO = 0.5


def get_run_insights(
    generations: List[Dict[str, Any]],
    operator_events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze generation data and return actionable insight cards.

    Each card: {"level": "warning"|"info"|"success", "title": str, "detail": str}
    """
    insights: List[Dict[str, Any]] = []
    if not generations:
        return insights

    insights.extend(_detect_stagnation(generations))
    insights.extend(_detect_fitness_decline(generations))
    insights.extend(_detect_low_acceptance(generations))
    insights.extend(_detect_diversity_collapse(generations))
    insights.extend(_detect_early_convergence(generations))
    if operator_events:
        insights.extend(_detect_operator_issues(operator_events))

    if not insights:
        insights.append({
            "level": "success",
            "title": "Run looks healthy",
            "detail": "No issues detected in the generation data.",
        })

    return insights


def _detect_stagnation(gens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_values = [g.get("best_fitness") for g in gens if g.get("best_fitness") is not None]
    if len(best_values) < 2:
        return []

    peak = max(best_values)
    peak_idx = best_values.index(peak)
    tail_length = len(best_values) - 1 - peak_idx

    if tail_length >= STAGNATION_THRESHOLD:
        improved = any(v > peak for v in best_values[peak_idx + 1:])
        if not improved:
            return [{
                "level": "warning",
                "title": f"Stagnation detected ({tail_length} gens)",
                "detail": (
                    f"Best fitness peaked at {peak:.4f} in generation {gens[peak_idx].get('gen', peak_idx)} "
                    f"and has not improved for {tail_length} generations."
                ),
            }]
    return []


def _detect_fitness_decline(gens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_values = [g.get("best_fitness") for g in gens if g.get("best_fitness") is not None]
    if len(best_values) < 3:
        return []

    peak = max(best_values)
    last = best_values[-1]
    if last < peak * 0.95 and peak > 0:
        drop_pct = (1 - last / peak) * 100
        return [{
            "level": "warning",
            "title": f"Fitness dropped {drop_pct:.1f}% from peak",
            "detail": f"Peak was {peak:.4f}, current best is {last:.4f}.",
        }]
    return []


def _detect_low_acceptance(gens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recent = gens[-3:] if len(gens) >= 3 else gens
    rates = [g.get("acceptance_rate") for g in recent if g.get("acceptance_rate") is not None]
    if not rates:
        return []

    avg_rate = sum(rates) / len(rates)
    if avg_rate < LOW_ACCEPTANCE_THRESHOLD:
        return [{
            "level": "warning",
            "title": f"Low acceptance rate ({avg_rate:.1%})",
            "detail": "Very few new candidates are being accepted. The population may be converged or mutations may be too disruptive.",
        }]
    return []


def _detect_diversity_collapse(gens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    divs = [(i, g.get("diversity")) for i, g in enumerate(gens) if g.get("diversity") is not None]
    if len(divs) < 3:
        return []

    peak_div = max(d for _, d in divs)
    last_div = divs[-1][1]
    if peak_div > 0 and last_div < peak_div * DIVERSITY_DROP_RATIO:
        return [{
            "level": "warning",
            "title": "Diversity collapse",
            "detail": f"Diversity dropped from {peak_div:.2f} to {last_div:.2f} ({(1 - last_div / peak_div) * 100:.0f}% reduction).",
        }]
    return []


def _detect_early_convergence(gens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_values = [g.get("best_fitness") for g in gens if g.get("best_fitness") is not None]
    if len(best_values) < 4:
        return []

    peak = max(best_values)
    peak_idx = best_values.index(peak)
    midpoint = len(best_values) // 2

    if peak_idx < midpoint and peak_idx < len(best_values) - 2:
        return [{
            "level": "info",
            "title": "Early convergence",
            "detail": (
                f"Best fitness reached {peak:.4f} at generation {gens[peak_idx].get('gen', peak_idx)}, "
                f"before the halfway point (gen {gens[midpoint].get('gen', midpoint)}). "
                "Later generations may benefit from increased mutation strength."
            ),
        }]
    return []


def _detect_operator_issues(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    op_counts: Dict[str, int] = {}
    for e in events:
        op = e.get("operator") or "unknown"
        op_counts[op] = op_counts.get(op, 0) + 1

    if len(op_counts) < 2:
        return []

    total = sum(op_counts.values())
    insights = []
    for op, count in op_counts.items():
        ratio = count / total
        if ratio > 0.7:
            insights.append({
                "level": "info",
                "title": f"Operator '{op}' dominates ({ratio:.0%})",
                "detail": f"The operator '{op}' accounts for {count}/{total} events. Consider diversifying operator selection.",
            })
    return insights


def get_global_insights(analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate global-level insights from the analysis data."""
    insights: List[Dict[str, Any]] = []

    stagnation = analysis_data.get("stagnation_runs", 0)
    if stagnation >= 5:
        insights.append({
            "level": "warning",
            "title": f"Global stagnation ({stagnation} runs)",
            "detail": f"The last {stagnation} runs have not exceeded the all-time best fitness.",
        })

    runs = analysis_data.get("runs", {})
    failed = runs.get("failed", 0)
    total = runs.get("total", 0)
    if total > 0 and failed / total > 0.3:
        insights.append({
            "level": "warning",
            "title": f"High failure rate ({failed}/{total})",
            "detail": f"{failed / total:.0%} of runs have failed. Check logs for recurring errors.",
        })

    op_mix = analysis_data.get("operator_mix", [])
    if len(op_mix) >= 2:
        total_ops = sum(o.get("count", 0) for o in op_mix)
        if total_ops > 0:
            top = op_mix[0]
            ratio = top.get("count", 0) / total_ops
            if ratio > 0.6:
                insights.append({
                    "level": "info",
                    "title": f"Operator imbalance: '{top.get('operator', '?')}'",
                    "detail": f"One operator accounts for {ratio:.0%} of all events across runs.",
                })

    if not insights:
        best = analysis_data.get("best_fitness")
        if best is not None:
            insights.append({
                "level": "success",
                "title": f"System healthy (best: {best:.4f})",
                "detail": "No global issues detected.",
            })

    return insights
