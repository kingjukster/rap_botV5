"""Compare runs service: fetch generation series and summaries for multi-run comparison."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from webapp.services.run_service import get_run_summary

logger = logging.getLogger(__name__)

MAX_COMPARE_RUNS = 10

_COMPARE_KEYS = (
    "arm",
    "population",
    "generations",
    "scheme",
    "init",
    "archive_mode",
    "emitter_strategy",
    "early_stop_stagnant_gens",
    "early_stop_reason",
    "early_stop_gen",
    "planned_generations",
)


def _mean_acceptance(gens: list) -> Optional[float]:
    vals = [
        float(g["acceptance_rate"])
        for g in gens
        if g.get("acceptance_rate") is not None
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _config_highlights(config_json: Any) -> Dict[str, Any]:
    if not isinstance(config_json, dict):
        return {}
    return {k: config_json.get(k) for k in _COMPARE_KEYS if k in config_json and config_json.get(k) is not None}


def compare_runs(run_ids: List[int]) -> Dict[str, Any]:
    """
    Fetch generation series and summary for each run_id.

    Returns {
        "runs": [ {run_id, script_name, theme_keywords, status, ...} ],
        "generation_series": { "<run_id>": [{gen, best_fitness, avg_fitness}, ...] },
    }
    """
    ids = run_ids[:MAX_COMPARE_RUNS]
    runs = []
    generation_series: Dict[str, list] = {}

    for rid in ids:
        summary = get_run_summary(rid, include_generations=True)
        if not summary:
            continue
        gens = summary.pop("_generations", [])
        summary["mean_acceptance_rate"] = _mean_acceptance(gens)
        cfgj = summary.get("config_json")
        summary["config_compare"] = _config_highlights(cfgj)
        runs.append(summary)
        generation_series[str(rid)] = [
            {
                "gen": g.get("gen"),
                "best_fitness": g.get("best_fitness"),
                "avg_fitness": g.get("avg_fitness"),
                "diversity": g.get("diversity"),
                "acceptance_rate": g.get("acceptance_rate"),
            }
            for g in gens
        ]

    return {
        "runs": runs,
        "generation_series": generation_series,
    }
