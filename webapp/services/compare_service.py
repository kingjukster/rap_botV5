"""Compare runs service: fetch generation series and summaries for multi-run comparison."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from webapp.services.run_service import get_run_summary

logger = logging.getLogger(__name__)

MAX_COMPARE_RUNS = 10


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
        runs.append(summary)
        generation_series[str(rid)] = [
            {
                "gen": g.get("gen"),
                "best_fitness": g.get("best_fitness"),
                "avg_fitness": g.get("avg_fitness"),
                "diversity": g.get("diversity"),
            }
            for g in gens
        ]

    return {
        "runs": runs,
        "generation_series": generation_series,
    }
