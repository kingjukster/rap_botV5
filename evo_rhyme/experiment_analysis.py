"""
Control impact analysis: mean differences, bootstrap CI, effect size, correlations.

Used by scripts/analyze_control_impact.py and webapp run_service.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple


def _bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) for bootstrap CI."""
    if not values:
        return (0.0, 0.0, 0.0)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = (1 - ci) / 2
    hi = 1 - lo
    return (
        sum(values) / n,
        means[int(lo * n_bootstrap)],
        means[int(hi * n_bootstrap)],
    )


def _cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d for two samples."""
    if not a or not b:
        return 0.0
    m1, m2 = sum(a) / len(a), sum(b) / len(b)
    v1 = sum((x - m1) ** 2 for x in a) / max(1, len(a) - 1)
    v2 = sum((x - m2) ** 2 for x in b) / max(1, len(b) - 1)
    pooled_std = math.sqrt((v1 + v2) / 2)
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def _correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / (n - 1)) if n > 1 else 0.0
    sy = math.sqrt(sum((b - my) ** 2 for b in y) / (n - 1)) if n > 1 else 0.0
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1) / (sx * sy)


def analyze_control_impact(
    rows: List[Dict[str, Any]],
    control_keys: Optional[List[str]] = None,
    bootstrap_n: int = 1000,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Compute per-control effects and correlations. Used by CLI and webapp."""
    if not rows:
        return {"runs": 0, "controls": {}, "by_control": {}, "correlations": {}}

    outcome_keys = ["fitness"] + list((rows[0].get("fitness_vector") or {}).keys())
    if control_keys is None:
        control_keys = set()
        for r in rows:
            control_keys.update(k for k in (r.get("controls") or {}).keys() if (r.get("controls") or {}).get(k) is not None)
        control_keys = sorted(control_keys)

    report = {
        "runs": len(rows),
        "outcome_keys": outcome_keys,
        "by_control": {},
        "correlations": {},
        "policy_performance": {},
    }

    for ck in control_keys:
        values_by_key: Dict[Any, List[Dict[str, Any]]] = {}
        for r in rows:
            val = (r.get("controls") or {}).get(ck)
            if val is None:
                continue
            key = val if isinstance(val, (int, float, str, bool)) else json.dumps(val, sort_keys=True)
            values_by_key.setdefault(key, []).append(r)

        if len(values_by_key) < 2:
            report["by_control"][ck] = {"values": list(values_by_key.keys()), "n_per_value": {str(k): len(v) for k, v in values_by_key.items()}}
            continue

        means_by_val: Dict[str, Dict[str, float]] = {}
        for val_key, run_list in values_by_key.items():
            fitnesses = [x["fitness"] for x in run_list if x.get("fitness") is not None]
            vec_means = {}
            for ok in outcome_keys:
                if ok == "fitness":
                    vec_means[ok] = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
                else:
                    vec_means[ok] = sum((x.get("fitness_vector") or {}).get(ok, 0.0) for x in run_list) / len(run_list)
            means_by_val[str(val_key)] = vec_means

        val_list = list(values_by_key.keys())
        run_lists = [values_by_key[v] for v in val_list]
        fitness_lists = [[x["fitness"] for x in rl if x.get("fitness") is not None] for rl in run_lists]
        deltas = {}
        cohens = {}
        for i in range(len(val_list)):
            for j in range(i + 1, len(val_list)):
                a, b = fitness_lists[i], fitness_lists[j]
                if a and b:
                    key = f"{val_list[i]} vs {val_list[j]}"
                    deltas[key] = (sum(a) / len(a)) - (sum(b) / len(b))
                    cohens[key] = _cohens_d(a, b)

        ci_by_val = {}
        for val_key, run_list in values_by_key.items():
            fitnesses = [x["fitness"] for x in run_list if x.get("fitness") is not None]
            if fitnesses:
                mean, lo, hi = _bootstrap_ci(fitnesses, n_bootstrap=bootstrap_n, ci=ci)
                ci_by_val[str(val_key)] = {"mean": mean, "ci_low": lo, "ci_high": hi}

        report["by_control"][ck] = {
            "values": [str(v) for v in values_by_key.keys()],
            "n_per_value": {str(k): len(v) for k, v in values_by_key.items()},
            "mean_outcome_by_value": means_by_val,
            "mean_differences": deltas,
            "cohens_d": cohens,
            "ci_by_value": ci_by_val,
        }

    for ck in control_keys:
        numeric_vals = []
        fitness_vals = []
        for r in rows:
            val = (r.get("controls") or {}).get(ck)
            if val is None or not isinstance(val, (int, float)):
                continue
            numeric_vals.append(float(val))
            fitness_vals.append(r.get("fitness") or 0.0)
        if len(numeric_vals) >= 3:
            report["correlations"][ck] = {"fitness": _correlation(numeric_vals, fitness_vals)}
            vec = rows[0].get("fitness_vector") or {}
            for ok in vec:
                x, y = [], []
                for r in rows:
                    val = (r.get("controls") or {}).get(ck)
                    if val is None or not isinstance(val, (int, float)):
                        continue
                    x.append(float(val))
                    y.append((r.get("fitness_vector") or {}).get(ok, 0.0))
                if len(x) >= 3 and len(x) == len(y):
                    report["correlations"][ck][ok] = _correlation(x, y)

    # Track policy version performance over time
    by_policy: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cfg = r.get("controls") or {}
        pv = cfg.get("policy_version") or (cfg.get("control_snapshot") or {}).get("policy_version")
        if pv is None:
            continue
        by_policy.setdefault(str(pv), []).append(r)
    if by_policy:
        perf: Dict[str, Dict[str, Any]] = {}
        for pv, items in by_policy.items():
            fitness_vals = [float(i.get("fitness") or 0.0) for i in items]
            vec_keys = list((items[0].get("fitness_vector") or {}).keys()) if items else []
            vec_means = {}
            for key in vec_keys:
                vec_means[key] = sum((i.get("fitness_vector") or {}).get(key, 0.0) for i in items) / max(1, len(items))
            perf[pv] = {
                "n_runs": len(items),
                "avg_fitness": sum(fitness_vals) / max(1, len(fitness_vals)),
                "fitness_vector_mean": vec_means,
            }
        report["policy_performance"] = perf

    return report
