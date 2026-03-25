"""
Predictive model: controls/features -> fitness (or fitness_vector).

Trains a small sklearn model for control attribution and optional SHAP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _flatten_controls(controls: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten controls to numeric/categorical for model input. Nested dicts are JSON-stringified key."""
    out = {}
    for k, v in controls.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float)):
            out[k] = float(v) if isinstance(v, (int, float)) else int(v)
        elif isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = hash(v) % (2 ** 20)  # categorical as hash
        elif isinstance(v, dict):
            out[f"{k}_hash"] = hash(json.dumps(v, sort_keys=True)) % (2 ** 20)
        else:
            out[f"{k}_hash"] = hash(str(v)) % (2 ** 20)
    return out


def build_control_matrix(
    rows: List[Dict[str, Any]],
    control_keys: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Build a list of flat feature dicts and the list of feature names.
    rows: list of { controls, fitness, fitness_vector }.
    """
    if control_keys is None:
        control_keys = set()
        for r in rows:
            control_keys.update((r.get("controls") or {}).keys())
        control_keys = sorted(control_keys)

    feature_names = []
    seen = set()
    for r in rows:
        flat = _flatten_controls(r.get("controls") or {})
        for k in flat:
            if k not in seen:
                seen.add(k)
                feature_names.append(k)
    feature_names.sort()

    X = []
    for r in rows:
        flat = _flatten_controls(r.get("controls") or {})
        vec = {f: flat.get(f, 0.0) for f in feature_names}
        X.append(vec)
    return X, feature_names


def train_predictive_model(
    rows: List[Dict[str, Any]],
    target: str = "fitness",
    control_keys: Optional[List[str]] = None,
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """
    Train a model to predict target from controls.
    target: "fitness" or a key in fitness_vector (e.g. "rhyme", "flow").
    Returns dict with model artifact, feature_importance, and optional SHAP info.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.warning("sklearn not installed; skipping control model")
        return {"error": "sklearn required"}

    X_list, feature_names = build_control_matrix(rows, control_keys=control_keys)
    if not X_list or not feature_names:
        return {"error": "no features"}

    # Convert to 2D array (same order as feature_names)
    import numpy as np
    X = np.array([[d[f] for f in feature_names] for d in X_list])
    y = []
    for r in rows:
        if target == "fitness":
            y.append(r.get("fitness") or 0.0)
        else:
            y.append((r.get("fitness_vector") or {}).get(target, 0.0))
    y = np.array(y)

    if len(y) < 5:
        return {"error": "too few samples", "n": len(y)}

    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X, y)
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    scores = cross_val_score(model, X, y, cv=min(5, len(y) - 1), scoring="r2")
    return {
        "model": model,
        "target": target,
        "feature_names": feature_names,
        "feature_importance": importance,
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
        "n_samples": len(y),
    }


def propose_config_from_model(
    rows: List[Dict[str, Any]],
    n_proposals: int = 20,
    top_k: int = 3,
    noise_scale: float = 0.15,
) -> List[Dict[str, Any]]:
    """Train RF on historical runs and propose new configs by perturbing the best.

    Args:
        rows: List of {controls: dict, fitness: float} from completed runs.
        n_proposals: Number of random perturbations to generate.
        top_k: Return the top-k proposals ranked by predicted fitness.
        noise_scale: Relative perturbation magnitude for numeric controls.

    Returns:
        List of {controls: dict, predicted_fitness: float}, best first.
    """
    result = train_predictive_model(rows, target="fitness")
    model = result.get("model")
    if model is None or result.get("error"):
        logger.warning("Control model training failed: %s", result.get("error"))
        return []

    feature_names = result["feature_names"]
    cv_r2 = result.get("cv_r2_mean", 0.0)
    if cv_r2 < 0.05:
        logger.info("Control model R^2=%.3f too low, skipping proposals", cv_r2)
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    X_list, _ = build_control_matrix(rows, control_keys=None)
    X = np.array([[d.get(f, 0.0) for f in feature_names] for d in X_list])
    y = np.array([r.get("fitness", 0.0) for r in rows])
    best_idx = int(np.argmax(y))
    best_x = X[best_idx]

    proposals = []
    rng = np.random.RandomState(42)
    for _ in range(n_proposals):
        candidate = best_x.copy()
        for j in range(len(candidate)):
            if abs(candidate[j]) < 1e-9:
                candidate[j] += rng.normal(0, 0.1)
            else:
                candidate[j] *= 1.0 + rng.normal(0, noise_scale)
        pred = float(model.predict(candidate.reshape(1, -1))[0])
        proposals.append((candidate, pred))

    proposals.sort(key=lambda x: x[1], reverse=True)

    best_controls = rows[best_idx].get("controls", {})
    numeric_keys = [
        k for k in best_controls
        if isinstance(best_controls.get(k), (int, float, bool))
    ]

    out = []
    for candidate_vec, pred in proposals[:top_k]:
        ctrl = dict(best_controls)
        for i, fname in enumerate(feature_names):
            if fname in ctrl and isinstance(ctrl[fname], (int, float)):
                if isinstance(ctrl[fname], int):
                    ctrl[fname] = max(1, int(round(candidate_vec[i])))
                else:
                    ctrl[fname] = round(float(candidate_vec[i]), 4)
        out.append({
            "controls": ctrl,
            "predicted_fitness": pred,
            "cv_r2": cv_r2,
        })

    logger.info(
        "Control model proposed %d configs (R^2=%.3f, best predicted=%.4f)",
        len(out), cv_r2, out[0]["predicted_fitness"] if out else 0.0,
    )
    return out
