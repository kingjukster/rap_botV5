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
        "target": target,
        "feature_names": feature_names,
        "feature_importance": importance,
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
        "n_samples": len(y),
    }
