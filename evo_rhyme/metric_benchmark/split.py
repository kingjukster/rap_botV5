"""Stratified 40/40/20 train/val/test split for labeled verses."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None  # type: ignore

logger = logging.getLogger(__name__)


def _tertile(x: float, q1: float, q2: float) -> str:
    if x <= q1:
        return "low"
    if x <= q2:
        return "mid"
    return "high"


def _stratify_key(row: Dict[str, Any], q1: float, q2: float) -> str:
    src = str(row.get("source", "corpus"))
    labels = row.get("labels")
    if isinstance(labels, dict) and "overall" in labels:
        t = _tertile(float(labels["overall"]), q1, q2)
    else:
        t = "unknown"
    return f"{src}|{t}"


def split_three_way(
    rows: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Labeled verses get split 40/40/20 with stratification on source|overall_tertile.
    Unlabeled rows are appended without split (caller writes to unassigned).
    """
    labeled = [r for r in rows if not r.get("unlabeled") and isinstance(r.get("labels"), dict)]
    if len(labeled) < 3:
        raise ValueError("Need at least 3 labeled verses to split")

    overall = [float(r["labels"]["overall"]) for r in labeled]  # type: ignore[index]
    q1 = float(np.quantile(overall, 1.0 / 3.0))
    q2 = float(np.quantile(overall, 2.0 / 3.0))

    strata = [_stratify_key(r, q1, q2) for r in labeled]
    counts = Counter(strata)
    small = [k for k, v in counts.items() if v < 2]
    if small:
        logger.warning("Strata with <2 samples (merging to 'other'): %s", small)
        strata = [s if s not in small else "other" for s in strata]

    indices = np.arange(len(labeled))
    s_arr = np.array(strata)

    manifest_extra: Dict[str, Any] = {
        "n_verses": len(labeled),
        "overall_q1": q1,
        "overall_q2": q2,
        "stratum_counts": dict(Counter(strata)),
    }

    if train_test_split is None:
        raise RuntimeError("scikit-learn required for stratified split")

    try:
        idx_tv, idx_test, _, _ = train_test_split(
            indices,
            s_arr,
            test_size=0.2,
            random_state=seed,
            stratify=s_arr,
        )
        s_tv = s_arr[idx_tv]
        rel = np.arange(len(idx_tv))
        rel_train, rel_val, _, _ = train_test_split(
            rel,
            s_tv,
            test_size=0.5,
            random_state=seed,
            stratify=s_tv,
        )
        idx_train = idx_tv[rel_train]
        idx_val = idx_tv[rel_val]
    except ValueError as e:
        logger.warning("Stratified split failed (%s); falling back to random", e)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(labeled))
        n_test = max(1, int(round(0.2 * len(labeled))))
        n_val = max(1, int(round(0.4 * len(labeled))))
        idx_test = perm[:n_test]
        idx_val = perm[n_test : n_test + n_val]
        idx_train = perm[n_test + n_val :]

    split_names = np.empty(len(labeled), dtype=object)
    split_names[idx_train] = "train"
    split_names[idx_val] = "val"
    split_names[idx_test] = "test"

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(labeled):
        rc = dict(r)
        rc["split"] = str(split_names[i])
        out.append(rc)

    unlabeled_out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("unlabeled") or not isinstance(r.get("labels"), dict):
            u = dict(r)
            u.pop("split", None)
            unlabeled_out.append(u)

    manifest_extra["splits"] = {
        "train": int(np.sum(split_names == "train")),
        "val": int(np.sum(split_names == "val")),
        "test": int(np.sum(split_names == "test")),
    }
    manifest_extra["unlabeled_omitted"] = len(unlabeled_out)
    return out + unlabeled_out, manifest_extra
