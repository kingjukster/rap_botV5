"""Run Spearman/MSE and pairwise metrics vs human labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from evo_rhyme.experiment_metrics import fitness_vector_from_scores
from evo_rhyme.fitness import score_verse
from evo_rhyme.individual import VerseIndividual, analyze_verse_individual
from evo_rhyme.metric_benchmark.pairwise_eval import summarize_pairwise
from evo_rhyme.metric_benchmark.protocol import default_protocol_manifest, load_protocol_manifest
from evo_rhyme.metric_benchmark.scoring import (
    axis_scalar_for_pairwise,
    fluency_composite_from_scores,
)
from evo_rhyme.metric_benchmark.schema import LABEL_KEYS, parse_pairwise_line

logger = logging.getLogger(__name__)


def _score_one(lines: List[str], fast: bool) -> Dict[str, float]:
    ind = VerseIndividual(lines=list(lines))
    analyze_verse_individual(ind)
    return score_verse(ind, include_graph_metrics=not fast)


def _correlation_axis(
    y_true: List[float],
    y_pred: List[float],
) -> Tuple[Optional[float], Optional[float]]:
    if len(y_true) < 2:
        return None, None
    r, p = spearmanr(y_true, y_pred)
    return (float(r) if not np.isnan(r) else None, float(p) if p is not None and not np.isnan(p) else None)


def _mse(y_true: List[float], y_pred: List[float]) -> float:
    a = np.array(y_true, dtype=float)
    b = np.array(y_pred, dtype=float)
    return float(np.mean((a - b) ** 2))


def run_evaluation(
    verses_path: Path,
    pairs_path: Optional[Path],
    *,
    fast: bool,
    epsilon: float,
    protocol_path: Optional[Path],
) -> Dict[str, Any]:
    manifest = (
        load_protocol_manifest(protocol_path)
        if protocol_path and protocol_path.exists()
        else default_protocol_manifest()
    )
    weights = manifest.get("overall_weights_v1")

    verses_raw: List[Dict[str, Any]] = []
    with open(verses_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            verses_raw.append(json.loads(line))

    scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
    for row in verses_raw:
        if row.get("unlabeled") or not isinstance(row.get("labels"), dict):
            continue
        lyrics = row.get("lyrics") or []
        if len(lyrics) < 2:
            continue
        try:
            scores = _score_one(lyrics, fast=fast)
        except Exception as e:
            logger.warning("score_verse failed for %s: %s", row.get("id"), e)
            continue
        scored.append((row, scores))

    report: Dict[str, Any] = {"n_scored": len(scored), "fast_mode": fast}

    label_to_pred = {
        "flow": lambda s: fitness_vector_from_scores(s)["flow"],
        "rhyme": lambda s: fitness_vector_from_scores(s)["rhyme"],
        "semantic": lambda s: fitness_vector_from_scores(s)["semantic"],
        "punchline": lambda s: fitness_vector_from_scores(s)["punchline"],
        "originality": lambda s: fitness_vector_from_scores(s)["novelty"],
        "fluency": lambda s: fluency_composite_from_scores(s),
        "overall": lambda s: axis_scalar_for_pairwise(s, "overall", weights=weights),
    }

    for lab_key, pred_fn in label_to_pred.items():
        ys: List[float] = []
        pr: List[float] = []
        for row, sc in scored:
            lab = row["labels"]
            if lab_key not in lab:
                continue
            ys.append(float(lab[lab_key]))
            pr.append(float(pred_fn(sc)))
        if len(ys) < 2:
            report[f"spearman_{lab_key}"] = None
            report[f"mse_{lab_key}"] = None
            continue
        rho, pval = _correlation_axis(ys, pr)
        report[f"spearman_{lab_key}"] = rho
        report[f"spearman_{lab_key}_pvalue"] = pval
        report[f"mse_{lab_key}"] = _mse(ys, pr)

    pairwise_block: Dict[str, Any] = {}
    if pairs_path and pairs_path.exists():
        id_to_scores = {row["id"]: sc for row, sc in scored}
        judgments: List[Tuple[str, float, float]] = []
        bad = 0
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = parse_pairwise_line(line, require_judgment=True)
                if rec.better is None:
                    continue
                sa = id_to_scores.get(rec.verse_a_id)
                sb = id_to_scores.get(rec.verse_b_id)
                if sa is None or sb is None:
                    bad += 1
                    continue
                axis = rec.axis or "overall"
                va = axis_scalar_for_pairwise(sa, axis, weights=weights)
                vb = axis_scalar_for_pairwise(sb, axis, weights=weights)
                judgments.append((rec.better, va, vb))
        summ = summarize_pairwise(judgments, epsilon=epsilon)
        pairwise_block = {
            "n_pairs_used": summ.full_matrix_n,
            "n_pairs_skipped_missing_verse": bad,
            "full_matrix_mean": summ.full_matrix_mean,
            "directional_accuracy_aux": summ.directional_mean,
            "directional_n": summ.directional_n,
            "epsilon": summ.epsilon,
        }
    report["pairwise"] = pairwise_block
    report["protocol_ref"] = str(protocol_path) if protocol_path else "defaults"
    return report
