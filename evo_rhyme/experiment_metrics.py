"""
Fitness vector view and outcome aggregation for experiment analysis.

Maps raw scores_json (couplet or verse) to a stable 5-axis objective space and
provides aggregation over candidates for run-level metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Schema version for reproducibility of fitness vector mapping
FITNESS_VECTOR_SCHEMA_VERSION = "v1"

# Axis keys in the canonical fitness vector
FITNESS_VECTOR_KEYS = ("rhyme", "flow", "semantic", "novelty", "punchline")


def ucb_score(
    mean_reward: float,
    n_pulls: int,
    total_pulls: int,
    exploration_c: float = 1.414,
) -> float:
    """UCB1-style score for uncertainty-aware policy arm ranking."""
    if n_pulls <= 0:
        return float("inf")
    return float(
        mean_reward
        + exploration_c * math.sqrt(math.log(max(total_pulls, 1)) / n_pulls)
    )


def fitness_vector_from_scores(scores: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Map raw component scores (couplet or verse) to the 5-axis fitness vector.

    - rhyme: end_rhyme, internal_rhyme, rhyme_graph, multisyllabic (and inverse of rhyme penalties)
    - flow: syllable_balance, stress_alignment, flow_alignment, beat_fit
    - semantic: semantic, coherence
    - novelty: novelty, 1 - corpus_overlap (or similar)
    - punchline: punchline

    Handles both couplet score keys and verse score keys (e.g. rhyme_scheme_score, flow_alignment).
    Missing keys contribute 0.0; result is always a dict with all five keys in [0, 1].
    """
    out: Dict[str, float] = {
        "rhyme": 0.0,
        "flow": 0.0,
        "semantic": 0.0,
        "novelty": 0.0,
        "punchline": 0.0,
    }
    if not scores:
        return out

    # Rhyme: end_rhyme, internal_rhyme, rhyme_graph, multisyllabic; verse: rhyme_scheme_score, internal_rhyme, rhyme_graph_*
    rhyme_components = []
    for k in ("end_rhyme", "internal_rhyme", "rhyme_graph", "multisyllabic", "rhyme_scheme_score"):
        if k in scores and scores[k] is not None:
            rhyme_components.append(float(scores[k]))
    for k in ("rhyme_graph_density", "rhyme_graph_cluster_coeff", "rhyme_graph_chain_length"):
        if k in scores and scores[k] is not None:
            rhyme_components.append(float(scores[k]))
    if rhyme_components:
        out["rhyme"] = sum(rhyme_components) / len(rhyme_components)
    # Penalties reduce rhyme perception
    for k in ("rhyme_family_repetition_penalty", "weak_tail_penalty"):
        if k in scores and scores[k] is not None:
            out["rhyme"] = max(0.0, out["rhyme"] + float(scores[k]))  # penalties are negative

    # Flow: syllable_balance, stress_alignment; verse: flow_alignment, beat_fit
    flow_components = []
    for k in ("syllable_balance", "stress_alignment", "flow_alignment", "beat_fit", "flow_continuity_score"):
        if k in scores and scores[k] is not None:
            flow_components.append(float(scores[k]))
    if flow_components:
        out["flow"] = sum(flow_components) / len(flow_components)
    else:
        out["flow"] = 0.5  # neutral if no flow signal

    # Semantic
    sem_components = []
    for k in ("semantic", "coherence"):
        if k in scores and scores[k] is not None:
            sem_components.append(float(scores[k]))
    if sem_components:
        out["semantic"] = sum(sem_components) / len(sem_components)

    # Novelty: novelty score and inverse of corpus overlap
    novelty_val = scores.get("novelty")
    if novelty_val is not None:
        out["novelty"] = float(novelty_val)
    corpus_overlap = scores.get("corpus_overlap_penalty")
    if corpus_overlap is not None:
        # penalty is negative or positive fraction; 1 - overlap is novelty
        overlap = max(0.0, min(1.0, float(corpus_overlap)))
        out["novelty"] = (out["novelty"] + (1.0 - overlap)) / 2.0 if novelty_val is not None else (1.0 - overlap)

    # Punchline
    if "punchline" in scores and scores["punchline"] is not None:
        out["punchline"] = max(0.0, min(1.0, float(scores["punchline"])))

    # Clamp all to [0, 1]
    for k in FITNESS_VECTOR_KEYS:
        out[k] = max(0.0, min(1.0, out[k]))
    return out


def aggregate_outcome(
    candidates: List[Dict[str, Any]],
    mode: str = "best",
    top_k: int = 5,
    score_key: str = "fitness",
    scores_key: str = "scores",
) -> Dict[str, float]:
    """
    Aggregate outcomes across candidates for a run.

    candidates: list of dicts with at least score_key (e.g. "fitness") and optionally
                scores_key (e.g. "scores") for component breakdown.
    mode: "best" = single best by score_key; "top_k_mean" = mean of top_k; "mean" = mean of all.
    top_k: used when mode == "top_k_mean".
    score_key: key for scalar fitness (e.g. "fitness").
    scores_key: key for component scores dict (e.g. "scores"); used to compute fitness vector for best/top_k.

    Returns dict with: fitness (aggregate scalar), fitness_vector (5-axis from best or mean of top_k),
    and optionally per-axis mean if mode is mean/top_k_mean.
    """
    if not candidates:
        return {
            "fitness": 0.0,
            "fitness_vector": {k: 0.0 for k in FITNESS_VECTOR_KEYS},
        }

    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.get(score_key) or 0.0),
        reverse=True,
    )

    if mode == "best":
        best = sorted_candidates[0]
        fitness = float(best.get(score_key) or 0.0)
        scores = best.get(scores_key)
        return {
            "fitness": fitness,
            "fitness_vector": fitness_vector_from_scores(scores),
        }
    if mode == "top_k_mean":
        subset = sorted_candidates[:top_k]
    else:
        subset = sorted_candidates

    fitnesses = [float(c.get(score_key) or 0.0) for c in subset]
    mean_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    vectors = [fitness_vector_from_scores(c.get(scores_key)) for c in subset]
    mean_vector: Dict[str, float] = {}
    for k in FITNESS_VECTOR_KEYS:
        mean_vector[k] = sum(v[k] for v in vectors) / len(vectors) if vectors else 0.0
    return {
        "fitness": mean_fitness,
        "fitness_vector": mean_vector,
    }
