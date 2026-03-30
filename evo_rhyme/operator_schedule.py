"""Blend mutation operator weights using recent operator_events counts (frequency prior, not Δfitness)."""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def blend_mutation_weights_from_counts(
    base: Dict[str, float],
    counts: Dict[str, int],
) -> Optional[Dict[str, float]]:
    """Return new weights matching *base* keys; scale so sum equals sum(base).

    Operators with higher recent usage get modestly higher sampling weight; keys
    absent from *counts* keep multiplier 1.0. Returns None if *counts* is empty.
    """
    if not base or not counts:
        return None
    total_c = sum(max(0, int(v)) for v in counts.values())
    if total_c <= 0:
        return None
    n_keys = max(1, len(base))
    out: Dict[str, float] = {}
    for k, w0 in base.items():
        w0f = float(w0)
        if w0f <= 0:
            out[k] = w0f
            continue
        c = max(0, int(counts.get(k, 0)))
        share = c / float(total_c)
        mult = 1.0 + min(1.5, share * n_keys * 2.0)
        out[k] = max(1e-8, w0f * mult)
    s0 = sum(float(v) for v in base.values())
    s1 = sum(out.values())
    if s1 <= 0 or s0 <= 0:
        return None
    scale = s0 / s1
    blended = {k: float(v) * scale for k, v in out.items()}
    logger.info(
        "Blended mutation weights from operator_events (top boosts): %s",
        sorted(
            ((k, round(blended[k] / max(1e-8, float(base[k])), 3)) for k in base if float(base[k]) > 0
             and blended.get(k, 0) > float(base[k]) * 1.05),
            key=lambda t: -t[1],
        )[:8],
    )
    return blended


def load_blended_mutation_weights_from_db(
    base: Dict[str, float],
    *,
    max_runs: int = 25,
) -> Optional[Dict[str, float]]:
    """Fetch recent operator counts from DB and blend into *base*."""
    try:
        from evo_rhyme import db as _db
        if not _db.db_enabled():
            return None
        counts = _db.operator_event_counts_recent_runs(max_runs=max_runs)
        return blend_mutation_weights_from_counts(base, counts)
    except Exception as e:
        logger.warning("load_blended_mutation_weights_from_db failed: %s", e)
        return None
