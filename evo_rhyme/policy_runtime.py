"""
Runtime helpers for learned-policy loading and epsilon-greedy exploration.
"""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from evo_rhyme.experiment_controls import CONTROL_REGISTRY


def load_learned_policy(path_value: str, base_dir: Path) -> Tuple[Dict[str, Any], str]:
    """
    Load learned policy JSON and return (recommended_controls, source_path).
    Returns ({}, source) when file is missing or malformed.
    """
    policy_path = Path(path_value)
    if not policy_path.is_absolute():
        policy_path = (base_dir / policy_path).resolve()
    if not policy_path.exists():
        return {}, str(policy_path)
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
        controls = data.get("recommended_controls") if isinstance(data, dict) else {}
        return controls if isinstance(controls, dict) else {}, str(policy_path)
    except Exception:
        return {}, str(policy_path)


def _weighted_choice_top_configs(top_configs: list[Dict[str, Any]]) -> Tuple[Dict[str, Any], int]:
    """Pick one top config with score-weighted sampling. Excludes configs with score <= 0.
    Returns (controls, rank)."""
    if not top_configs:
        return {}, -1
    # Exclude failed configs (score <= 0) so they aren't resampled
    viable = [
        (i, item)
        for i, item in enumerate(top_configs)
        if isinstance(item, dict)
    ]
    weights = []
    indices = []
    for i, item in viable:
        try:
            score = float(item.get("score", 0.0))
        except Exception:
            score = 0.0
        if score <= 0:
            continue
        weights.append(score)
        indices.append((i, item))
    if not indices:
        return {}, -1
    idx_in_viable = random.choices(range(len(indices)), weights=weights, k=1)[0]
    orig_idx, chosen = indices[idx_in_viable]
    ctrls = chosen.get("controls") if isinstance(chosen, dict) else {}
    return (ctrls if isinstance(ctrls, dict) else {}), orig_idx + 1


def resolve_policy_controls(path_value: str, base_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Resolve runtime controls from learned policy with metadata.

    Supports two shapes:
    - {recommended_controls: {...}}
    - {top_configs: [{controls: {...}, score: ...}, ...], ...}
    """
    policy_path = Path(path_value)
    if not policy_path.is_absolute():
        policy_path = (base_dir / policy_path).resolve()
    metadata: Dict[str, Any] = {
        "policy_source": str(policy_path),
        "policy_version": None,
        "policy_hash": None,
        "sampled_policy_rank": None,
    }
    if not policy_path.exists():
        return {}, metadata
    try:
        raw = policy_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}, metadata
        metadata["policy_version"] = data.get("policy_version")
        metadata["policy_hash"] = data.get("policy_hash") or hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        top_configs = data.get("top_configs")
        if isinstance(top_configs, list) and top_configs:
            ctrls, rank = _weighted_choice_top_configs(top_configs)
            metadata["sampled_policy_rank"] = rank
            if ctrls:
                return ctrls, metadata
            # All top_configs had score <= 0; fall back to recommended_controls
        controls = data.get("recommended_controls")
        return (controls if isinstance(controls, dict) else {}), metadata
    except Exception:
        return {}, metadata


def apply_controls_to_args(
    args: Any,
    controls: Dict[str, Any],
    *,
    protected_keys: Optional[Set[str]] = None,
) -> int:
    """
    Apply control values onto argparse namespace attrs where names match.
    Returns number of updated fields.
    """
    protected = protected_keys or set()
    updates = 0
    for key, value in controls.items():
        if key in protected:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
            updates += 1
    return updates


def maybe_epsilon_perturb(
    args: Any,
    *,
    epsilon: float,
    protected_keys: Optional[Set[str]] = None,
) -> bool:
    """
    Epsilon-greedy exploration: with probability epsilon, perturb numeric controls.
    Uses CONTROL_REGISTRY domains when available.
    Returns True if perturbation was applied.
    """
    if epsilon <= 0 or random.random() >= epsilon:
        return False
    protected = protected_keys or set()
    for key, spec in CONTROL_REGISTRY.items():
        if key in protected or not hasattr(args, key):
            continue
        current = getattr(args, key, None)
        if current is None:
            continue
        if not isinstance(current, (int, float)):
            continue
        domain = spec.domain
        if isinstance(domain, list) and len(domain) == 2 and all(isinstance(v, (int, float)) for v in domain):
            lo = float(domain[0])
            hi = float(domain[1])
            span = hi - lo
            if span <= 0:
                continue
            # Small exploratory jitter around current value
            jitter = random.uniform(-0.15, 0.15) * span
            nxt = max(lo, min(hi, float(current) + jitter))
            if isinstance(current, int):
                nxt = int(round(nxt))
            setattr(args, key, nxt)
    return True
