"""
Canonical control registry and snapshot builders for causal experiment logging.

Provides: ControlSpec, CONTROL_REGISTRY (layer-tagged controls), build_control_snapshot_*
and flatten_controls for stable JSON-safe representation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Layer tags for targeted tuning: micro (word-level), line, verse, meta (evolution/scoring)
LAYER_MICRO = "micro"
LAYER_LINE = "line"
LAYER_VERSE = "verse"
LAYER_META = "meta"


@dataclass(frozen=True)
class ControlSpec:
    """Metadata for a single control knob."""
    key: str
    layer: str
    type: str  # "numeric" | "categorical" | "boolean"
    domain: Optional[Any] = None  # e.g. [0, 1], ["AABB", "ABAB"], None for freeform
    description: str = ""


def _spec(key: str, layer: str, type_: str, domain: Any = None, description: str = "") -> ControlSpec:
    return ControlSpec(key=key, layer=layer, type=type_, domain=domain, description=description)


# Canonical controls with layer tags for experiment analysis
CONTROL_REGISTRY: Dict[str, ControlSpec] = {
    # Meta: evolution / selection / scoring
    "population": _spec("population", LAYER_META, "numeric", [1, 5000], "Population size"),
    "generations": _spec("generations", LAYER_META, "numeric", [1, 1000], "Number of generations"),
    "elites": _spec("elites", LAYER_META, "numeric", [1, 100], "Elites preserved per generation"),
    "tournament_k": _spec("tournament_k", LAYER_META, "numeric", [2, 10], "Tournament size"),
    "immigrants": _spec("immigrants", LAYER_META, "numeric", [0, 100], "Random immigrants per generation"),
    "init": _spec("init", LAYER_META, "categorical", ["mixed", "random", "template", "lm"], "Population init mode"),
    "use_embeddings": _spec("use_embeddings", LAYER_META, "boolean", None, "Use embedding-based semantic"),
    "embedding_weight": _spec("embedding_weight", LAYER_META, "numeric", [0.0, 1.0], "Weight of embedding in semantic"),
    "multiobjective": _spec("multiobjective", LAYER_META, "boolean", None, "Pareto multi-objective evolution"),
    "use_niching": _spec("use_niching", LAYER_META, "boolean", None, "Niching in elite selection"),
    "min_fluency_accept": _spec("min_fluency_accept", LAYER_META, "numeric", [0.0, 1.0], "Min fluency to accept mutation"),
    "min_semantic_accept": _spec("min_semantic_accept", LAYER_META, "numeric", [0.0, 1.0], "Min semantic to accept"),
    "min_lexical_accept": _spec("min_lexical_accept", LAYER_META, "numeric", [0.0, 1.0], "Min lexical validity"),
    "min_ngram_fluency_accept": _spec("min_ngram_fluency_accept", LAYER_META, "numeric", [0.0, 1.0], "Min ngram fluency"),
    "use_lm_fluency": _spec("use_lm_fluency", LAYER_META, "boolean", None, "Blend LM perplexity in fluency"),
    "lm_fluency_weight": _spec("lm_fluency_weight", LAYER_META, "numeric", [0.0, 1.0], "LM weight in fluency blend"),
    "style_weight": _spec("style_weight", LAYER_META, "numeric", [0.0, 1.0], "Style similarity weight in fitness"),
    "require_theme_presence": _spec("require_theme_presence", LAYER_META, "boolean", None, "Require theme keyword in couplet"),
    # Verse-level
    "scheme": _spec("scheme", LAYER_VERSE, "categorical", ["AABB", "ABAB", "ABBA", "AAAA", "ABCB", "AABA"], "Rhyme scheme"),
    "num_lines": _spec("num_lines", LAYER_VERSE, "categorical", [4, 8, 16], "Lines per verse"),
    "theme": _spec("theme", LAYER_VERSE, "categorical", None, "Theme keywords (comma-separated)"),
    # Line / proposer (QD)
    "lm_budget": _spec("lm_budget", LAYER_LINE, "numeric", [0, 500], "LM mutation budget per generation"),
    "proposer_model": _spec("proposer_model", LAYER_LINE, "categorical", None, "Proposer model name"),
    "proposer_backend": _spec("proposer_backend", LAYER_LINE, "categorical", ["openai", "local_hf"], "Proposer backend"),
    "line_pop": _spec("line_pop", LAYER_LINE, "numeric", None, "Line population size"),
    "line_gens": _spec("line_gens", LAYER_LINE, "numeric", None, "Line generations per verse gen"),
    "line_seeds": _spec("line_seeds", LAYER_LINE, "numeric", None, "Line LM seed count"),
    "line_lm_budget": _spec("line_lm_budget", LAYER_LINE, "numeric", None, "Line LM mutation budget"),
    "compose_ratio": _spec("compose_ratio", LAYER_LINE, "numeric", [0.0, 1.0], "Composed offspring ratio"),
    "emitter_strategy": _spec("emitter_strategy", LAYER_LINE, "categorical", None, "Emitter strategy"),
    "archive_mode": _spec("archive_mode", LAYER_VERSE, "categorical", None, "MAP-Elites archive mode"),
    "style_genome": _spec("style_genome", LAYER_VERSE, "boolean", None, "Enable style genome"),
    "prompt_genome": _spec("prompt_genome", LAYER_VERSE, "boolean", None, "Enable prompt genome"),
    "novelty_weight": _spec("novelty_weight", LAYER_META, "numeric", [0.0, 1.0], "Novelty weight in QD"),
    "min_fluency": _spec("min_fluency", LAYER_META, "numeric", [0.0, 1.0], "Min fluency floor (QD)"),
    "min_semantic": _spec("min_semantic", LAYER_META, "numeric", [0.0, 1.0], "Min semantic floor (QD)"),
    "graph_top_k": _spec("graph_top_k", LAYER_LINE, "numeric", None, "Graph top-k"),
    "expensive_top_k": _spec("expensive_top_k", LAYER_LINE, "numeric", None, "Expensive scoring top-k"),
    "graph_edge_mode": _spec("graph_edge_mode", LAYER_LINE, "categorical", None, "Graph edge mode"),
    "fast_mode": _spec("fast_mode", LAYER_META, "boolean", None, "Fast mode"),
    "curriculum_switch_gen": _spec("curriculum_switch_gen", LAYER_META, "numeric", None, "Curriculum switch generation"),
    "prompt_llm_fraction": _spec("prompt_llm_fraction", LAYER_VERSE, "numeric", [0.0, 1.0], "Prompt LLM fraction"),
}


def flatten_controls(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a flat dict with stable key order and JSON-safe scalars.
    Nested dicts (e.g. fitness_weights) are JSON-stringified so they remain queryable as single keys
    (e.g. fitness_weights_json) for storage; or you can keep them as nested in config_json.
    """
    result: Dict[str, Any] = {}
    # Sort keys for determinism
    for key in sorted(snapshot.keys()):
        val = snapshot[key]
        if val is None:
            result[key] = None
        elif isinstance(val, (bool, int, float, str)):
            result[key] = val
        elif isinstance(val, (list, tuple)):
            result[key] = list(val)  # keep lists as-is for JSON
        elif isinstance(val, dict):
            # Keep nested dict for config_json; for fully flat use a single string key
            result[key] = val
        else:
            result[key] = str(val)
    return result


def build_control_snapshot_from_couplet_args(
    args: Any,
    defaults: Dict[str, Any],
    *,
    fitness_weights: Optional[Dict[str, float]] = None,
    mutation_weights: Optional[Dict[str, float]] = None,
    seed_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build full control snapshot for a couplet evolution run from CLI args and defaults.
    Used for DB logging and experiment arms.
    """
    snapshot: Dict[str, Any] = {
        "runner": "couplet",
        "theme": getattr(args, "theme", "") or "",
        "population": getattr(args, "population", defaults.get("population", 100)),
        "generations": getattr(args, "generations", defaults.get("generations", 10)),
        "elites": getattr(args, "elites", defaults.get("elites", 5)),
        "tournament_k": defaults.get("tournament_k", 3),
        "immigrants": getattr(args, "immigrants", defaults.get("immigrants", 10)),
        "init": getattr(args, "init", defaults.get("init", "mixed")),
        "use_embeddings": getattr(args, "use_embeddings", defaults.get("use_embeddings", False)),
        "embedding_weight": getattr(args, "embedding_weight", defaults.get("embedding_weight", 0.5)),
        "multiobjective": getattr(args, "multiobjective", defaults.get("multiobjective", False)),
        "use_niching": getattr(args, "use_niching", defaults.get("use_niching", False)),
        "min_fluency_accept": _resolve_float(args, "min_fluency", defaults.get("min_fluency"), 0.0),
        "min_semantic_accept": _resolve_float(args, "min_semantic", defaults.get("min_semantic"), 0.0),
        "min_lexical_accept": getattr(args, "min_lexical", defaults.get("min_lexical", 0.0)) or 0.0,
        "min_ngram_fluency_accept": _resolve_float(args, "min_ngram", defaults.get("min_ngram"), 0.0),
        "use_lm_fluency": getattr(args, "lm_fluency", defaults.get("lm_fluency", False)),
        "lm_fluency_weight": getattr(args, "lm_fluency_weight", defaults.get("lm_fluency_weight", 0.5)),
        "style_weight": getattr(args, "style_weight", defaults.get("style_weight", 0.1)) or 0.0,
        "require_theme_presence": getattr(args, "require_theme", defaults.get("require_theme", False)),
    }
    if fitness_weights is not None:
        snapshot["fitness_weights"] = fitness_weights
    if mutation_weights is not None:
        snapshot["mutation_weights"] = mutation_weights
    if seed_info is not None:
        snapshot["seed_info"] = seed_info
    if hasattr(args, "policy_mode"):
        snapshot["policy_mode"] = getattr(args, "policy_mode", "static")
    if hasattr(args, "epsilon"):
        snapshot["epsilon"] = getattr(args, "epsilon", 0.0)
    if hasattr(args, "_policy_source"):
        snapshot["policy_source"] = getattr(args, "_policy_source", "defaults")
    if hasattr(args, "_exploration_applied"):
        snapshot["exploration_applied"] = bool(getattr(args, "_exploration_applied", False))
    if hasattr(args, "_policy_version"):
        snapshot["policy_version"] = getattr(args, "_policy_version", None)
    if hasattr(args, "_policy_hash"):
        snapshot["policy_hash"] = getattr(args, "_policy_hash", None)
    if hasattr(args, "_sampled_policy_rank"):
        snapshot["sampled_policy_rank"] = getattr(args, "_sampled_policy_rank", None)
    return snapshot


def _resolve_float(args: Any, attr: str, default: Any, fallback: float) -> float:
    val = getattr(args, attr, default)
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


def build_control_snapshot_from_qd_args(
    args: Any,
    defaults: Dict[str, Any],
    *,
    seed_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build full control snapshot for a QD verse evolution run from CLI args and defaults.
    """
    snapshot: Dict[str, Any] = {
        "runner": "qd",
        "theme": getattr(args, "theme", "") or "",
        "population": getattr(args, "population", defaults.get("population", 100)),
        "generations": getattr(args, "generations", defaults.get("generations", 100)),
        "elites": getattr(args, "elites", defaults.get("elites", 5)),
        "immigrants": getattr(args, "immigrants", defaults.get("immigrants", 20)),
        "scheme": getattr(args, "scheme", defaults.get("scheme", "AABB")),
        "num_lines": getattr(args, "num_lines", defaults.get("num_lines", 4)),
        "init": getattr(args, "init", defaults.get("init", "lm")),
        "lm_budget": getattr(args, "lm_budget", defaults.get("lm_budget", 20)),
        "use_embeddings": getattr(args, "use_embeddings", defaults.get("use_embeddings", False)),
        "embedding_weight": getattr(args, "embedding_weight", defaults.get("embedding_weight", 0.4)),
        "proposer_model": getattr(args, "proposer_model", defaults.get("proposer_model", "gpt-4.1-nano")),
        "proposer_backend": getattr(args, "proposer_backend", defaults.get("proposer_backend", "openai")),
        "min_fluency": getattr(args, "min_fluency", defaults.get("min_fluency", 0.3)),
        "min_semantic": getattr(args, "min_semantic", defaults.get("min_semantic", 0.0)),
        "line_pop": getattr(args, "line_pop", defaults.get("line_pop", 1500)),
        "line_gens": getattr(args, "line_gens", defaults.get("line_gens", 3)),
        "line_seeds": getattr(args, "line_seeds", defaults.get("line_seeds", 80)),
        "line_lm_budget": getattr(args, "line_lm_budget", defaults.get("line_lm_budget", 15)),
        "compose_ratio": getattr(args, "compose_ratio", defaults.get("compose_ratio", 0.5)),
        "emitter_strategy": getattr(args, "emitter_strategy", defaults.get("emitter_strategy", "multi")),
        "archive_mode": getattr(args, "archive_mode", defaults.get("archive_mode", "compact_style")),
        "style_genome": getattr(args, "style_genome", defaults.get("style_genome", True)),
        "prompt_genome": getattr(args, "prompt_genome", defaults.get("prompt_genome", True)),
        "novelty_weight": getattr(args, "novelty_weight", defaults.get("novelty_weight", 0.3)),
        "fast_mode": getattr(args, "fast_mode", defaults.get("fast_mode", True)),
        "graph_top_k": getattr(args, "graph_top_k", defaults.get("graph_top_k", 24)),
        "expensive_top_k": getattr(args, "expensive_top_k", defaults.get("expensive_top_k", 40)),
        "graph_edge_mode": getattr(args, "graph_edge_mode", defaults.get("graph_edge_mode", "phonetic")),
        "curriculum_switch_gen": getattr(args, "curriculum_switch_gen", defaults.get("curriculum_switch_gen", 20)),
        "prompt_llm_fraction": getattr(args, "prompt_llm_fraction", defaults.get("prompt_llm_fraction", 0.2)),
    }
    if seed_info is not None:
        snapshot["seed_info"] = seed_info
    if hasattr(args, "policy_mode"):
        snapshot["policy_mode"] = getattr(args, "policy_mode", "static")
    if hasattr(args, "epsilon"):
        snapshot["epsilon"] = getattr(args, "epsilon", 0.0)
    if hasattr(args, "_policy_source"):
        snapshot["policy_source"] = getattr(args, "_policy_source", "defaults")
    if hasattr(args, "_exploration_applied"):
        snapshot["exploration_applied"] = bool(getattr(args, "_exploration_applied", False))
    if hasattr(args, "_policy_version"):
        snapshot["policy_version"] = getattr(args, "_policy_version", None)
    if hasattr(args, "_policy_hash"):
        snapshot["policy_hash"] = getattr(args, "_policy_hash", None)
    if hasattr(args, "_sampled_policy_rank"):
        snapshot["sampled_policy_rank"] = getattr(args, "_sampled_policy_rank", None)
    arm = getattr(args, "arm", None) or os.environ.get("RAPBOT_CONTINUOUS_ARM", "")
    if arm:
        snapshot["arm"] = str(arm)
    return snapshot


def get_controls_by_layer(layer: str) -> List[str]:
    """Return control keys for a given layer."""
    return [k for k, spec in CONTROL_REGISTRY.items() if spec.layer == layer]
