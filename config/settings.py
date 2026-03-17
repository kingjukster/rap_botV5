"""
Shared configuration loader for Rap Bot / Blacklight.

Resolves core paths via:
  1. Defaults relative to the repository root.
  2. Optional config file (JSON/YAML) provided via argument or RAPBOT_CONFIG.
  3. Environment variables for quick overrides (RAPBOT_*).

This keeps entrypoints free of hard-coded absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass
class Settings:
    base_model_name: str
    adapter_dir: Path
    tokenizer_dir: Path
    rhyme_groups_csv: Path
    siamese_model_dir: Path
    elite_corpus_path: Path
    lora_output_dir: Path
    tokenizer_save_dir: Path
    ngram_output_path: Path
    topic_model_path: Path
    generation_log_path: Path
    scored_dataset_path: Path
    weighted_corpus_path: Path
    local_critic_dir: Path
    generation_defaults: Dict[str, Any]
    stage3: Dict[str, Any]
    critic: Dict[str, Any]
    stats: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """
        Convenience helper for logging / introspection.
        """
        return {
            "base_model_name": self.base_model_name,
            "adapter_dir": str(self.adapter_dir),
            "tokenizer_dir": str(self.tokenizer_dir),
            "rhyme_groups_csv": str(self.rhyme_groups_csv),
            "siamese_model_dir": str(self.siamese_model_dir),
            "elite_corpus_path": str(self.elite_corpus_path),
            "lora_output_dir": str(self.lora_output_dir),
            "tokenizer_save_dir": str(self.tokenizer_save_dir),
            "ngram_output_path": str(self.ngram_output_path),
            "topic_model_path": str(self.topic_model_path),
            "generation_log_path": str(self.generation_log_path),
            "scored_dataset_path": str(self.scored_dataset_path),
            "weighted_corpus_path": str(self.weighted_corpus_path),
            "local_critic_dir": str(self.local_critic_dir),
            "generation_defaults": _stringify_section(self.generation_defaults),
            "stage3": _stringify_section(self.stage3),
            "critic": _stringify_section(self.critic),
            "stats": _stringify_section(self.stats),
        }


DEFAULTS: Dict[str, str] = {
    "base_model_name": "Qwen/Qwen2.5-14B-Instruct",
    "adapter_dir": "lora_elite_v2",
    "tokenizer_dir": "elite_tokenizer",
    "rhyme_groups_csv": "data/rhymes_grouped.csv",
    "siamese_model_dir": "rhyme_siamese",
    "elite_corpus_path": "data/elite_kaggle_corpus_clean.txt",
    "lora_output_dir": "checkpoints/elite_qwen_siamese",
    "tokenizer_save_dir": "elite_tokenizer",
    "ngram_output_path": "artifacts/elite_ngrams.tsv",
    "topic_model_path": "artifacts/elite_w2v.model",
    "generation_log_path": "data/generated_raw.jsonl",
    "scored_dataset_path": "data/scored_dataset.jsonl",
    "weighted_corpus_path": "data/weighted_corpus_stage3.txt",
    "local_critic_dir": "models/local_critic",
}

PATH_FIELDS = {
    "adapter_dir",
    "tokenizer_dir",
    "rhyme_groups_csv",
    "siamese_model_dir",
    "elite_corpus_path",
    "lora_output_dir",
    "tokenizer_save_dir",
    "ngram_output_path",
    "topic_model_path",
    "generation_log_path",
    "scored_dataset_path",
    "weighted_corpus_path",
    "local_critic_dir",
}

GENERATION_SECTION_DEFAULTS: Dict[str, Any] = {
    "scheme": "AABB",
    "num_bars": 16,
    "candidates": 8,
    "max_new_tokens": 40,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "attempts": 4,
    "verse_accept_threshold": 0.3,
    "log_json": "data/generated_raw.jsonl",
    "persona": None,
    "theme_hint": None,
    "style_hint": None,
    "topic_hint": None,
    "vocab_hint": None,
    "syllable_map": None,
}

STAGE3_SECTION_DEFAULTS: Dict[str, Any] = {
    "samples_per_seed": 4,
    "parallel_workers": 1,
    "refresh_rhymes": True,
    "min_overall_score": 0.0,
    "min_average_score": 0.0,
    "hist_bins": 20,
    "seed_manifest": None,
}

CRITIC_SECTION_DEFAULTS: Dict[str, Any] = {
    "model": "gpt-4o-mini",
    "sleep": 0.5,
    "concurrency": 2,
    "max_per_run": None,
    "retry_backoff": 5.0,
    "max_retries": 6,
    "manifest_path": "data/critic_manifests/run_manifest.jsonl",
    "input_cost_per_mtok": 0.0,
    "output_cost_per_mtok": 0.0,
}

STATS_SECTION_DEFAULTS: Dict[str, Any] = {
    "summary_path": "data/stats/stage3_summary.json",
    "per_seed_csv": "data/stats/stage3_per_seed.csv",
    "histogram_path": "data/stats/stage3_histograms.json",
}

GEN_SECTION_PATHS = {"log_json"}
STAGE3_SECTION_PATHS = {"seed_manifest"}
CRITIC_SECTION_PATHS = {"manifest_path"}
STATS_SECTION_PATHS = {"summary_path", "per_seed_csv", "histogram_path"}

ENV_MAP = {
    "base_model_name": "RAPBOT_BASE_MODEL",
    "adapter_dir": "RAPBOT_ADAPTER_DIR",
    "tokenizer_dir": "RAPBOT_TOKENIZER_DIR",
    "rhyme_groups_csv": "RAPBOT_RHYME_CSV",
    "siamese_model_dir": "RAPBOT_SIAMESE_DIR",
    "elite_corpus_path": "RAPBOT_ELITE_CORPUS",
    "lora_output_dir": "RAPBOT_LORA_OUT",
    "tokenizer_save_dir": "RAPBOT_TOKENIZER_SAVE",
    "ngram_output_path": "RAPBOT_NGRAM_OUTPUT",
    "topic_model_path": "RAPBOT_TOPIC_MODEL",
    "generation_log_path": "RAPBOT_GENERATION_LOG",
    "scored_dataset_path": "RAPBOT_SCORED_DATASET",
    "weighted_corpus_path": "RAPBOT_WEIGHTED_CORPUS",
    "local_critic_dir": "RAPBOT_LOCAL_CRITIC_DIR",
}


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load shared settings, honoring overrides from config files / env vars.
    """
    base_dir = _repo_root()
    cfg_data: Dict[str, Any] = {}

    cfg_path = config_path or os.environ.get("RAPBOT_CONFIG")
    if not cfg_path:
        default_cfg = base_dir / "config" / "rapbot.yaml"
        if default_cfg.exists():
            cfg_path = str(default_cfg)
    if cfg_path:
        cfg_file = _resolve_path(cfg_path, base_dir)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")
        cfg_data = _load_config_file(cfg_file)

    env_overrides = {
        field: os.environ[var]
        for field, var in ENV_MAP.items()
        if os.environ.get(var)
    }

    resolved: Dict[str, Any] = {}
    for field, default_value in DEFAULTS.items():
        value = env_overrides.get(field)
        if value is None:
            value = cfg_data.get(field)
        if value is None:
            # Also check nested "paths" dict for convenience
            paths_section = cfg_data.get("paths") if isinstance(cfg_data.get("paths"), dict) else {}
            if paths_section:
                value = paths_section.get(field)
        if value is None:
            value = default_value

        if field in PATH_FIELDS:
            resolved[field] = _resolve_path(value, base_dir)
        else:
            resolved[field] = str(value)

    generation_defaults = _resolve_section(
        "generation",
        GENERATION_SECTION_DEFAULTS,
        cfg_data,
        base_dir,
        GEN_SECTION_PATHS,
    )
    stage3_defaults = _resolve_section(
        "stage3",
        STAGE3_SECTION_DEFAULTS,
        cfg_data,
        base_dir,
        STAGE3_SECTION_PATHS,
    )
    critic_defaults = _resolve_section(
        "critic",
        CRITIC_SECTION_DEFAULTS,
        cfg_data,
        base_dir,
        CRITIC_SECTION_PATHS,
    )
    stats_defaults = _resolve_section(
        "stats",
        STATS_SECTION_DEFAULTS,
        cfg_data,
        base_dir,
        STATS_SECTION_PATHS,
    )

    return Settings(
        base_model_name=str(resolved["base_model_name"]),
        adapter_dir=Path(resolved["adapter_dir"]),
        tokenizer_dir=Path(resolved["tokenizer_dir"]),
        rhyme_groups_csv=Path(resolved["rhyme_groups_csv"]),
        siamese_model_dir=Path(resolved["siamese_model_dir"]),
        elite_corpus_path=Path(resolved["elite_corpus_path"]),
        lora_output_dir=Path(resolved["lora_output_dir"]),
        tokenizer_save_dir=Path(resolved["tokenizer_save_dir"]),
        ngram_output_path=Path(resolved["ngram_output_path"]),
        topic_model_path=Path(resolved["topic_model_path"]),
        generation_log_path=Path(resolved["generation_log_path"]),
        scored_dataset_path=Path(resolved["scored_dataset_path"]),
        weighted_corpus_path=Path(resolved["weighted_corpus_path"]),
        local_critic_dir=Path(resolved["local_critic_dir"]),
        generation_defaults=generation_defaults,
        stage3=stage3_defaults,
        critic=critic_defaults,
        stats=stats_defaults,
    )


def _repo_root() -> Path:
    env_override = os.environ.get("RAPBOT_HOME")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str | os.PathLike[str], base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_config_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs.")
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text or "{}")
    else:
        raise ValueError(f"Unsupported config format: {path}")

    if not isinstance(data, dict):
        raise ValueError("Config file must define a dictionary at the top level.")
    return data


def _resolve_section(
    name: str,
    defaults: Dict[str, Any],
    cfg_data: Dict[str, Any],
    base_dir: Path,
    path_keys: Optional[set[str]] = None,
) -> Dict[str, Any]:
    section = cfg_data.get(name) if isinstance(cfg_data.get(name), dict) else {}
    merged: Dict[str, Any] = {**defaults, **(section or {})}
    if path_keys:
        for key in path_keys:
            if merged.get(key):
                merged[key] = _resolve_path(merged[key], base_dir)
    return merged


def _stringify_section(section: Dict[str, Any]) -> Dict[str, Any]:
    rendered = {}
    for key, value in section.items():
        if isinstance(value, Path):
            rendered[key] = str(value)
        else:
            rendered[key] = value
    return rendered


def get_elite_corpus_path() -> Path:
    """
    Single source of truth for elite corpus path. Tries settings, then env, then
    default paths. Falls back to alternate files if primary does not exist.
    """
    base = _repo_root()
    path: Optional[Path] = None
    try:
        settings = load_settings()
        path = settings.elite_corpus_path
        if path and not path.is_absolute():
            path = base / path
    except Exception:
        pass
    if path is None:
        env_val = os.environ.get("RAPBOT_ELITE_CORPUS")
        if env_val:
            path = Path(env_val).expanduser()
            if not path.is_absolute():
                path = base / path
    if path is None:
        path = base / "data" / "elite_kaggle_corpus_clean.txt"
    if not path.exists():
        for alt in ("data/elite_kaggle_corpus_clean_plus.jsonl", "data/phaseA_kaggle_verse.txt"):
            candidate = base / alt
            if candidate.exists():
                return candidate
    return path
