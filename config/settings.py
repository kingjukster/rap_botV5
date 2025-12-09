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

    def as_dict(self) -> Dict[str, str]:
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
        }


DEFAULTS: Dict[str, str] = {
    "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
    "adapter_dir": "lora_elite_v2",
    "tokenizer_dir": "elite_tokenizer",
    "rhyme_groups_csv": "rhymes_grouped.csv",
    "siamese_model_dir": "rhyme_siamese",
    "elite_corpus_path": "data/elite_kaggle_corpus_clean.txt",
    "lora_output_dir": "checkpoints/elite_qwen_siamese",
    "tokenizer_save_dir": "elite_tokenizer",
    "ngram_output_path": "elite_ngrams.tsv",
    "topic_model_path": "elite_w2v.model",
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
