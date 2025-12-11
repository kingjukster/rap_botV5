#!/usr/bin/env python
"""
run_stage3_pipeline.py

Utility wrapper that chains the main Stage-3 steps:
  1. Optionally refresh rhymes_grouped.csv via update_rhyme_groups.py
  2. Generate multiple verses per seed using generate_rhymed_verse.py
  3. Optionally train the local critic head (train_local_critic.py)
  4. Consolidate logs + critic scores via consolidate_stage3.py

This lets you fire off the end-to-end loop with a single command.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import load_settings
from scripts.tools.update_rhyme_groups import expand_rhyme_groups


def run_cmd(cmd: List[str]):
    print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_parallel(commands: List[List[str]], workers: int):
    if not commands:
        return
    if workers <= 1:
        for cmd in commands:
            run_cmd(cmd)
        return

    print(f"[PIPELINE] Dispatching {len(commands)} generation jobs with {workers} parallel worker(s).")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_cmd, cmd) for cmd in commands]
        for future in as_completed(futures):
            future.result()


def run_inprocess_generation(jobs: List[Dict[str, Any]], settings):
    if not jobs:
        return
    from argparse import Namespace
    from scripts.generation import generate_rhymed_verse as grv

    total = len(jobs)
    for idx, job in enumerate(jobs, start=1):
        seed_preview = job.get("seed_id") or job.get("seed")
        print(f"[PIPELINE] [in-process] job {idx}/{total} seed='{seed_preview}' scheme={job.get('scheme')}")
        ns = Namespace(**job)
        grv.run_generation(ns, settings)


def rhyme_csv_is_stale(corpus_path: Path, csv_path: Path) -> bool:
    csv_path = Path(csv_path)
    corpus_path = Path(corpus_path)
    if not csv_path.exists():
        return True
    if not corpus_path.exists():
        return False
    csv_mtime = csv_path.stat().st_mtime
    corpus_mtime = corpus_path.stat().st_mtime
    if csv_path.stat().st_size == 0:
        return True
    return csv_mtime < corpus_mtime


@dataclass
class SeedSpec:
    text: str
    scheme: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    persona: Optional[str] = None
    theme: Optional[str] = None
    style: Optional[str] = None
    topic: Optional[str] = None
    vocab: Optional[str] = None
    seed_id: Optional[str] = None
    samples: Optional[int] = None
    target_syllables: Optional[int] = None
    meter_sigma: Optional[float] = None
    verse_accept_threshold: Optional[float] = None
    syllable_map: Optional[List[int]] = None

    def cli_args(self) -> List[str]:
        args: List[str] = ["--seed", self.text]
        if self.seed_id:
            args.extend(["--seed_id", self.seed_id])
        if self.tags:
            args.extend(["--seed_tags", ",".join(self.tags)])
        if self.persona:
            args.extend(["--persona", self.persona])
        if self.theme:
            args.extend(["--theme_hint", self.theme])
        if self.style:
            args.extend(["--style_hint", self.style])
        if self.topic:
            args.extend(["--topic_hint", self.topic])
        if self.vocab:
            args.extend(["--vocab_hint", self.vocab])
        if self.syllable_map:
            map_str = ",".join(str(v) for v in self.syllable_map)
            args.extend(["--syllable_map", map_str])
        return args

    def apply_generation_overrides(self, cmd: List[str]):
        if self.target_syllables is not None:
            cmd.extend(["--target_syllables", str(self.target_syllables)])
        if self.meter_sigma is not None:
            cmd.extend(["--meter_sigma", str(self.meter_sigma)])
        if self.verse_accept_threshold is not None:
            cmd.extend(["--verse_accept_threshold", str(self.verse_accept_threshold)])

    def sample_count(self, default_samples: int) -> int:
        samples = self.samples if isinstance(self.samples, int) else default_samples
        return max(1, samples)


def sanitize_run_name(name: str) -> str:
    base = name.strip()
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base)
    return base or datetime.utcnow().strftime("refresh_%Y%m%d-%H%M%S")


def append_registry(registry_path: Path, record: Dict[str, Any]):
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def deploy_artifact(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if target.exists() or target.is_symlink():
        backup = target.with_name(f"{target.name}_backup_{timestamp}")
        shutil.move(str(target), str(backup))
    shutil.copytree(source, target)
    print(f"[PIPELINE] Promoted {source} -> {target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Stage-3 generation + consolidation pipeline.")
    parser.add_argument("--artist", type=str, default="mf_doom")
    parser.add_argument("--scheme", type=str, default=None)
    parser.add_argument("--num_bars", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--attempts", type=int, default=None)
    parser.add_argument("--verse_accept_threshold", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", default=[], help="List of seed strings.")
    parser.add_argument("--seeds_file", type=str, default=None, help="Optional file with one seed per line.")
    parser.add_argument("--samples_per_seed", type=int, default=None, help="How many verse generations per seed.")
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=None,
        help="How many verse generation commands to run concurrently.",
    )
    parser.add_argument("--persona", type=str, default=None, help="Optional persona hint applied to all seeds.")
    parser.add_argument("--theme_hint", type=str, default=None, help="Global theme overlay for prompts.")
    parser.add_argument("--style_hint", type=str, default=None, help="Global style cue (e.g., noir multis).")
    parser.add_argument("--topic_hint", type=str, default=None, help="Global topic cue for prompts.")
    parser.add_argument(
        "--vocab_hint",
        type=str,
        default=None,
        help="Comma-separated vocabulary hints applied when seed lacks its own.",
    )
    parser.add_argument(
        "--syllable_map",
        type=str,
        default=None,
        help="Global syllable map (comma-separated) when seeds do not define one.",
    )
    parser.add_argument("--log_json", type=str, default=None, help="Override log file path for generate_rhymed_verse.")
    parser.add_argument(
        "--seed_manifest",
        type=str,
        default=None,
        help="Optional JSON/JSONL manifest with structured seed metadata.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config path.")
    parser.add_argument("--critic_scores", type=str, default=None, help="Critic scores JSONL (optional).")
    parser.add_argument(
        "--local_critic_head",
        type=str,
        default=None,
        help="Path to trained reward head (supports comma-separated list for ensembles).",
    )
    parser.add_argument(
        "--local_critic_calibration",
        type=str,
        default=None,
        help="Optional calibration JSON for the local critic outputs.",
    )
    parser.add_argument(
        "--skip_update_rhymes",
        action="store_true",
        help="Skip update_rhyme_groups stage.",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation stage (useful if logs already exist).",
    )
    parser.add_argument(
        "--train_local_critic",
        action="store_true",
        help="Train the local critic head after generation.",
    )
    parser.add_argument(
        "--skip_consolidate",
        action="store_true",
        help="Skip consolidate_stage3.py step.",
    )
    parser.add_argument(
        "--force_refresh_rhymes",
        action="store_true",
        help="Always run the rhyme CSV refresh even if files look up to date.",
    )
    parser.add_argument(
        "--subprocess_generation",
        action="store_true",
        help="Run each verse via a subprocess (disables in-process model reuse).",
    )
    parser.add_argument(
        "--train_lora_refresh",
        action="store_true",
        help="After consolidation, fine-tune a new LoRA on the weighted corpus.",
    )
    parser.add_argument(
        "--lora_run_name",
        type=str,
        default=None,
        help="Optional name for the LoRA refresh run (defaults to timestamp).",
    )
    parser.add_argument(
        "--promote_lora",
        action="store_true",
        help="After LoRA refresh, copy the new adapter/tokenizer into adapter_dir/tokenizer_dir.",
    )
    return parser.parse_args()


def _parse_seed_tags(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, list):
        raw = value
    else:
        return []
    tags = [str(tag).strip() for tag in raw if str(tag).strip()]
    seen = set()
    ordered: List[str] = []
    for tag in tags:
        if tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    return ordered


def _parse_syllable_map(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    tokens: List[Any]
    if isinstance(value, str):
        normalized = value.replace(";", ",")
        tokens = [token.strip() for token in normalized.split(",")]
    elif isinstance(value, list):
        tokens = value
    else:
        return None
    mapping = []
    for token in tokens:
        num = _maybe_int(token)
        if num:
            mapping.append(num)
    return mapping or None


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_manifest_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    records: List[Dict[str, Any]] = []
    if suffix == ".json":
        data = json.loads(text or "{}")
        if isinstance(data, dict):
            data = data.get("seeds", [])
        if not isinstance(data, list):
            raise ValueError(f"Seed manifest {path} must be a list of objects.")
        records = [rec for rec in data if isinstance(rec, dict)]
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _spec_from_record(record: Dict[str, Any]) -> Optional[SeedSpec]:
    seed_text = record.get("seed") or record.get("text")
    if not seed_text:
        return None
    scheme = record.get("scheme") or record.get("rhyme_scheme")
    tags = _parse_seed_tags(record.get("tags"))
    persona = record.get("persona")
    theme = record.get("theme") or record.get("theme_hint")
    style = record.get("style") or record.get("style_hint")
    topic = record.get("topic") or record.get("topic_hint")
    vocab = record.get("vocab") or record.get("vocab_hint")
    seed_id = record.get("id") or record.get("name")
    samples = _maybe_int(record.get("samples") or record.get("shots"))
    target_syllables = _maybe_int(record.get("target_syllables"))
    meter_sigma = _maybe_float(record.get("meter_sigma"))
    verse_accept_threshold = _maybe_float(record.get("verse_accept_threshold"))
    syllable_map = _parse_syllable_map(record.get("syllable_map"))
    return SeedSpec(
        text=str(seed_text),
        scheme=str(scheme).upper() if scheme else None,
        tags=tags,
        persona=str(persona) if persona else None,
        theme=str(theme) if theme else None,
        style=str(style) if style else None,
        topic=str(topic) if topic else None,
        vocab=str(vocab) if vocab else None,
        seed_id=str(seed_id) if seed_id else None,
        samples=samples,
        target_syllables=target_syllables,
        meter_sigma=meter_sigma,
        verse_accept_threshold=verse_accept_threshold,
        syllable_map=syllable_map,
    )


def load_seeds(args, stage3_cfg) -> List[SeedSpec]:
    seeds: List[SeedSpec] = []
    for seed_text in args.seeds or []:
        seed_text = seed_text.strip()
        if seed_text:
            seeds.append(SeedSpec(text=seed_text))

    if args.seeds_file:
        path = Path(args.seeds_file)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                seed = line.strip()
                if seed:
                    seeds.append(SeedSpec(text=seed))

    manifest_path: Optional[Path] = None
    manifest_cfg = stage3_cfg.get("seed_manifest")
    if args.seed_manifest:
        manifest_path = Path(args.seed_manifest)
    elif manifest_cfg:
        manifest_path = Path(manifest_cfg)

    if manifest_path and not manifest_path.is_absolute():
        manifest_path = (ROOT / manifest_path).resolve()

    if manifest_path:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Seed manifest not found: {manifest_path}")
        records = _load_manifest_records(manifest_path)
        manifest_specs = []
        for record in records:
            spec = _spec_from_record(record)
            if spec:
                manifest_specs.append(spec)
        seeds.extend(manifest_specs)
        print(f"[PIPELINE] Loaded {len(manifest_specs)} structured seeds from {manifest_path}")

    return seeds


def main():
    args = parse_args()
    settings = load_settings(args.config)
    gen_cfg = settings.generation_defaults
    stage3_cfg = settings.stage3

    refresh_rhymes = stage3_cfg.get("refresh_rhymes", True) and not args.skip_update_rhymes
    if refresh_rhymes and not args.force_refresh_rhymes:
        if not rhyme_csv_is_stale(settings.elite_corpus_path, settings.rhyme_groups_csv):
            refresh_rhymes = False
            print("[PIPELINE] Rhyme CSV already up to date; skipping refresh.")

    if refresh_rhymes:
        print("[PIPELINE] Refreshing rhyme CSV...")
        expand_rhyme_groups(
            corpus_path=str(settings.elite_corpus_path),
            existing_csv=str(settings.rhyme_groups_csv),
            output_csv=str(settings.rhyme_groups_csv),
            siamese_model_dir=str(settings.siamese_model_dir),
        )
    else:
        print("[PIPELINE] Skipping rhyme CSV refresh.")

    seeds = load_seeds(args, stage3_cfg)
    if not seeds:
        raise SystemExit("No seeds provided. Use --seeds/--seeds_file or configure stage3.seed_manifest.")

    python_bin = sys.executable
    scheme = (args.scheme or gen_cfg.get("scheme", "AABB")).upper()
    num_bars = args.num_bars or int(gen_cfg.get("num_bars", 16))
    candidates = args.candidates or int(gen_cfg.get("candidates", 8))
    max_new_tokens = args.max_new_tokens or int(gen_cfg.get("max_new_tokens", 32))
    temperature = args.temperature if args.temperature is not None else float(gen_cfg.get("temperature", 0.8))
    top_p = args.top_p if args.top_p is not None else float(gen_cfg.get("top_p", 0.9))
    repetition_penalty = (
        args.repetition_penalty if args.repetition_penalty is not None else float(gen_cfg.get("repetition_penalty", 1.05))
    )
    attempts = args.attempts or int(gen_cfg.get("attempts", 4))
    verse_accept_threshold = args.verse_accept_threshold or float(gen_cfg.get("verse_accept_threshold", 0.3))
    samples_per_seed = args.samples_per_seed or int(stage3_cfg.get("samples_per_seed", 1))
    parallel_workers = args.parallel_workers or int(stage3_cfg.get("parallel_workers", 1))
    samples_per_seed = max(1, samples_per_seed)
    parallel_workers = max(1, parallel_workers)
    log_path = args.log_json or gen_cfg.get("log_json") or str(settings.generation_log_path)
    log_path = str(log_path)
    persona_default = args.persona or gen_cfg.get("persona")
    theme_default = args.theme_hint or gen_cfg.get("theme_hint")
    style_default = args.style_hint or gen_cfg.get("style_hint")
    topic_default = args.topic_hint or gen_cfg.get("topic_hint")
    vocab_default = args.vocab_hint or gen_cfg.get("vocab_hint")
    syllable_map_default = args.syllable_map or gen_cfg.get("syllable_map")

    target_syllables_default = int(gen_cfg.get("target_syllables", 13))
    meter_sigma_default = float(gen_cfg.get("meter_sigma", 2.0))
    anchor_candidates_default = int(gen_cfg.get("anchor_candidates_per_line", 6))

    if not args.skip_generation:
        commands: List[List[str]] = []
        jobs: List[Dict[str, Any]] = []
        for seed_spec in seeds:
            seed_scheme = (seed_spec.scheme or scheme).upper()
            per_seed_samples = seed_spec.sample_count(samples_per_seed)
            for _ in range(per_seed_samples):
                cmd = [
                    python_bin,
                    "scripts/generation/generate_rhymed_verse.py",
                    "--artist",
                    args.artist,
                    "--scheme",
                    seed_scheme,
                    "--num_bars",
                    str(num_bars),
                    "--candidates",
                    str(candidates),
                    "--max_new_tokens",
                    str(max_new_tokens),
                    "--temperature",
                    str(temperature),
                    "--top_p",
                    str(top_p),
                    "--repetition_penalty",
                    str(repetition_penalty),
                    "--attempts",
                    str(attempts),
                ]
                cmd.extend(seed_spec.cli_args())
                seed_spec.apply_generation_overrides(cmd)
                if not seed_spec.persona and persona_default:
                    cmd.extend(["--persona", persona_default])
                if not seed_spec.theme and theme_default:
                    cmd.extend(["--theme_hint", theme_default])
                if not seed_spec.style and style_default:
                    cmd.extend(["--style_hint", style_default])
                if not seed_spec.topic and topic_default:
                    cmd.extend(["--topic_hint", topic_default])
                if not seed_spec.vocab and vocab_default:
                    cmd.extend(["--vocab_hint", vocab_default])
                if not seed_spec.syllable_map and syllable_map_default:
                    cmd.extend(["--syllable_map", syllable_map_default])
                if seed_spec.verse_accept_threshold is None:
                    cmd.extend(["--verse_accept_threshold", str(verse_accept_threshold)])
                if log_path:
                    cmd.extend(["--log_json", log_path])
                if args.config:
                    cmd.extend(["--config", args.config])
                commands.append(cmd)
                job_kwargs: Dict[str, Any] = {
                    "artist": args.artist,
                    "section": "VERSE",
                    "seed": seed_spec.text,
                    "seed_id": seed_spec.seed_id,
                    "seed_tags": ",".join(seed_spec.tags) if seed_spec.tags else None,
                    "scheme": seed_scheme,
                    "num_bars": num_bars,
                    "candidates": candidates,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "attempts": attempts,
                    "verse_accept_threshold": (
                        seed_spec.verse_accept_threshold if seed_spec.verse_accept_threshold is not None else verse_accept_threshold
                    ),
                    "log_json": log_path,
                    "log_dir": None,
                    "config": args.config,
                    "hybrid": True,
                    "topic_model_path": None,
                    "ngram_path": None,
                    "target_syllables": (
                        seed_spec.target_syllables if seed_spec.target_syllables is not None else target_syllables_default
                    ),
                    "meter_sigma": seed_spec.meter_sigma if seed_spec.meter_sigma is not None else meter_sigma_default,
                    "anchor_candidates_per_line": anchor_candidates_default,
                    "persona": seed_spec.persona or persona_default,
                    "theme_hint": seed_spec.theme or theme_default,
                    "style_hint": seed_spec.style or style_default,
                    "topic_hint": seed_spec.topic or topic_default,
                    "vocab_hint": seed_spec.vocab or vocab_default,
                    "syllable_map": (
                        ",".join(str(v) for v in seed_spec.syllable_map)
                        if seed_spec.syllable_map
                        else syllable_map_default
                    ),
                }
                jobs.append(job_kwargs)
        if commands:
            print(
                f"[PIPELINE] Generating {len(commands)} samples "
                f"(~{samples_per_seed} per seed across {len(seeds)} entries)."
            )
            if args.subprocess_generation:
                run_parallel(commands, parallel_workers)
            else:
                if parallel_workers > 1:
                    print("[PIPELINE][WARN] parallel_workers>1 ignored for in-process generation (running sequentially).")
                run_inprocess_generation(jobs, settings)

    head_path = args.local_critic_head or str(settings.local_critic_dir / "reward_head.pt")
    calibration_path = args.local_critic_calibration
    if not calibration_path:
        default_calibration = settings.local_critic_dir / "calibration.json"
        if default_calibration.exists():
            calibration_path = str(default_calibration)

    if args.train_local_critic:
        cmd = [python_bin, "scripts/training/train_local_critic.py"]
        if args.config:
            cmd.extend(["--config", args.config])
        cmd.extend(["--output_dir", str(settings.local_critic_dir)])
        run_cmd(cmd)

    if not args.skip_consolidate:
        cmd = [python_bin, "scripts/pipeline/consolidate_stage3.py"]
        if args.config:
            cmd.extend(["--config", args.config])
        if args.critic_scores:
            cmd.extend(["--critic_scores", args.critic_scores])
        if args.local_critic_head:
            cmd.extend(
                [
                    "--local_critic_head",
                    args.local_critic_head,
                    "--local_critic_siamese",
                    str(settings.siamese_model_dir),
                ]
            )
            if calibration_path:
                cmd.extend(["--local_critic_calibration", calibration_path])
        elif args.train_local_critic or Path(head_path).exists():
            cmd.extend(
                [
                    "--local_critic_head",
                    head_path,
                    "--local_critic_siamese",
                    str(settings.siamese_model_dir),
                ]
            )
            if calibration_path:
                cmd.extend(["--local_critic_calibration", calibration_path])
        else:
            print("[PIPELINE] No local critic head found; consolidation will require --critic_scores.")
        run_cmd(cmd)

    if args.train_lora_refresh:
        run_label = sanitize_run_name(args.lora_run_name or datetime.utcnow().strftime("refresh_%Y%m%d-%H%M%S"))
        runs_root = Path(settings.lora_output_dir) / "runs"
        run_dir = runs_root / run_label
        adapter_run_dir = run_dir / "adapter"
        tokenizer_run_dir = run_dir / "tokenizer"
        adapter_run_dir.mkdir(parents=True, exist_ok=False)
        tokenizer_run_dir.mkdir(parents=True, exist_ok=False)
        metadata_path = run_dir / "metadata.json"
        lora_cmd = [
            python_bin,
            "scripts/training/train_elite_qwen.py",
            "--text_path",
            str(settings.weighted_corpus_path),
            "--output_dir",
            str(adapter_run_dir),
            "--tokenizer_save_dir",
            str(tokenizer_run_dir),
            "--metadata_out",
            str(metadata_path),
        ]
        if args.config:
            lora_cmd.extend(["--config", args.config])
        run_cmd(lora_cmd)
        registry_path = Path(settings.lora_output_dir) / "registry.jsonl"
        append_registry(
            registry_path,
            {
                "run_name": run_label,
                "timestamp": datetime.utcnow().isoformat(),
                "adapter_path": str(adapter_run_dir),
                "tokenizer_path": str(tokenizer_run_dir),
                "metadata_path": str(metadata_path),
                "weighted_corpus": str(settings.weighted_corpus_path),
            },
        )
        if args.promote_lora:
            deploy_artifact(adapter_run_dir, Path(settings.adapter_dir))
            deploy_artifact(tokenizer_run_dir, Path(settings.tokenizer_dir))

    print("[PIPELINE] Done.")


if __name__ == "__main__":
    main()
