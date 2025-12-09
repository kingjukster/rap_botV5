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

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from config.settings import load_settings
from update_rhyme_groups import expand_rhyme_groups


def run_cmd(cmd: List[str]):
    print(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Stage-3 generation + consolidation pipeline.")
    parser.add_argument("--artist", type=str, default="mf_doom")
    parser.add_argument("--scheme", type=str, default="AABB")
    parser.add_argument("--num_bars", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--seeds", nargs="*", default=[], help="List of seed strings.")
    parser.add_argument("--seeds_file", type=str, default=None, help="Optional file with one seed per line.")
    parser.add_argument("--samples_per_seed", type=int, default=4, help="How many verse generations per seed.")
    parser.add_argument("--log_json", type=str, default=None, help="Override log file path for generate_rhymed_verse.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config path.")
    parser.add_argument("--critic_scores", type=str, default=None, help="Critic scores JSONL (optional).")
    parser.add_argument("--local_critic_head", type=str, default=None, help="Path to trained reward head (optional).")
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
    return parser.parse_args()


def load_seeds(args) -> List[str]:
    seeds = list(args.seeds or [])
    if args.seeds_file:
        with open(args.seeds_file, "r", encoding="utf-8") as f:
            for line in f:
                seed = line.strip()
                if seed:
                    seeds.append(seed)
    return seeds


def main():
    args = parse_args()
    settings = load_settings(args.config)

    if not args.skip_update_rhymes:
        print("[PIPELINE] Refreshing rhyme CSV...")
        expand_rhyme_groups(
            corpus_path=str(settings.elite_corpus_path),
            existing_csv=str(settings.rhyme_groups_csv),
            output_csv=str(settings.rhyme_groups_csv),
            siamese_model_dir=str(settings.siamese_model_dir),
        )

    seeds = load_seeds(args)
    if not seeds:
        raise SystemExit("No seeds provided. Use --seeds or --seeds_file.")

    python_bin = sys.executable

    if not args.skip_generation:
        for seed in seeds:
            for sample_idx in range(args.samples_per_seed):
                cmd = [
                    python_bin,
                    "generate_rhymed_verse.py",
                    "--artist",
                    args.artist,
                    "--seed",
                    seed,
                    "--scheme",
                    args.scheme,
                    "--num_bars",
                    str(args.num_bars),
                    "--candidates",
                    str(args.candidates),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--temperature",
                    str(args.temperature),
                    "--top_p",
                    str(args.top_p),
                    "--repetition_penalty",
                    str(args.repetition_penalty),
                    "--attempts",
                    str(args.attempts),
                ]
                if args.log_json:
                    cmd.extend(["--log_json", args.log_json])
                if args.config:
                    cmd.extend(["--config", args.config])
                run_cmd(cmd)

    head_path = args.local_critic_head or str(settings.local_critic_dir / "reward_head.pt")

    if args.train_local_critic:
        cmd = [python_bin, "train_local_critic.py"]
        if args.config:
            cmd.extend(["--config", args.config])
        cmd.extend(["--output_dir", str(settings.local_critic_dir)])
        run_cmd(cmd)

    if not args.skip_consolidate:
        cmd = [python_bin, "consolidate_stage3.py"]
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
        elif args.train_local_critic or Path(head_path).exists():
            cmd.extend(
                [
                    "--local_critic_head",
                    head_path,
                    "--local_critic_siamese",
                    str(settings.siamese_model_dir),
                ]
            )
        else:
            print("[PIPELINE] No local critic head found; consolidation will require --critic_scores.")
        run_cmd(cmd)

    print("[PIPELINE] Done.")


if __name__ == "__main__":
    main()
