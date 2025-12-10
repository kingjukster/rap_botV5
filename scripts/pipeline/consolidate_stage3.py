#!/usr/bin/env python
"""
consolidate_stage3.py

Combine generation logs + critic scores into:
  1) scored_dataset.jsonl (per-verse metadata + critic record)
  2) weighted_corpus_stage3.txt (duplicate verses proportional to critic score)

Usage:
    python scripts/pipeline/consolidate_stage3.py \
        --critic_scores data/critic_scores.jsonl \
        --log_path data/generated_raw.jsonl \
        --scored_output data/scored_dataset.jsonl \
        --weighted_output data/weighted_corpus_stage3.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Dict, Iterable, List

from config.settings import load_settings

SCORE_FIELDS = ["overall_score", "depth_score", "coherence_score", "originality_score"]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge generation logs with critic scores.")
    parser.add_argument(
        "--critic_scores",
        type=str,
        default=None,
        help="JSONL file containing critic outputs (must include verse_id field). Optional when using --local_critic_head.",
    )
    parser.add_argument(
        "--log_path",
        action="append",
        default=None,
        help="JSONL generation logs (can be passed multiple times). Defaults to config log path.",
    )
    parser.add_argument(
        "--scored_output",
        type=str,
        default=None,
        help="Output JSONL path for combined dataset (default from config).",
    )
    parser.add_argument(
        "--weighted_output",
        type=str,
        default=None,
        help="Output TXT path for weighted corpus (default from config).",
    )
    parser.add_argument(
        "--local_critic_head",
        type=str,
        default=None,
        help="Path to trained local critic head (reward_head.pt). When provided, missing critic scores are predicted offline.",
    )
    parser.add_argument(
        "--local_critic_siamese",
        type=str,
        default=None,
        help="Siamese model dir for the local critic (defaults to config).",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum critic score for normalization.",
    )
    parser.add_argument(
        "--max_score",
        type=float,
        default=5.0,
        help="Maximum critic score for normalization.",
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=50.0,
        help="Weight scaling factor (higher => more repetitions).",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=2.4,
        help="Exponent applied to normalized score before scaling.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON/YAML config to resolve default paths.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_logs(paths: List[Path]) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    for path in paths:
        if not path.exists():
            print(f"[WARN] Log file not found: {path}")
            continue
        for record in read_jsonl(path):
            verse_id = record.get("verse_id")
            if verse_id:
                records[verse_id] = record
    return records


def normalize(value: float, min_score: float, max_score: float) -> float:
    denom = max_score - min_score
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (value - min_score) / denom))


def verse_block(text: str) -> str:
    text = text.strip()
    if "<END_SONG>" not in text:
        text = text + "\n<END_SONG>"
    if not text.endswith("\n"):
        text += "\n"
    return text


def main():
    args = parse_args()
    settings = load_settings(args.config)

    log_paths = args.log_path or [str(settings.generation_log_path)]
    log_paths = [Path(p) for p in log_paths]
    critic_path = Path(args.critic_scores) if args.critic_scores else None
    scored_output = Path(args.scored_output or settings.scored_dataset_path)
    weighted_output = Path(args.weighted_output or settings.weighted_corpus_path)

    logs = load_logs(log_paths)
    print(f"[INFO] Loaded {len(logs):,} generation log entries from {len(log_paths)} file(s).")

    scored_output.parent.mkdir(parents=True, exist_ok=True)
    weighted_output.parent.mkdir(parents=True, exist_ok=True)

    critic_lookup: Dict[str, Dict] = {}
    if critic_path:
        if not critic_path.exists():
            raise FileNotFoundError(f"Critic scores file not found: {critic_path}")
        for critic_record in read_jsonl(critic_path):
            verse_id = critic_record.get("verse_id")
            if verse_id:
                critic_lookup[verse_id] = critic_record
        print(f"[INFO] Loaded {len(critic_lookup):,} critic entries from {critic_path}")

    local_critic = None
    if args.local_critic_head:
        from scoring import OfflineCritic  # lazy import to avoid circular issues

        siamese_dir = args.local_critic_siamese or str(settings.siamese_model_dir)
        local_critic = OfflineCritic(args.local_critic_head, siamese_dir)
        print("[INFO] Local critic ready for offline scoring.")

    if not critic_lookup and local_critic is None:
        raise RuntimeError("No critic scores provided and local critic head not specified.")

    scored_count = 0
    total_weight = 0

    with open(scored_output, "w", encoding="utf-8") as scored_f, open(
        weighted_output, "w", encoding="utf-8"
    ) as weighted_f:
        for verse_id, log_entry in logs.items():
            critic_record = critic_lookup.get(verse_id)
            if critic_record is None and local_critic is not None:
                predictions = local_critic.score(log_entry.get("verse_text", ""))
                critic_record = {"verse_id": verse_id, "source": "local", **predictions}

            if critic_record is None:
                continue

            combined = {
                "verse_id": verse_id,
                "artist": log_entry.get("artist"),
                "seed": log_entry.get("seed"),
                "scheme": log_entry.get("scheme"),
                "verse_score": log_entry.get("verse_score"),
                "accepted": log_entry.get("accepted"),
                "bars": log_entry.get("bars"),
                "bar_metrics": log_entry.get("bar_metrics"),
                "verse_text": log_entry.get("verse_text"),
                "generation": {
                    "timestamp": log_entry.get("timestamp"),
                    "generation_params": log_entry.get("generation_params"),
                    "settings_snapshot": log_entry.get("settings_snapshot"),
                },
                "critic": critic_record,
            }
            scored_f.write(json.dumps(combined, ensure_ascii=False) + "\n")
            scored_count += 1

            scores = [
                normalize(float(critic_record.get(field, args.min_score)), args.min_score, args.max_score)
                for field in SCORE_FIELDS
                if critic_record.get(field) is not None
            ]
            if not scores:
                continue
            score_norm = sum(scores) / len(scores)
            weight = max(1, int(round((score_norm ** args.power) * args.scaling_factor)))
            verse_text = log_entry.get("verse_text") or "\n".join(
                (bar.get("text", "") for bar in log_entry.get("bars", []))
            )
            block = verse_block(verse_text)
            for _ in range(weight):
                weighted_f.write(block)
            total_weight += weight

    print(f"[DONE] Wrote {scored_count:,} records to {scored_output}")
    print(f"[DONE] Weighted corpus total verses written: {total_weight:,} (file: {weighted_output})")


if __name__ == "__main__":
    main()
