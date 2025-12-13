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
import csv
import json
import hashlib
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config.settings import load_settings

SCORE_FIELDS = ["overall_score", "depth_score", "coherence_score", "originality_score"]
WORD_RE = re.compile(r"[A-Za-z']+")


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
        help="Path (or comma-separated paths) to trained local critic head(s). When provided, missing critic scores are predicted offline.",
    )
    parser.add_argument(
        "--local_critic_siamese",
        type=str,
        default=None,
        help="Siamese model dir for the local critic (defaults to config).",
    )
    parser.add_argument(
        "--local_critic_calibration",
        type=str,
        default=None,
        help="Optional JSON file with calibration params for local critic predictions.",
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
        "--min_overall",
        type=float,
        default=None,
        help="Drop verses whose overall_score is below this threshold.",
    )
    parser.add_argument(
        "--min_average",
        type=float,
        default=None,
        help="Drop verses whose average critic score is below this threshold.",
    )
    parser.add_argument(
        "--stats_output",
        type=str,
        default=None,
        help="Optional JSON path for summary stats (defaults to config.stats.summary_path).",
    )
    parser.add_argument(
        "--per_seed_csv",
        type=str,
        default=None,
        help="Optional CSV path with per-seed metrics (defaults to config.stats.per_seed_csv).",
    )
    parser.add_argument(
        "--hist_output",
        type=str,
        default=None,
        help="Optional JSON path for histogram data (defaults to config.stats.histogram_path).",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=None,
        help="Number of bins to use for score histograms.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON/YAML config to resolve default paths.",
    )
    parser.add_argument(
        "--dedupe_mode",
        choices=["none", "exact", "fingerprint"],
        default="fingerprint",
        help="Duplicate rejection mode for verses (exact text hash or rhyme-ending fingerprint).",
    )
    parser.add_argument(
        "--min_unique_endings",
        type=int,
        default=0,
        help="Minimum distinct normalized end words required per verse (0 disables).",
    )
    parser.add_argument(
        "--max_repeat_per_ending",
        type=int,
        default=0,
        help="Reject verses where any normalized end word occurs more than this many times (0 disables).",
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


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", str(word).lower())


def canonical_verse_text(text: str) -> str:
    normalized_lines = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line or raw_line.startswith("<"):
            continue
        normalized = re.sub(r"[^a-z0-9 ]+", "", raw_line.lower())
        if normalized:
            normalized_lines.append(normalized)
    return "\n".join(normalized_lines)


def extract_endings(bars: List[Dict]) -> Tuple[List[str], Counter]:
    endings: List[str] = []
    freq = Counter()
    for bar in bars or []:
        text = bar.get("text") if isinstance(bar, dict) else None
        if not text:
            continue
        tokens = WORD_RE.findall(text.lower())
        if not tokens:
            continue
        normalized = normalize_word(tokens[-1])
        if normalized:
            endings.append(normalized)
            freq[normalized] += 1
    return endings, freq


def describe(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def histogram(values: List[float], bins: int, value_range: tuple[float, float] | None = None) -> Dict[str, List[float]]:
    if not values or bins <= 0:
        return {"counts": [], "edges": []}
    hist, edges = np.histogram(values, bins=bins, range=value_range)
    return {"counts": hist.astype(int).tolist(), "edges": edges.tolist()}


def main():
    args = parse_args()
    settings = load_settings(args.config)
    stats_cfg = settings.stats
    stage3_cfg = settings.stage3

    log_paths = args.log_path or [str(settings.generation_log_path)]
    log_paths = [Path(p) for p in log_paths]
    critic_path = Path(args.critic_scores) if args.critic_scores else None
    scored_output = Path(args.scored_output or settings.scored_dataset_path)
    weighted_output = Path(args.weighted_output or settings.weighted_corpus_path)
    stats_output = Path(args.stats_output or stats_cfg.get("summary_path"))
    per_seed_csv_path = args.per_seed_csv or stats_cfg.get("per_seed_csv")
    hist_output_path = args.hist_output or stats_cfg.get("histogram_path")
    per_seed_csv = Path(per_seed_csv_path) if per_seed_csv_path else None
    hist_output = Path(hist_output_path) if hist_output_path else None
    hist_bins = args.hist_bins or int(stage3_cfg.get("hist_bins", 20))
    min_overall = args.min_overall
    if min_overall is None:
        min_overall = float(stage3_cfg.get("min_overall_score", 0.0))
    min_average = args.min_average
    if min_average is None:
        min_average = float(stage3_cfg.get("min_average_score", 0.0))

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
        from rapbot.scoring import OfflineCritic  # lazy import to avoid circular issues

        siamese_dir = args.local_critic_siamese or str(settings.siamese_model_dir)
        calibration_path = args.local_critic_calibration
        local_critic = OfflineCritic(args.local_critic_head, siamese_dir, calibration_path=calibration_path)
        print("[INFO] Local critic ready for offline scoring.")
    if not critic_lookup and local_critic is None:
        raise RuntimeError("No critic scores provided and local critic head not specified.")

    scored_count = 0
    total_weight = 0
    rejection_counts = {
        "missing_critic": 0,
        "below_overall": 0,
        "below_average": 0,
        "duplicate": 0,
        "low_unique_endings": 0,
        "ending_repeat": 0,
    }
    score_tracker = {field: [] for field in SCORE_FIELDS}
    verse_scores: List[float] = []
    bar_counts: List[int] = []
    timestamps: List[str] = []
    per_seed = defaultdict(lambda: {"count": 0, "overall": [], "accepted": 0, "verse_scores": []})
    per_scheme = defaultdict(lambda: {"count": 0, "overall": []})
    critic_sources = Counter()
    tag_counter = Counter()
    seen_fingerprints: set[str] = set()

    with open(scored_output, "w", encoding="utf-8") as scored_f, open(
        weighted_output, "w", encoding="utf-8"
    ) as weighted_f:
        for verse_id, log_entry in logs.items():
            critic_record = critic_lookup.get(verse_id)
            if critic_record is None and local_critic is not None:
                predictions = local_critic.score(log_entry.get("verse_text", ""))
                critic_record = {"verse_id": verse_id, "source": "local", **predictions}

            if critic_record is None:
                rejection_counts["missing_critic"] += 1
                continue

            critic_sources[critic_record.get("source", "remote")] += 1

            overall_score = critic_record.get("overall_score")
            if overall_score is None:
                rejection_counts["missing_critic"] += 1
                continue
            field_values = [float(critic_record.get(field, 0.0)) for field in SCORE_FIELDS if critic_record.get(field) is not None]
            avg_score = sum(field_values) / len(field_values) if field_values else None
            if min_overall and overall_score < min_overall:
                rejection_counts["below_overall"] += 1
                continue
            if min_average and avg_score is not None and avg_score < min_average:
                rejection_counts["below_average"] += 1
                continue

            verse_text = log_entry.get("verse_text")
            if not verse_text:
                bars_for_text = log_entry.get("bars", [])
                verse_text = "\n".join((bar.get("text", "") for bar in bars_for_text))
            bars = log_entry.get("bars") or []
            endings, ending_freq = extract_endings(bars)
            unique_endings = len(set(endings))
            if args.min_unique_endings and unique_endings < args.min_unique_endings:
                rejection_counts["low_unique_endings"] += 1
                continue
            if args.max_repeat_per_ending and ending_freq:
                most_common = ending_freq.most_common(1)[0][1]
                if most_common > args.max_repeat_per_ending:
                    rejection_counts["ending_repeat"] += 1
                    continue

            if args.dedupe_mode != "none":
                if args.dedupe_mode == "exact":
                    fingerprint_source = canonical_verse_text(verse_text)
                else:
                    fingerprint_source = "|".join(sorted(endings)) if endings else canonical_verse_text(verse_text)
                fingerprint = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()
                if fingerprint in seen_fingerprints:
                    rejection_counts["duplicate"] += 1
                    continue
                seen_fingerprints.add(fingerprint)

            combined = {
                "verse_id": verse_id,
                "artist": log_entry.get("artist"),
                "seed": log_entry.get("seed"),
                "seed_id": log_entry.get("seed_id"),
                "seed_tags": log_entry.get("seed_tags"),
                "persona": log_entry.get("persona"),
                "theme_hint": log_entry.get("theme_hint"),
                "style_hint": log_entry.get("style_hint"),
                "topic_hint": log_entry.get("topic_hint"),
                "vocab_hint": log_entry.get("vocab_hint"),
                "scheme": log_entry.get("scheme"),
                "verse_score": log_entry.get("verse_score"),
                "accepted": log_entry.get("accepted"),
                "bars": bars,
                "bar_metrics": log_entry.get("bar_metrics"),
                "verse_text": verse_text,
                "generation": {
                    "timestamp": log_entry.get("timestamp"),
                    "generation_params": log_entry.get("generation_params"),
                    "settings_snapshot": log_entry.get("settings_snapshot"),
                },
                "critic": critic_record,
            }
            scored_f.write(json.dumps(combined, ensure_ascii=False) + "\n")
            scored_count += 1

            for field in SCORE_FIELDS:
                value = critic_record.get(field)
                if value is not None:
                    score_tracker[field].append(float(value))

            verse_score = log_entry.get("verse_score")
            if verse_score is not None:
                verse_scores.append(float(verse_score))
            bar_counts.append(len(log_entry.get("bars") or []))
            timestamp = log_entry.get("timestamp")
            if timestamp:
                timestamps.append(timestamp)

            seed_key = log_entry.get("seed") or "<unknown>"
            seed_entry = per_seed[seed_key]
            seed_entry["count"] += 1
            seed_entry["overall"].append(float(overall_score))
            seed_entry["verse_scores"].append(float(verse_score) if verse_score is not None else 0.0)
            if log_entry.get("accepted"):
                seed_entry["accepted"] += 1

            scheme_key = log_entry.get("scheme") or "<unknown>"
            scheme_entry = per_scheme[scheme_key]
            scheme_entry["count"] += 1
            scheme_entry["overall"].append(float(overall_score))

            tags = log_entry.get("seed_tags") or []
            for tag in tags:
                tag_counter[tag] += 1

            scores = [
                normalize(float(critic_record.get(field, args.min_score)), args.min_score, args.max_score)
                for field in SCORE_FIELDS
                if critic_record.get(field) is not None
            ]
            if not scores:
                continue
            score_norm = sum(scores) / len(scores)
            weight = max(1, int(round((score_norm ** args.power) * args.scaling_factor)))
            block = verse_block(verse_text)
            for _ in range(weight):
                weighted_f.write(block)
            total_weight += weight

    print(f"[DONE] Wrote {scored_count:,} records to {scored_output}")
    print(f"[DONE] Weighted corpus total verses written: {total_weight:,} (file: {weighted_output})")

    summary = {
        "generated_entries": len(logs),
        "scored_entries": scored_count,
        "total_weight": total_weight,
        "rejections": rejection_counts,
        "critic_sources": dict(critic_sources),
        "min_overall_applied": min_overall,
        "min_average_applied": min_average,
        "score_summary": {field: describe(values) for field, values in score_tracker.items()},
        "verse_score_summary": describe(verse_scores),
        "bar_count_summary": describe(bar_counts),
        "timestamp_range": {"min": min(timestamps) if timestamps else None, "max": max(timestamps) if timestamps else None},
        "top_seeds": [],
        "bottom_seeds": [],
        "scheme_summary": [],
        "tag_counts": tag_counter.most_common(25),
    }

    seed_rows = []
    for seed, data in per_seed.items():
        avg_overall = sum(data["overall"]) / data["count"] if data["count"] else 0.0
        seed_rows.append(
            {
                "seed": seed,
                "count": data["count"],
                "avg_overall": avg_overall,
                "accept_rate": data["accepted"] / data["count"] if data["count"] else 0.0,
                "avg_verse_score": sum(data["verse_scores"]) / data["count"] if data["count"] else 0.0,
            }
        )

    if seed_rows:
        summary["top_seeds"] = sorted(seed_rows, key=lambda row: row["avg_overall"], reverse=True)[:10]
        summary["bottom_seeds"] = sorted(seed_rows, key=lambda row: row["avg_overall"])[:10]

    scheme_rows = []
    for scheme, data in per_scheme.items():
        avg_overall = sum(data["overall"]) / data["count"] if data["count"] else 0.0
        scheme_rows.append({"scheme": scheme, "count": data["count"], "avg_overall": avg_overall})
    summary["scheme_summary"] = sorted(scheme_rows, key=lambda row: row["avg_overall"], reverse=True)

    summary["histograms"] = {
        field: histogram(values, hist_bins, (args.min_score, args.max_score))
        for field, values in score_tracker.items()
        if values
    }
    if verse_scores:
        summary["histograms"]["verse_score"] = histogram(verse_scores, hist_bins, None)

    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[STATS] Summary written to {stats_output}")

    if per_seed_csv:
        per_seed_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(per_seed_csv, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["seed", "count", "avg_overall", "accept_rate", "avg_verse_score"])
            writer.writeheader()
            for row in sorted(seed_rows, key=lambda r: r["avg_overall"], reverse=True):
                writer.writerow(row)
        print(f"[STATS] Per-seed CSV written to {per_seed_csv}")

    if hist_output:
        hist_output.parent.mkdir(parents=True, exist_ok=True)
        hist_output.write_text(json.dumps(summary.get("histograms", {}), indent=2), encoding="utf-8")
        print(f"[STATS] Histogram data written to {hist_output}")


if __name__ == "__main__":
    main()
