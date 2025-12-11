#!/usr/bin/env python
"""build_phase_corpus.py

Turn the enriched Kaggle corpus JSONL into a filtered text corpus for LoRA phase A.

Each output sample concatenates a fixed number of high-quality bars and appends
"<END_SONG>" to mimic Stage-3 logs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def passes_filters(entry: Dict, min_unique: int, require_dense: bool) -> bool:
    if not entry.get("english_like", False):
        return False
    if entry.get("section_hint") == "hook":
        return False
    if entry.get("unique_rhymes_window", 0) < min_unique:
        return False
    if require_dense and entry.get("dense_bars_window", 0) <= 0:
        return False
    return True


def build_samples(
    enriched_path: Path,
    output_path: Path,
    bars_per_sample: int,
    min_unique: int,
    require_dense: bool,
    max_samples: int | None,
) -> int:
    tracks: Dict[int, List[str]] = defaultdict(list)
    with open(enriched_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not passes_filters(entry, min_unique=min_unique, require_dense=require_dense):
                continue
            text = entry.get("text", "").strip()
            if not text:
                continue
            track_idx = int(entry.get("track_index", -1))
            tracks[track_idx].append(text)

    total_written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for track_idx in sorted(tracks.keys()):
            bars = tracks[track_idx]
            if len(bars) < bars_per_sample:
                continue
            for start in range(0, len(bars), bars_per_sample):
                chunk = bars[start : start + bars_per_sample]
                if len(chunk) < bars_per_sample:
                    break
                verse = " ".join(chunk).strip()
                if not verse:
                    continue
                out_f.write(verse + "\n")
                total_written += 1
                if max_samples and total_written >= max_samples:
                    return total_written
    return total_written


def parse_args():
    parser = argparse.ArgumentParser(description="Build phase-A training corpus from enriched Kaggle JSONL.")
    parser.add_argument("--input", type=str, required=True, help="Path to enriched Kaggle JSONL (from enrich_kaggle_corpus).")
    parser.add_argument("--output", type=str, required=True, help="Destination text file for LoRA training.")
    parser.add_argument("--bars_per_sample", type=int, default=16)
    parser.add_argument("--min_unique_rhymes", type=int, default=2)
    parser.add_argument("--require_dense", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input enriched corpus not found: {input_path}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = build_samples(
        enriched_path=input_path,
        output_path=output_path,
        bars_per_sample=args.bars_per_sample,
        min_unique=args.min_unique_rhymes,
        require_dense=args.require_dense,
        max_samples=args.max_samples,
    )
    print(f"[DONE] Wrote {written:,} samples to {output_path}")


if __name__ == "__main__":
    main()
