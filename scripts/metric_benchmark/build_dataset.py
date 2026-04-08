#!/usr/bin/env python3
"""
Build metric benchmark verse JSONL from Kaggle songs.csv (or similar) + optional synthetic.

Example:
  python scripts/metric_benchmark/build_dataset.py \\
    --csv data/songs.csv --out data/metric_benchmark/master_verses.jsonl \\
    --max-rows 500 --synthetic-count 20 --seed 42

Lyrics column is auto-detected: lyrics, Lyrics, lyric, text, or first column with
multiline-looking strings.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_standalone_module(rel: str):
    """Load metric_benchmark submodule without importing evo_rhyme package __init__ (avoids torch/numpy)."""
    path = ROOT / rel
    name = "_mb_" + rel.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_mb_protocol = _load_standalone_module("evo_rhyme/metric_benchmark/protocol.py")
_mb_segmentation = _load_standalone_module("evo_rhyme/metric_benchmark/segmentation.py")
_mb_synthetic = _load_standalone_module("evo_rhyme/metric_benchmark/synthetic.py")

default_protocol_manifest = _mb_protocol.default_protocol_manifest
save_protocol_manifest = _mb_protocol.save_protocol_manifest
segment_song_to_verses = _mb_segmentation.segment_song_to_verses
generate_synthetic_verses = _mb_synthetic.generate_synthetic_verses

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _detect_lyrics_column(fieldnames: List[str]) -> Optional[str]:
    if not fieldnames:
        return None
    lower = {f.lower().strip(): f for f in fieldnames}
    for candidate in ("lyrics", "lyric", "text", "lyrics_clean"):
        if candidate in lower:
            return lower[candidate]
    return fieldnames[0]


def build_records_from_csv(
    path: Path,
    *,
    max_rows: int,
    min_lines: int,
    max_bars: int,
    stride: Optional[int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    stats_agg = {
        "csv_rows_read": 0,
        "verses_from_corpus": 0,
        "rows_skipped_empty_lyrics": 0,
        "segmentation_merged": 0,
        "segmentation_dropped": 0,
        "sliding_windows": 0,
        "raw_stanzas": 0,
    }

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row")
        fieldnames = list(reader.fieldnames)
        col = _detect_lyrics_column(fieldnames)
        if not col:
            raise SystemExit("Could not detect lyrics column in CSV")

        for row in reader:
            if stats_agg["csv_rows_read"] >= max_rows:
                break
            stats_agg["csv_rows_read"] += 1
            raw = row.get(col) if isinstance(row, dict) else None
            if raw is None or not str(raw).strip():
                stats_agg["rows_skipped_empty_lyrics"] += 1
                continue
            verses, seg_stats = segment_song_to_verses(
                str(raw),
                min_lines=min_lines,
                max_bars=max_bars,
                stride=stride,
            )
            stats_agg["segmentation_merged"] += seg_stats.merged_short_stanzas
            stats_agg["segmentation_dropped"] += seg_stats.dropped_incomplete
            stats_agg["sliding_windows"] += seg_stats.sliding_windows
            stats_agg["raw_stanzas"] += seg_stats.raw_stanzas
            row_id = row.get("track_id") or row.get("id") or row.get("song_id") or stats_agg["csv_rows_read"]
            for vi, (lines, harvest_extra) in enumerate(verses):
                rid = str(uuid.uuid4())
                harvest: Dict[str, Any] = {
                    "kaggle_row": str(row_id),
                    "verse_index": vi,
                    **harvest_extra,
                }
                ms = (row.get("metric_benchmark_stratum") or "").strip()
                if ms:
                    harvest["metric_benchmark_stratum"] = ms
                records.append(
                    {
                        "id": rid,
                        "lyrics": lines,
                        "n_bars": len(lines),
                        "source": "corpus",
                        "harvest": harvest,
                        "unlabeled": True,
                    }
                )
                stats_agg["verses_from_corpus"] += 1

    return records, stats_agg


def load_evo_top_candidates(path: Path) -> List[Dict[str, Any]]:
    """Load lines from top_candidates.json (gen -> list) or a flat list of candidates."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for _gen, cands in data.items():
            if not isinstance(cands, list):
                continue
            for c in cands:
                if not isinstance(c, dict):
                    continue
                ls = c.get("lines")
                if isinstance(ls, list) and len(ls) >= 2:
                    rid = str(uuid.uuid4())
                    records.append(
                        {
                            "id": rid,
                            "lyrics": [str(x) for x in ls],
                            "n_bars": len(ls),
                            "source": "model",
                            "harvest": {"evo_path": str(path), "generator": "top_candidates"},
                            "unlabeled": True,
                        }
                    )
    elif isinstance(data, list):
        for c in data:
            if not isinstance(c, dict):
                continue
            ls = c.get("lines")
            if isinstance(ls, list) and len(ls) >= 2:
                rid = str(uuid.uuid4())
                records.append(
                    {
                        "id": rid,
                        "lyrics": [str(x) for x in ls],
                        "n_bars": len(ls),
                        "source": "model",
                        "harvest": {"evo_path": str(path), "generator": "top_candidates"},
                        "unlabeled": True,
                    }
                )
    return records


def main() -> int:
    p = argparse.ArgumentParser(description="Build metric benchmark verse JSONL.")
    p.add_argument("--csv", type=Path, help="Path to songs.csv (Kaggle)")
    p.add_argument("--out", type=Path, default=ROOT / "data/metric_benchmark/master_verses.jsonl")
    p.add_argument("--protocol-out", type=Path, default=ROOT / "data/metric_benchmark/protocol_defaults.json")
    p.add_argument("--max-rows", type=int, default=200, help="Max CSV rows to read")
    p.add_argument("--min-lines", type=int, default=2)
    p.add_argument("--max-bars", type=int, default=16)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--synthetic-count", type=int, default=0)
    p.add_argument("--evo-json", type=Path, default=None, help="top_candidates.json from a run")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_records: List[Dict[str, Any]] = []
    build_stats: Dict[str, Any] = {"seed": args.seed}

    if args.csv and args.csv.exists():
        recs, st = build_records_from_csv(
            args.csv,
            max_rows=args.max_rows,
            min_lines=args.min_lines,
            max_bars=args.max_bars,
            stride=args.stride,
            seed=args.seed,
        )
        all_records.extend(recs)
        build_stats["csv"] = st
    elif args.csv:
        logger.warning("CSV path does not exist: %s — corpus records skipped", args.csv)

    if args.synthetic_count > 0:
        syn = generate_synthetic_verses(args.synthetic_count, args.seed + 999)
        all_records.extend(syn)
        build_stats["synthetic"] = len(syn)

    if args.evo_json and args.evo_json.exists():
        evo_recs = load_evo_top_candidates(args.evo_json)
        all_records.extend(evo_recs)
        build_stats["evo_candidates"] = len(evo_recs)
    elif args.evo_json:
        logger.warning("evo-json not found: %s", args.evo_json)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_protocol_manifest(args.protocol_out, default_protocol_manifest())
    build_stats["total_verses_written"] = len(all_records)
    build_stats["output"] = str(args.out)
    logger.info("Wrote %d verses to %s", len(all_records), args.out)
    with open(args.out.with_suffix(".build_stats.json"), "w", encoding="utf-8") as sf:
        json.dump(build_stats, sf, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
