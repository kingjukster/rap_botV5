#!/usr/bin/env python3
"""
Stratified 40/40/20 split of labeled verses into train/val/test.

Requires labels.overall for quality tertiles; combines with source for stratification key.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo_rhyme.metric_benchmark.protocol import default_protocol_manifest, save_protocol_manifest
from evo_rhyme.metric_benchmark.split import split_three_way

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_verses(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Labeled JSONL (e.g. master_labeled.jsonl)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/metric_benchmark/splits")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = _load_verses(args.input)
    split_rows, extra = split_three_way(rows, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    loose: List[Dict[str, Any]] = []

    for r in split_rows:
        sp = r.get("split")
        if sp in by_split:
            by_split[sp].append(r)
        else:
            loose.append(r)

    for name, lst in by_split.items():
        outp = args.out_dir / f"{name}.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for r in lst:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows to %s", len(lst), outp)

    if loose:
        with open(args.out_dir / "unassigned.jsonl", "w", encoding="utf-8") as f:
            for r in loose:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = default_protocol_manifest()
    manifest["split_seed"] = args.seed
    manifest["split_stats"] = extra
    n_bars_hist: Counter[int] = Counter()
    for r in split_rows:
        if "lyrics" in r:
            n_bars_hist[len(r["lyrics"])] += 1
    manifest["n_bars_histogram"] = {str(k): v for k, v in sorted(n_bars_hist.items())}

    save_protocol_manifest(args.out_dir / "split_manifest.json", manifest)
    logger.info("Wrote manifest %s", args.out_dir / "split_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
