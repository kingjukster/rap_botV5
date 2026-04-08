#!/usr/bin/env python3
"""
Generate pairwise JSONL (same split only): random and adversarial_flow_rhyme.

Use --mock-judgments to fill better from labels (overall or axis) for pipeline tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _by_split(verses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for v in verses:
        sp = v.get("split")
        if sp in d and isinstance(v.get("labels"), dict):
            d[sp].append(v)
    return d


def _mock_better(va: Dict[str, Any], vb: Dict[str, Any], axis: str) -> str:
    la = va["labels"]
    lb = vb["labels"]
    if axis == "overall":
        ka, kb = "overall", "overall"
    elif axis == "flow":
        ka, kb = "flow", "flow"
    else:
        ka, kb = "overall", "overall"
    da = float(la[ka])
    db = float(lb[kb])
    eps = 1e-6
    if abs(da - db) < 0.02:
        return "tie"
    return "a" if da > db else "b"


def random_pairs(
    bucket: List[Dict[str, Any]],
    n: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    ids = [v["id"] for v in bucket]
    if len(ids) < 2:
        return []
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        i, j = rng.integers(0, len(ids), size=2)
        if i == j:
            continue
        a, b = ids[int(i)], ids[int(j)]
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "verse_a_id": a,
                "verse_b_id": b,
                "pair_type": "random",
            }
        )
    return out


def adversarial_pairs(
    bucket: List[Dict[str, Any]],
    n: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    hi_rh_lo_f = [
        v
        for v in bucket
        if float(v["labels"]["rhyme"]) >= 0.55 and float(v["labels"]["flow"]) <= 0.5
    ]
    lo_rh_hi_f = [
        v
        for v in bucket
        if float(v["labels"]["rhyme"]) <= 0.5 and float(v["labels"]["flow"]) >= 0.55
    ]
    if not hi_rh_lo_f or not lo_rh_hi_f:
        logger.warning("Adversarial buckets empty for this split; skipping adversarial pairs")
        return []
    out: List[Dict[str, Any]] = []
    for _ in range(n):
        va = hi_rh_lo_f[int(rng.integers(0, len(hi_rh_lo_f)))]
        vb = lo_rh_hi_f[int(rng.integers(0, len(lo_rh_hi_f)))]
        if va["id"] == vb["id"]:
            continue
        out.append(
            {
                "verse_a_id": va["id"],
                "verse_b_id": vb["id"],
                "pair_type": "adversarial_flow_rhyme",
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, default=ROOT / "data/metric_benchmark/splits")
    p.add_argument("--per-split-random", type=int, default=30)
    p.add_argument("--per-split-adversarial", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--axis", type=str, default="overall", choices=("overall", "flow"))
    p.add_argument("--mock-judgments", action="store_true")
    p.add_argument("--judge", type=str, default="mock")
    args = p.parse_args()

    all_v: List[Dict[str, Any]] = []
    for name in ("train", "val", "test"):
        fp = args.splits_dir / f"{name}.jsonl"
        if fp.exists():
            all_v.extend(_load(fp))

    by_sp = _by_split(all_v)
    id_to_v = {v["id"]: v for v in all_v}
    rng = np.random.default_rng(args.seed)
    pair_counter = 0

    for split_name in ("train", "val", "test"):
        bucket = by_sp[split_name]
        outp = args.splits_dir / f"pairs_{split_name}.jsonl"
        rows: List[Dict[str, Any]] = []

        for spec in random_pairs(bucket, args.per_split_random, rng):
            pair_counter += 1
            rec: Dict[str, Any] = {
                "id": f"pair_{split_name}_{pair_counter:06d}",
                "verse_a_id": spec["verse_a_id"],
                "verse_b_id": spec["verse_b_id"],
                "axis": args.axis,
                "pair_type": spec["pair_type"],
                "split": split_name,
                "judge": args.judge,
            }
            if args.mock_judgments:
                va = id_to_v[rec["verse_a_id"]]
                vb = id_to_v[rec["verse_b_id"]]
                rec["better"] = _mock_better(va, vb, args.axis)
            rows.append(rec)

        for spec in adversarial_pairs(bucket, args.per_split_adversarial, rng):
            pair_counter += 1
            rec = {
                "id": f"pair_{split_name}_{pair_counter:06d}",
                "verse_a_id": spec["verse_a_id"],
                "verse_b_id": spec["verse_b_id"],
                "axis": args.axis,
                "pair_type": spec["pair_type"],
                "split": split_name,
                "judge": args.judge,
            }
            if args.mock_judgments:
                va = id_to_v[rec["verse_a_id"]]
                vb = id_to_v[rec["verse_b_id"]]
                rec["better"] = _mock_better(va, vb, args.axis)
            rows.append(rec)

        with open(outp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %d pairs to %s", len(rows), outp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
