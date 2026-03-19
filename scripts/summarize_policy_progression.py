#!/usr/bin/env python3
"""Print compact policy progression summary from an analyze_control_impact report JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize policy progression from report JSON")
    ap.add_argument("--report", required=True, help="Path to report JSON")
    args = ap.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"Report not found: {path}", file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    perf = report.get("policy_performance") or {}
    if not perf:
        print("No policy_performance found in report.")
        return 0
    versions = sorted(perf.keys())
    prev = None
    print("policy_version | runs | avg_fitness | delta_vs_prev | status")
    print("--------------|------|-------------|---------------|-------")
    for v in versions:
        item = perf.get(v) or {}
        runs = int(item.get("n_runs", 0))
        avg = float(item.get("avg_fitness", 0.0))
        if prev is None:
            delta = "—"
            status = "baseline"
        else:
            dv = avg - prev
            delta = f"{dv:+.3f}"
            status = "improved" if dv > 0.01 else ("regressed" if dv < -0.01 else "neutral")
        print(f"{v} | {runs} | {avg:.3f} | {delta} | {status}")
        prev = avg
    return 0


if __name__ == "__main__":
    sys.exit(main())
