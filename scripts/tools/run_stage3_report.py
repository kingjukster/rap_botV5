#!/usr/bin/env python
"""
run_stage3_report.py

Automate execution of the Stage-3 stats notebook via Papermill and optionally
export an HTML report. Helps keep nightly dashboards reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import os
from datetime import datetime

import papermill as pm
from nbconvert import HTMLExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the Stage-3 stats notebook via Papermill.")
    parser.add_argument(
        "--notebook",
        type=str,
        default="notebooks/stage3_stats.ipynb",
        help="Source notebook to execute.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory to place executed notebooks/HTML exports.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="stage3_report",
        help="Base label for generated files (timestamp appended automatically).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path; sets RAPBOT_CONFIG during execution.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also export an HTML version of the executed notebook.",
    )
    return parser.parse_args()


def export_html(notebook_path: Path, html_path: Path):
    exporter = HTMLExporter()
    body, _ = exporter.from_filename(str(notebook_path))
    html_path.write_text(body, encoding="utf-8")
    print(f"[REPORT] HTML exported to {html_path}")


def main():
    args = parse_args()
    input_nb = Path(args.notebook)
    if not input_nb.exists():
        raise FileNotFoundError(f"Notebook not found: {input_nb}")

    if args.config:
        os.environ["RAPBOT_CONFIG"] = args.config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    executed_nb = output_dir / f"{args.label}_{timestamp}.ipynb"

    print(f"[REPORT] Executing {input_nb} -> {executed_nb}")
    pm.execute_notebook(
        input_path=str(input_nb),
        output_path=str(executed_nb),
        parameters={},
        progress_bar=False,
    )
    print("[REPORT] Notebook execution complete.")

    if args.html:
        html_path = executed_nb.with_suffix(".html")
        export_html(executed_nb, html_path)


if __name__ == "__main__":
    main()
