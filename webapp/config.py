"""Web app configuration."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "webapp" / "templates"
STATIC_DIR = ROOT / "webapp" / "static"
RUNS_DIR = ROOT / "data" / "evo_rhyme" / "runs"
