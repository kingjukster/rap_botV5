"""
rhyme_resources.py

Rhyme group loading and utilities for evo_rhyme.
- Loads rhyme groups from rhymes_grouped.csv
- Provides get_rhyme_group, rhyme_key, rhyme_similarity
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def _load_settings_paths() -> Path:
    """
    Try to pull canonical rhyme CSV path from config.settings so every script
    agrees on where rhyme assets live. Falls back to repo-relative path
    if the config module is unavailable.
    """
    try:
        from config.settings import load_settings

        cfg = load_settings(os.environ.get("RAPBOT_CONFIG"))
        return Path(cfg.rhyme_groups_csv)
    except Exception:
        return ROOT / "data" / "rhymes_grouped.csv"


_DEFAULT_RHYME_CSV = _load_settings_paths()

RHYME_CSV_PATH = Path(
    os.environ.get("RAPBOT_RHYME_CSV")
    or os.environ.get("EVO_RHYME_RHYME_CSV")
    or str(_DEFAULT_RHYME_CSV)
)


# ---------------------------------------------------------------------
# Rhyme groups
# ---------------------------------------------------------------------

def load_rhyme_groups(csv_path: str | Path, validate: bool = True) -> Dict[str, int]:
    """
    Load a mapping word -> group_id from rhymes_grouped.csv.

    Args:
        csv_path: Path to rhymes_grouped.csv
        validate: If True, perform basic validation checks

    Returns:
        Dictionary mapping word -> group_id
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] Rhyme CSV not found at {csv_path}, continuing with empty mapping.")
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read rhyme CSV {csv_path}: {e}")
        return {}

    # Validation checks
    if validate:
        if "word" not in df.columns or "group" not in df.columns:
            print(f"[WARN] Rhyme CSV missing required columns (word, group). Found: {list(df.columns)}")
            return {}

        # Check for empty dataframe
        if len(df) == 0:
            print(f"[WARN] Rhyme CSV is empty")
            return {}

    mapping: Dict[str, int] = {}
    duplicate_count = 0
    invalid_count = 0

    for _, row in df.iterrows():
        w = str(row["word"]).strip().lower()
        if not w:
            continue

        try:
            g = int(row["group"])
            if g < 0:
                invalid_count += 1
                continue

            # Handle duplicates by keeping the last entry
            if w in mapping and mapping[w] != g:
                duplicate_count += 1
            mapping[w] = g
        except (ValueError, KeyError):
            invalid_count += 1
            continue

    if validate and (duplicate_count > 0 or invalid_count > 0):
        print(f"[WARN] Found {duplicate_count} duplicate words and {invalid_count} invalid entries in rhyme CSV")

    print(f"[INFO] Loaded {len(mapping)} rhyme-group entries from {csv_path}")
    return mapping


# Load rhyme groups with validation
RHYME_GROUPS: Dict[str, int] = load_rhyme_groups(RHYME_CSV_PATH, validate=True)


def get_rhyme_group(word: str) -> Optional[int]:
    """Return the rhyme group ID for a word, or None if not found."""
    return RHYME_GROUPS.get(str(word).lower(), None)


def rhyme_key(word: str, max_len: int = 4) -> str:
    """
    Simple fallback rhyme key = lowercased alpha tail, up to max_len chars.
    """
    w = re.sub(r"[^a-zA-Z]", "", str(word).lower())
    if not w:
        return ""
    return w[-max_len:]


def rhyme_similarity(k1: str, k2: str) -> float:
    """
    Suffix-based similarity in [0,1] between two rhyme keys.
    """
    if not k1 or not k2:
        return 0.0
    max_len = min(len(k1), len(k2))
    if max_len == 0:
        return 0.0
    matches = 0
    for i in range(1, max_len + 1):
        if k1[-i:] == k2[-i:]:
            matches = i
        else:
            break
    return matches / max_len


__all__ = [
    "load_rhyme_groups",
    "RHYME_GROUPS",
    "get_rhyme_group",
    "rhyme_key",
    "rhyme_similarity",
    "RHYME_CSV_PATH",
]
