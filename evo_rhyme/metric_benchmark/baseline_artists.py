"""
Baseline artist set for metric benchmark filtering (research-style diversity).

Tiers are rolled into four sampling groups with default quotas:
  technical 30% — Tier 1 (flow) + Tier 2 (rhyme density)
  semantic   30% — Tier 3 (storytelling)
  modern     20% — Tier 4 (bounce / rhythmic lanes)
  control    20% — Tier 5 (structural variety: simpler schemes, looser cadence,
                 irregular timing, repetition-heavy, or inconsistent song structure —
                 not a "bad artist" label; exposes metric sensitivity)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, FrozenSet, List, Optional

# Group order for tie-break when a track lists multiple baseline artists (e.g. feat.)
GROUP_PRIORITY: List[str] = ["technical", "semantic", "modern", "control"]

_DEFAULT_QUOTAS: Dict[str, float] = {
    "technical": 0.30,
    "semantic": 0.30,
    "modern": 0.20,
    "control": 0.20,
}


def default_quota_fractions() -> Dict[str, float]:
    return dict(_DEFAULT_QUOTAS)


def normalize_artist(name: str) -> str:
    """Lowercase, strip, fold accents, collapse internal space."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# All keys are normalize_artist(...) results; include common aliases.
_TECHNICAL: FrozenSet[str] = frozenset(
    {
        normalize_artist(x)
        for x in (
            "Kendrick Lamar",
            "Eminem",
            "JID",
            "André 3000",
            "Andre 3000",
            "MF DOOM",
            "MF Doom",
            "Black Thought",
            "Royce da 5'9\"",
            "Royce Da 5'9\"",
            "Royce 5'9\"",
            "Joey Bada$$",
        )
    }
)

_SEMANTIC: FrozenSet[str] = frozenset(
    {
        normalize_artist(x)
        for x in (
            "Nas",
            "J. Cole",
            "J Cole",
            "Tupac Shakur",
            "2Pac",
            "Tupac",
            "Common",
        )
    }
)

_MODERN: FrozenSet[str] = frozenset(
    {
        normalize_artist(x)
        for x in (
            "Travis Scott",
            "Lil Baby",
            "Future",
        )
    }
)

_CONTROL: FrozenSet[str] = frozenset(
    {
        normalize_artist(x)
        for x in (
            # Prior + expanded Tier 5 (Kaggle-style names; add aliases as needed)
            "Blueface",
            "Ice Spice",
            "Lil Pump",
            "6ix9ine",
            "Tekashi6ix9ine",
            "Tekashi 6ix9ine",
            "Soulja Boy",
            "Soulja Boy Tell 'Em",
            "Fetty Wap",
            "Desiigner",
            "Rich The Kid",
            "Rich the Kid",
            "Silkk the Shocker",
            "Silkk The Shocker",
            "645AR",
            "645ar",
            # Playboi Carti: moved from modern — sampled here for timing/structure variance
            "Playboi Carti",
            "Smokepurpp",
            "Smoke Purpp",
            "Yung Joc",
            "DDG",
            "Tyga",
            "Famous Dex",
        )
    }
)

_GROUP_SETS: Dict[str, FrozenSet[str]] = {
    "technical": _TECHNICAL,
    "semantic": _SEMANTIC,
    "modern": _MODERN,
    "control": _CONTROL,
}


def classify_track_group(artist_names: List[str]) -> Optional[str]:
    """
    Return sampling group for this track, or None if no baseline artist appears.

    If multiple groups match (e.g. feature), pick the earliest in GROUP_PRIORITY.
    """
    matched: List[str] = []
    for raw in artist_names:
        n = normalize_artist(raw)
        if not n:
            continue
        # Exact set membership (avoids 'nas' matching unrelated names)
        for g in GROUP_PRIORITY:
            if n in _GROUP_SETS[g]:
                matched.append(g)
                break
    if not matched:
        return None
    return min(matched, key=lambda g: GROUP_PRIORITY.index(g))


def parse_artists_field(artists_cell: str) -> List[str]:
    """Parse Spotify/Kaggle JSON array string from CSV."""
    import json

    s = (artists_cell or "").strip()
    if not s:
        return []
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if x is not None and str(x).strip()]
