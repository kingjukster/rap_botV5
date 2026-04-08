"""
JSONL record validation for metric benchmark verses and pairwise judgments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Human / AI scalar labels (Option A — aligned with plan)
LABEL_KEYS: Tuple[str, ...] = (
    "flow",
    "rhyme",
    "semantic",
    "punchline",
    "fluency",
    "originality",
    "overall",
)

VALID_SOURCES = frozenset({"corpus", "model", "synthetic"})
PAIR_TYPES = frozenset({"random", "adversarial_flow_rhyme"})
BETTER_VALUES = frozenset({"a", "b", "tie"})
AXES = frozenset({"overall", "flow", "rhyme", "semantic", "punchline"})


@dataclass
class VerseRecord:
    id: str
    lyrics: List[str]
    labels: Optional[Dict[str, float]]
    source: str
    label_provenance: Optional[Dict[str, str]] = None
    n_bars: Optional[int] = None
    harvest: Optional[Dict[str, Any]] = None
    split: Optional[str] = None
    unlabeled: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseRecord:
    id: str
    verse_a_id: str
    verse_b_id: str
    better: Optional[str]
    axis: str
    pair_type: str
    split: str
    judge: Optional[str] = None
    reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _as_float_01(x: Any, key: str) -> float:
    v = float(x)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{key} must be in [0,1], got {v}")
    return v


def validate_verse_record(
    obj: Dict[str, Any],
    *,
    min_bars: int = 2,
    require_labels: bool = True,
) -> Tuple[bool, List[str]]:
    """Return (ok, errors)."""
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["record must be a JSON object"]
    vid = obj.get("id")
    if not vid or not isinstance(vid, str):
        errors.append("missing or invalid id")
    lyrics = obj.get("lyrics")
    if not isinstance(lyrics, list) or len(lyrics) < min_bars:
        errors.append(f"lyrics must be a list of at least {min_bars} strings")
    else:
        for i, line in enumerate(lyrics):
            if not isinstance(line, str) or not line.strip():
                errors.append(f"lyrics[{i}] must be a non-empty string")
    src = obj.get("source", "corpus")
    if src not in VALID_SOURCES:
        errors.append(f"source must be one of {sorted(VALID_SOURCES)}, got {src!r}")

    unlabeled = bool(obj.get("unlabeled")) or obj.get("labels") is None
    if require_labels and not unlabeled:
        labels = obj.get("labels")
        if not isinstance(labels, dict):
            errors.append("labels must be an object when not unlabeled")
        else:
            for k in LABEL_KEYS:
                if k not in labels:
                    errors.append(f"labels missing key {k!r}")
                else:
                    try:
                        _as_float_01(labels[k], k)
                    except (TypeError, ValueError) as e:
                        errors.append(str(e))
            extra = set(labels.keys()) - set(LABEL_KEYS)
            if extra:
                errors.append(f"unknown label keys: {sorted(extra)}")
    prov = obj.get("label_provenance")
    if prov is not None and not isinstance(prov, dict):
        errors.append("label_provenance must be an object or omitted")

    return len(errors) == 0, errors


def parse_verse_line(line: str, **kwargs: Any) -> VerseRecord:
    obj = json.loads(line)
    ok, errs = validate_verse_record(obj, **kwargs)
    if not ok:
        raise ValueError("; ".join(errs))
    lyrics = [str(x).strip() for x in obj["lyrics"]]
    unlabeled = bool(obj.get("unlabeled")) or obj.get("labels") is None
    return VerseRecord(
        id=str(obj["id"]),
        lyrics=lyrics,
        labels=None if unlabeled else {k: float(obj["labels"][k]) for k in LABEL_KEYS},
        source=str(obj.get("source", "corpus")),
        label_provenance=dict(obj["label_provenance"]) if obj.get("label_provenance") else None,
        n_bars=int(obj["n_bars"]) if obj.get("n_bars") is not None else len(lyrics),
        harvest=dict(obj["harvest"]) if obj.get("harvest") else None,
        split=obj.get("split"),
        unlabeled=unlabeled,
        raw=dict(obj),
    )


def validate_pairwise_record(
    obj: Dict[str, Any],
    *,
    require_judgment: bool = True,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return False, ["record must be a JSON object"]
    for k in ("id", "verse_a_id", "verse_b_id", "split"):
        v = obj.get(k)
        if not v or not isinstance(v, str):
            errors.append(f"missing or invalid {k}")
    if obj.get("verse_a_id") == obj.get("verse_b_id"):
        errors.append("verse_a_id and verse_b_id must differ")
    pt = obj.get("pair_type", "random")
    if pt not in PAIR_TYPES:
        errors.append(f"pair_type must be one of {sorted(PAIR_TYPES)}")
    ax = obj.get("axis", "overall")
    if ax not in AXES:
        errors.append(f"axis must be one of {sorted(AXES)}")
    sp = obj.get("split")
    if sp not in ("train", "val", "test"):
        errors.append("split must be train, val, or test")
    bet = obj.get("better")
    if require_judgment:
        if bet not in BETTER_VALUES:
            errors.append(f"better must be one of {sorted(BETTER_VALUES)}")
    elif bet is not None and bet not in BETTER_VALUES:
        errors.append(f"better must be null or one of {sorted(BETTER_VALUES)}")
    return len(errors) == 0, errors


def parse_pairwise_line(line: str, **kwargs: Any) -> PairwiseRecord:
    obj = json.loads(line)
    ok, errs = validate_pairwise_record(obj, **kwargs)
    if not ok:
        raise ValueError("; ".join(errs))
    return PairwiseRecord(
        id=str(obj["id"]),
        verse_a_id=str(obj["verse_a_id"]),
        verse_b_id=str(obj["verse_b_id"]),
        better=str(obj["better"]) if obj.get("better") is not None else None,
        axis=str(obj.get("axis", "overall")),
        pair_type=str(obj.get("pair_type", "random")),
        split=str(obj["split"]),
        judge=obj.get("judge"),
        reason=obj.get("reason"),
        raw=dict(obj),
    )


def verse_ids_by_split(verses: List[VerseRecord]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    for v in verses:
        if v.split in out:
            out[v.split].add(v.id)
    return out


def pairwise_crosses_split(
    pair: PairwiseRecord,
    verse_splits: Dict[str, Set[str]],
) -> bool:
    """True if a and b are not in the same split bucket (leakage)."""
    sa = sb = None
    for name, ids in verse_splits.items():
        if pair.verse_a_id in ids:
            sa = name
        if pair.verse_b_id in ids:
            sb = name
    if sa is None or sb is None:
        return True
    return sa != sb
