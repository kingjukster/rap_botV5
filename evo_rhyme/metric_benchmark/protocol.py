"""
Default protocol manifest: rubrics, tie policy, overall weights for evaluation.
Serialized to split_manifest.json / protocol.json alongside splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def default_protocol_manifest() -> Dict[str, Any]:
    return {
        "labeling_protocol_version": "v1",
        "segmentation_rule_version": "stanza_blank_line_v2",
        "tie_policy_version": "v1",
        "flow_definition": {
            "1.0": "Perfect cadence, consistent rhythm, natural to rap aloud",
            "0.7": "Mostly consistent; minor timing or phrasing issues",
            "0.5": "Noticeable rhythm inconsistencies or uneven bar feel",
            "0.3": "Poor cadence, awkward phrasing, hard to rap smoothly",
            "0.0": "Unrappable or chaotic as a rhythmic unit",
        },
        "punchline_definition_v1": (
            "Score memorability and impact of the strongest closing line(s) in the verse "
            "(typically the final bar); ignore unrelated earlier lines unless they set up the payoff."
        ),
        "fluency_definition_v1": (
            "Grammaticality, readability, and natural phrasing as written text — not rhyme quality or topic depth."
        ),
        "originality_definition_v1": (
            "Freshness vs generic or cliché rap phrasing; reward unexpected images or angles."
        ),
        "tie_policy": {
            "epsilon": 0.05,
            "prediction": (
                "if score_a - score_b > epsilon -> a; elif score_b - score_a > epsilon -> b; else tie"
            ),
        },
        "fluency_composite_formula": "mean of available keys among fluency, lm_fluency, lexical_validity from score_verse output",
        "overall_weights_v1": {
            "flow": 0.2,
            "rhyme": 0.2,
            "semantic": 0.2,
            "punchline": 0.15,
            "novelty": 0.15,
            "fluency_composite": 0.1,
        },
    }


def load_protocol_manifest(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("protocol manifest must be a JSON object")
    return data


def save_protocol_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
