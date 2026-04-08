"""Smoke test for run_evaluation on tiny labeled JSONL."""

import json
from pathlib import Path

from evo_rhyme.metric_benchmark.schema import LABEL_KEYS
from evo_rhyme.metric_benchmark.evaluate import run_evaluation


def _verse(vid: str, lines: list, overall: float = 0.5):
    lab = {k: 0.5 for k in LABEL_KEYS}
    lab["overall"] = overall
    lab["flow"] = 0.55
    lab["rhyme"] = 0.5
    return {
        "id": vid,
        "lyrics": lines,
        "labels": lab,
        "source": "synthetic",
        "unlabeled": False,
    }


def test_run_evaluation_smoke(tmp_path: Path):
    vpath = tmp_path / "v.jsonl"
    lines_data = [
        _verse(
            "v1",
            [
                "I step into the cipher with the rhythm and the fire",
                "Every syllable precise while the crowd is getting higher",
            ],
            overall=0.72,
        ),
        _verse(
            "v2",
            [
                "The pen is heavy when the night is long and cold",
                "I trade a quiet truth for the silver and the gold",
            ],
            overall=0.55,
        ),
    ]
    with open(vpath, "w", encoding="utf-8") as f:
        for row in lines_data:
            f.write(json.dumps(row) + "\n")

    ppath = tmp_path / "p.jsonl"
    with open(ppath, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "p1",
                    "verse_a_id": "v1",
                    "verse_b_id": "v2",
                    "better": "a",
                    "axis": "overall",
                    "pair_type": "random",
                    "split": "train",
                }
            )
            + "\n"
        )

    report = run_evaluation(vpath, ppath, fast=True, epsilon=0.05, protocol_path=None)
    assert report["n_scored"] == 2
    assert report["mse_overall"] is not None
    assert report["pairwise"]["n_pairs_used"] >= 1
