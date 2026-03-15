#!/usr/bin/env python
"""
Build and inspect a lightweight rhyme embedding artifact.

This script is intentionally fast and deterministic. It does not train a deep
model; it materializes the current rhyme embedding vectors derived from rhyme
tails and writes them for offline inspection/use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evo_rhyme.rhyme_embedding import get_rhyme_embedding_space


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build rhyme embedding artifact")
    p.add_argument(
        "--output",
        type=str,
        default="data/evo_rhyme/rhyme_embeddings_light.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--max-words",
        type=int,
        default=50000,
        help="Max number of words to write",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    space = get_rhyme_embedding_space()

    out = {
        "dim": int(space.config.dim),
        "lexical_noise_scale": float(space.config.lexical_noise_scale),
        "word_count": len(space.word_vectors),
        "vectors": {},
    }

    for i, (word, vec) in enumerate(space.word_vectors.items()):
        if i >= max(1, args.max_words):
            break
        out["vectors"][word] = vec

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[2] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh)

    print(f"Wrote {len(out['vectors'])} vectors -> {out_path}")


if __name__ == "__main__":
    main()

