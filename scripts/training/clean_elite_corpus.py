#!/usr/bin/env python
"""Filter elite corpus: drop bars with invalid text (too short, no alphanumeric)."""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BAR_TAG = "[BAR]"


def extract_bar_text(line: str) -> str:
    """Extract the lyric text between [BAR] and the first tag ([RHY= / [SYL_ / [INT_])."""
    if BAR_TAG not in line:
        return ""

    start = line.index(BAR_TAG) + len(BAR_TAG)
    rest = line[start:]

    # stop at the first structural tag if present
    for tag in ("[RHY=", "[SYL_", "[INT_"):
        if tag in rest:
            rest = rest.split(tag, 1)[0]
    return rest.strip()

def is_valid_bar_text(text: str) -> bool:
    # must contain at least one alphanumeric character
    if not re.search(r"[A-Za-z0-9]", text):
        return False
    # optionally, require at least 2 tokens to avoid tiny ad-libs
    tokens = text.split()
    if len(tokens) < 2:
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Filter elite corpus: drop invalid bars.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "elite_songs_corpus_clean.txt",
        help="Input corpus path (TXT with [BAR] lines).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "elite_songs_corpus_clean_filtered.txt",
        help="Output corpus path.",
    )
    args = parser.parse_args()
    in_path = args.input
    out_path = args.output

    if not in_path.exists():
        raise SystemExit(f"Input corpus not found: {in_path}")

    kept = 0
    dropped = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if BAR_TAG in line:
                bar_text = extract_bar_text(line)
                if not is_valid_bar_text(bar_text):
                    dropped += 1
                    continue
                kept += 1
                fout.write(line)
            else:
                # Keep metadata and <END_SONG> lines as-is
                fout.write(line)

    print(f"[CLEAN] Finished.")
    print(f"[CLEAN] Kept bars   : {kept}")
    print(f"[CLEAN] Dropped bars: {dropped}")
    print(f"[CLEAN] Output file : {out_path}")


if __name__ == "__main__":
    main()
