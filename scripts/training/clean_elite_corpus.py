#!/usr/bin/env python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
from pathlib import Path

IN_PATH = Path("/workspace/rap-botV4/data/elite_kaggle_corpus_clean.txt")
OUT_PATH = Path("/workspace/rap-botV4/data/elite_kaggle_corpus_clean_filtered.txt")

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
    if not IN_PATH.exists():
        raise SystemExit(f"Input corpus not found: {IN_PATH}")

    kept = 0
    dropped = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open(
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
    print(f"[CLEAN] Output file : {OUT_PATH}")

if __name__ == "__main__":
    main()
