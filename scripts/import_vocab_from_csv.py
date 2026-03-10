#!/usr/bin/env python3
"""
Import vocab from CSV/text files into evo_rhyme vocab format.

Usage:
  python scripts/import_vocab_from_csv.py [options]

  --verbs PATH       Path to verbs.csv (present,past,past_participle)
  --participles PATH Path to words-multiple-present-participle.csv
  --nouns PATH       Path to nounlist.csv (one word per line)
  --merge            Merge with existing vocab (default)
  --replace          Replace existing vocab entirely
  --dry-run          Print what would be written, don't write files
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


_IRREGULAR_ING: dict[str, str] = {
    "be": "being", "is": "being", "are": "being",
    "see": "seeing", "flee": "fleeing", "agree": "agreeing",
    "lie": "lying", "die": "dying", "tie": "tying", "vie": "vying",
}

def _derive_ing(verb: str) -> str:
    """Derive present participle from base verb. Simple rules for common cases."""
    v = verb.lower().strip()
    if not v:
        return ""
    if v in _IRREGULAR_ING:
        return _IRREGULAR_ING[v]
    # Ends in -e: drop e and add ing (make->making)
    if len(v) > 2 and v.endswith("e"):
        return v[:-1] + "ing"
    # Ends in CVC (consonant-vowel-consonant): double final consonant
    if len(v) >= 3:
        c, vc, cc = v[-3], v[-2], v[-1]
        if cc in "bdfgklmnprst" and vc in "aeiou" and c not in "aeiou":
            return v + cc + "ing"
    return v + "ing"


def _derive_past(verb: str, csv_map: dict[str, str]) -> str:
    """Derive past tense; use CSV mapping when available."""
    if verb in csv_map:
        return csv_map[verb]
    v = verb.lower().strip()
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def load_verbs_csv(path: Path) -> tuple[list[str], list[str]]:
    """Load verbs.csv. Returns (presents, pasts) aligned by row."""
    presents, pasts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                p, past = row[0].strip().lower(), row[1].strip().lower()
                if p and past:
                    presents.append(p)
                    pasts.append(past)
    return presents, pasts


def load_nounlist(path: Path) -> list[str]:
    """Load nounlist (one word per line). Returns list of lowercase nouns."""
    if not path.exists():
        return []
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_participles_csv(path: Path) -> dict[str, list[str]]:
    """Load words-multiple-present-participle.csv. Returns verb -> [ing_forms]."""
    result: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("Word", "").strip().lower()
            p1 = row.get("Present Participle", "").strip().lower()
            p2 = row.get("Present Participle Alternative", "").strip().lower()
            if not word or not p1:
                continue
            forms = [p1]
            if p2:
                for alt in re.split(r"[\s,]+", p2):
                    if alt and alt not in forms:
                        forms.append(alt)
            result[word] = forms
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Import verb vocab from CSV into evo_rhyme")
    default_dl = Path.home() / "Downloads"
    ap.add_argument(
        "--verbs",
        type=Path,
        default=default_dl / "verbs.csv",
        help="Path to verbs.csv",
    )
    ap.add_argument(
        "--participles",
        type=Path,
        default=default_dl / "words-multiple-present-participle.csv",
        help="Path to words-multiple-present-participle.csv",
    )
    vocab_default = Path(__file__).resolve().parents[1] / "data" / "evo_rhyme" / "vocab"
    ap.add_argument(
        "--nouns",
        type=Path,
        default=vocab_default / "nounlist.csv",
        help="Path to nounlist.csv (one word per line). Default: data/evo_rhyme/vocab/nounlist.csv",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge with existing vocab (default)",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing vocab entirely",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written, don't write",
    )
    args = ap.parse_args()

    merge = args.merge and not args.replace

    # Project paths
    root = Path(__file__).resolve().parents[1]
    vocab_dir = root / "data" / "evo_rhyme" / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    # Load existing
    def _load_txt(name: str) -> list[str]:
        p = vocab_dir / name
        if not p.exists():
            return []
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    existing_verbs = _load_txt("verbs.txt")
    existing_past = _load_txt("verbs_past.txt")
    existing_ing = _load_txt("verbs_ing.txt")
    existing_nouns = _load_txt("nouns.txt")

    # Build verb -> past mapping from CSV
    verb_to_past: dict[str, str] = {}
    if args.verbs.exists():
        verbs_csv, pasts_csv = load_verbs_csv(args.verbs)
        verb_to_past = dict(zip(verbs_csv, pasts_csv))
        print(f"Loaded {len(verb_to_past)} verbs from {args.verbs}")
    else:
        print(f"Warning: {args.verbs} not found. Skipping verbs import.")

    # Build verb -> ing forms from participles CSV
    participles_map: dict[str, list[str]] = {}
    if args.participles.exists():
        participles_map = load_participles_csv(args.participles)
        print(f"Loaded {len(participles_map)} participle variants from {args.participles}")

    # Build final verbs and verbs_past
    if merge and existing_verbs:
        # Merge: keep existing order, add new verbs from CSV, update past forms from CSV
        seen = set(existing_verbs)
        for v, p in verb_to_past.items():
            if v not in seen:
                seen.add(v)
                existing_verbs.append(v)
        # Align past: use CSV when available, else keep existing or derive
        pasts_final = []
        for i, v in enumerate(existing_verbs):
            past = verb_to_past.get(v) or (existing_past[i] if i < len(existing_past) else _derive_past(v, verb_to_past))
            pasts_final.append(past)
        verbs_final = existing_verbs
    else:
        # Replace: use CSV verbs only
        verbs_final = list(verb_to_past.keys()) if verb_to_past else []
        pasts_final = [_derive_past(v, verb_to_past) for v in verbs_final]

    # Build verbs_ing: derive from all verbs, override with participles CSV
    ing_set: set[str] = set(existing_ing) if merge else set()
    for v in verbs_final:
        if v in participles_map:
            for form in participles_map[v]:
                ing_set.add(form)
        else:
            ing_set.add(_derive_ing(v))
    verbs_ing_final = sorted(ing_set, key=lambda w: (len(w), w))

    # Build nouns from nounlist
    if args.nouns and args.nouns.exists():
        nouns_csv = load_nounlist(args.nouns)
        print(f"Loaded {len(nouns_csv)} nouns from {args.nouns}")
        if merge and existing_nouns:
            seen = set(w.lower() for w in existing_nouns)
            for n in nouns_csv:
                if n not in seen:
                    seen.add(n)
                    existing_nouns.append(n)
            nouns_final = existing_nouns
        else:
            nouns_final = nouns_csv
    else:
        nouns_final = None

    # Write
    def _write(name: str, lines: list[str]) -> None:
        path = vocab_dir / name
        text = "\n".join(lines) + "\n"
        if args.dry_run:
            print(f"[dry-run] Would write {len(lines)} lines to {path}")
            print(f"  Sample: {lines[:5]}...")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"Wrote {len(lines)} lines to {path}")

    _write("verbs.txt", verbs_final)
    _write("verbs_past.txt", pasts_final)
    _write("verbs_ing.txt", verbs_ing_final)
    if nouns_final is not None:
        _write("nouns.txt", nouns_final)


if __name__ == "__main__":
    main()
