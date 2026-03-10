"""
evo_rhyme/seed_generator.py

Seed generator for evolutionary couplet population. Provides generate_seed_couplets
for corpus-based initial population. Can be replaced by LLM-based generators.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

from evo_rhyme.individual import CoupletIndividual


BAR_TAG = "[BAR]"


def _extract_bar_text(line: str) -> str:
    """Extract lyric text between [BAR] and first tag ([RHY= / [SYL_ / [INT_])."""
    if BAR_TAG not in line:
        return ""
    start = line.index(BAR_TAG) + len(BAR_TAG)
    rest = line[start:]
    for tag in ("[RHY=", "[SYL_", "[INT_"):
        if tag in rest:
            rest = rest.split(tag, 1)[0]
    return rest.strip()


def _load_plain_lines(content: str) -> List[str]:
    """
    Load lines from plain text format (e.g. phaseA_kaggle_verse.txt).
    Splits on newlines; long paragraphs are chunked into bar-sized segments.
    """
    lines: List[str] = []
    for para in content.split("\n"):
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) < 3:
            continue
        if len(words) <= 20:
            lines.append(para)
        else:
            for i in range(0, len(words), 12):
                chunk = words[i : i + 16]
                if len(chunk) >= 4:
                    lines.append(" ".join(chunk))
    return lines


def _load_jsonl_lines(corpus_path: Path) -> List[str]:
    """Load bar texts from JSONL (e.g. elite_kaggle_corpus_clean_plus.jsonl)."""
    import json
    lines: List[str] = []
    if not corpus_path.exists():
        return lines
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get("text") if isinstance(rec, dict) else None
                if text and len(str(text).split()) >= 2:
                    lines.append(str(text).strip())
            except json.JSONDecodeError:
                continue
    return lines


def load_corpus_lines(corpus_path: Path) -> List[str]:
    """Load bar texts from corpus file. Supports [BAR] format, JSONL, or plain line format."""
    return _load_corpus_lines(corpus_path)


def _load_corpus_lines(corpus_path: Path) -> List[str]:
    """Internal: load bar texts from corpus file."""
    lines: List[str] = []
    if not corpus_path.exists():
        return lines
    if corpus_path.suffix.lower() == ".jsonl":
        return _load_jsonl_lines(corpus_path)
    content = corpus_path.read_text(encoding="utf-8")
    # Check if file uses [BAR] format
    if BAR_TAG in content:
        for line in content.split("\n"):
            bar = _extract_bar_text(line)
            if bar and len(bar.split()) >= 2:
                lines.append(bar)
    else:
        lines = _load_plain_lines(content)
    return lines


class SeedGenerator:
    """
    Generator that produces seed couplets from a corpus.
    Implements generate_seed_couplets(theme_keywords, size) for population initialization.
    """

    def __init__(self, corpus_path: Optional[Path] = None):
        self.corpus_path = corpus_path
        self._lines: Optional[List[str]] = None

    def _get_lines(self) -> List[str]:
        if self._lines is not None:
            return self._lines
        path = self.corpus_path
        if path is None:
            root = Path(__file__).resolve().parents[1]
            for candidate in [
                "elite_kaggle_corpus_clean.txt",
                "elite_kaggle_corpus_clean_plus.jsonl",
                "phaseA_kaggle_verse.txt",
            ]:
                p = root / "data" / candidate
                if p.exists():
                    path = p
                    break
            if path is None:
                path = root / "data" / "phaseA_kaggle_verse.txt"
        self._lines = _load_corpus_lines(path)
        return self._lines

    def generate_seed_couplets(
        self,
        theme_keywords: Optional[List[str]] = None,
        size: int = 50,
    ) -> List[CoupletIndividual]:
        """
        Generate initial population of couplets from corpus.

        Args:
            theme_keywords: Optional keywords to prefer (lines containing these ranked higher).
            size: Number of couplets to generate.

        Returns:
            List of CoupletIndividual with line1, line2 set.
        """
        import random

        lines = self._get_lines()
        if len(lines) < 2:
            return []

        keywords = set(w.lower() for w in (theme_keywords or [])) if theme_keywords else set()
        couplets: List[CoupletIndividual] = []
        seen: Set[tuple] = set()

        # Build consecutive pairs from corpus
        pairs: List[tuple] = []
        for i in range(len(lines) - 1):
            l1, l2 = lines[i].strip(), lines[i + 1].strip()
            if not l1 or not l2:
                continue
            key = (l1, l2)
            if key in seen:
                continue
            seen.add(key)
            score = 0
            if keywords:
                text = f"{l1} {l2}".lower()
                score = sum(1 for kw in keywords if kw in text)
            pairs.append((l1, l2, score))

        # Prefer theme-matching pairs, then random
        if keywords and pairs:
            pairs.sort(key=lambda x: -x[2])
        random.shuffle(pairs)

        for l1, l2, _ in pairs[: size * 2]:  # oversample in case of duplicates
            if len(couplets) >= size:
                break
            couplets.append(CoupletIndividual(line1=l1, line2=l2))

        # If not enough from pairs, add random line combinations
        while len(couplets) < size and len(lines) >= 2:
            i, j = random.sample(range(len(lines)), 2)
            if i > j:
                i, j = j, i
            l1, l2 = lines[i].strip(), lines[j].strip()
            if l1 and l2 and (l1, l2) not in seen:
                seen.add((l1, l2))
                couplets.append(CoupletIndividual(line1=l1, line2=l2))

        return couplets[:size]
