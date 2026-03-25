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

    @staticmethod
    def _line_fingerprint(line: str) -> str:
        """Lowercase first 3 words -- fast similarity proxy."""
        return " ".join(line.lower().split()[:3])

    def generate_seed_couplets(
        self,
        theme_keywords: Optional[List[str]] = None,
        size: int = 50,
    ) -> List[CoupletIndividual]:
        """
        Generate initial population of couplets from corpus.

        Lines are sampled independently (not consecutively) to maximize
        diversity.  Theme-matching lines are preferred but paired randomly.
        A fingerprint dedup prevents near-duplicate couplets.
        """
        import random

        lines = self._get_lines()
        if len(lines) < 2:
            return []

        keywords = set(w.lower() for w in (theme_keywords or [])) if theme_keywords else set()

        if keywords:
            themed = [l for l in lines if any(kw in l.lower() for kw in keywords)]
            other = [l for l in lines if l not in set(themed)]
            random.shuffle(themed)
            random.shuffle(other)
            pool = themed + other
        else:
            pool = list(lines)
            random.shuffle(pool)

        couplets: List[CoupletIndividual] = []
        seen_fps: Set[tuple] = set()
        max_attempts = size * 5

        for _ in range(max_attempts):
            if len(couplets) >= size:
                break
            i, j = random.sample(range(len(pool)), 2)
            l1, l2 = pool[i].strip(), pool[j].strip()
            if not l1 or not l2:
                continue
            fp = (self._line_fingerprint(l1), self._line_fingerprint(l2))
            if fp in seen_fps:
                continue
            seen_fps.add(fp)
            couplets.append(CoupletIndividual(line1=l1, line2=l2))

        return couplets[:size]
