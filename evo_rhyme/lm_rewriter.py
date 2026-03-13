"""
evo_rhyme/lm_rewriter.py

LM-guided constrained rewriting for rap bar mutations. Instead of destructive
lexical swaps (replacing one word randomly), asks an LM to rewrite entire bars
while preserving meaning and satisfying specific constraints (rhyme target,
syllable range, theme, internal rhyme position).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from evo_rhyme.phonetics import (
    count_syllables,
    extract_rhyme_tail,
    multisyllable_overlap,
    syllable_count_line,
    tokenize_line,
)

logger = logging.getLogger(__name__)

_NUMBERING_RE = re.compile(r"^\s*(?:\d+[\.\)\-]|\-|\*)\s*")


@dataclass
class RewriterConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.85
    max_tokens: int = 60
    candidates_per_rewrite: int = 5
    timeout: float = 20.0
    max_retries: int = 2
    cache_maxsize: int = 2048


class _LRUCache:
    """Simple ordered-dict LRU cache with a max size."""

    def __init__(self, maxsize: int = 2048):
        self._maxsize = maxsize
        self._store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._maxsize:
                self._store.popitem(last=False)
        self._store[key] = value


def _cache_key(line: str, method: str, constraints: Tuple) -> str:
    raw = f"{line}|{method}|{constraints!r}"
    return hashlib.sha256(raw.encode()).hexdigest()


class BarRewriter:
    """LM-guided bar rewriter. Lazy-initialises the OpenAI client on first call."""

    def __init__(self, config: RewriterConfig | None = None):
        self._config = config or RewriterConfig()
        self._client = None
        self._cache = _LRUCache(maxsize=self._config.cache_maxsize)

    # ------------------------------------------------------------------
    # Lazy client init
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

        from openai import OpenAI

        api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
        kwargs: Dict = {"api_key": api_key}
        if self._config.api_base:
            kwargs["base_url"] = self._config.api_base
        self._client = OpenAI(**kwargs)

    # ------------------------------------------------------------------
    # Raw LM call with retry + cache
    # ------------------------------------------------------------------

    def _call_lm(self, prompt: str, cache_key: str) -> str:
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache hit for key=%s", cache_key[:12])
            return cached

        self._ensure_client()
        last_err: Exception | None = None

        for attempt in range(1, self._config.max_retries + 2):
            try:
                resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=self._config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    timeout=self._config.timeout,
                )
                text = resp.choices[0].message.content or ""
                self._cache.put(cache_key, text)
                return text
            except Exception as exc:
                last_err = exc
                wait = 2 ** attempt
                logger.warning(
                    "LM call attempt %d/%d failed: %s – retrying in %ds",
                    attempt,
                    self._config.max_retries + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("LM call failed after %d attempts: %s", self._config.max_retries + 1, last_err)
        return ""

    # ------------------------------------------------------------------
    # Response parsing & filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_lines(raw: str) -> List[str]:
        """Split raw LM response into individual candidate lines."""
        candidates: List[str] = []
        for line in raw.strip().splitlines():
            line = _NUMBERING_RE.sub("", line).strip()
            line = line.strip('"').strip("'").strip()
            if line:
                candidates.append(line)
        return candidates

    @staticmethod
    def _filter_syllables(
        lines: List[str],
        syl_range: Tuple[int, int],
    ) -> List[str]:
        lo, hi = syl_range
        kept: List[str] = []
        for line in lines:
            sc = syllable_count_line(line)
            if lo <= sc <= hi:
                kept.append(line)
        return kept

    @staticmethod
    def _filter_end_rhyme(
        lines: List[str],
        rhyme_target: str,
        min_overlap: int = 2,
    ) -> List[str]:
        target_tail = extract_rhyme_tail(rhyme_target)
        if target_tail is None:
            return lines
        kept: List[str] = []
        for line in lines:
            tokens = tokenize_line(line)
            if not tokens:
                continue
            end_tail = extract_rhyme_tail(tokens[-1])
            if multisyllable_overlap(end_tail, target_tail) >= min_overlap:
                kept.append(line)
        return kept

    @staticmethod
    def _dedupe(lines: List[str], original: str) -> List[str]:
        seen: set[str] = set()
        orig_lower = original.strip().lower()
        result: List[str] = []
        for line in lines:
            key = line.strip().lower()
            if key == orig_lower:
                continue
            if key in seen:
                continue
            seen.add(key)
            result.append(line)
        return result

    def _post_process(
        self,
        raw: str,
        original: str,
        syl_range: Tuple[int, int],
        *,
        rhyme_target: str | None = None,
    ) -> List[str]:
        lines = self._parse_lines(raw)
        lines = self._filter_syllables(lines, syl_range)
        if rhyme_target is not None:
            lines = self._filter_end_rhyme(lines, rhyme_target)
        lines = self._dedupe(lines, original)
        return lines

    # ------------------------------------------------------------------
    # Public rewrite methods
    # ------------------------------------------------------------------

    def rhyme_rewrite(
        self,
        line: str,
        rhyme_target: str,
        theme_keywords: List[str],
        syllable_range: Tuple[int, int] = (8, 14),
    ) -> List[str]:
        """Rewrite bar to end with a word rhyming with *rhyme_target*."""
        n = self._config.candidates_per_rewrite
        theme = ", ".join(theme_keywords) if theme_keywords else "hip-hop"
        prompt = (
            f'Rewrite this rap bar. Keep the meaning and flow.\n'
            f'The last word MUST rhyme with "{rhyme_target}".\n'
            f'Stay between {syllable_range[0]}-{syllable_range[1]} syllables.\n'
            f'Theme: {theme}\n\n'
            f'Original: "{line}"\n\n'
            f'Write {n} different rewrites, one per line. Output ONLY the rewritten bars.'
        )
        key = _cache_key(line, "rhyme_rewrite", (rhyme_target, tuple(theme_keywords), syllable_range))
        logger.debug("rhyme_rewrite line=%r target=%r", line[:40], rhyme_target)
        raw = self._call_lm(prompt, key)
        return self._post_process(raw, line, syllable_range, rhyme_target=rhyme_target)

    def internal_rhyme_rewrite(
        self,
        line: str,
        target_position: int,
        rhyme_with: str,
        theme_keywords: List[str],
        syllable_range: Tuple[int, int] = (8, 14),
    ) -> List[str]:
        """Rewrite bar to add internal rhyme at approximate word position."""
        n = self._config.candidates_per_rewrite
        prompt = (
            f'Rewrite this rap bar to add an internal rhyme.\n'
            f'Around word position {target_position}, use a word that rhymes with "{rhyme_with}".\n'
            f'Keep the meaning. Stay between {syllable_range[0]}-{syllable_range[1]} syllables.\n\n'
            f'Original: "{line}"\n\n'
            f'Write {n} rewrites, one per line. Output ONLY the rewritten bars.'
        )
        key = _cache_key(
            line, "internal_rhyme_rewrite",
            (target_position, rhyme_with, tuple(theme_keywords), syllable_range),
        )
        logger.debug("internal_rhyme_rewrite line=%r pos=%d rhyme_with=%r", line[:40], target_position, rhyme_with)
        raw = self._call_lm(prompt, key)
        return self._post_process(raw, line, syllable_range)

    def theme_rewrite(
        self,
        line: str,
        theme_keywords: List[str],
        syllable_range: Tuple[int, int] = (8, 14),
    ) -> List[str]:
        """Rewrite bar to increase theme relevance while preserving rhyme/flow."""
        n = self._config.candidates_per_rewrite
        theme = ", ".join(theme_keywords) if theme_keywords else "hip-hop"
        prompt = (
            f'Rewrite this rap bar to better fit the theme: {theme}.\n'
            f'Keep the flow and rhyme scheme. Stay between {syllable_range[0]}-{syllable_range[1]} syllables.\n\n'
            f'Original: "{line}"\n\n'
            f'Write {n} rewrites, one per line. Output ONLY the rewritten bars.'
        )
        key = _cache_key(line, "theme_rewrite", (tuple(theme_keywords), syllable_range))
        logger.debug("theme_rewrite line=%r theme=%r", line[:40], theme)
        raw = self._call_lm(prompt, key)
        return self._post_process(raw, line, syllable_range)

    def paraphrase(
        self,
        line: str,
        preserve_end_rhyme: bool = True,
        syllable_range: Tuple[int, int] = (8, 14),
    ) -> List[str]:
        """Paraphrase bar preserving rhyme and syllable count."""
        n = self._config.candidates_per_rewrite
        rhyme_note = "Keep the end rhyme sound.\n" if preserve_end_rhyme else ""
        prompt = (
            f'Paraphrase this rap bar. Different words, same meaning and vibe.\n'
            f'{rhyme_note}'
            f'Stay between {syllable_range[0]}-{syllable_range[1]} syllables.\n\n'
            f'Original: "{line}"\n\n'
            f'Write {n} paraphrases, one per line. Output ONLY the rewritten bars.'
        )
        key = _cache_key(line, "paraphrase", (preserve_end_rhyme, syllable_range))
        logger.debug("paraphrase line=%r preserve_end=%s", line[:40], preserve_end_rhyme)
        raw = self._call_lm(prompt, key)

        rhyme_target: str | None = None
        if preserve_end_rhyme:
            tokens = tokenize_line(line)
            rhyme_target = tokens[-1] if tokens else None

        return self._post_process(raw, line, syllable_range, rhyme_target=rhyme_target)

    def tighten(
        self,
        line: str,
        target_syllables: int,
        syllable_range: Tuple[int, int] = (6, 12),
    ) -> List[str]:
        """Rewrite bar to be shorter by 1-2 syllables."""
        n = self._config.candidates_per_rewrite
        prompt = (
            f'Make this rap bar tighter. Reduce to about {target_syllables} syllables.\n'
            f'Keep the meaning and end rhyme.\n\n'
            f'Original: "{line}"\n\n'
            f'Write {n} tighter versions, one per line. Output ONLY the rewritten bars.'
        )
        key = _cache_key(line, "tighten", (target_syllables, syllable_range))
        logger.debug("tighten line=%r target_syl=%d", line[:40], target_syllables)
        raw = self._call_lm(prompt, key)
        return self._post_process(raw, line, syllable_range)

    def expand(
        self,
        line: str,
        target_syllables: int,
        syllable_range: Tuple[int, int] = (10, 16),
    ) -> List[str]:
        """Rewrite bar to be longer by 1-2 syllables."""
        n = self._config.candidates_per_rewrite
        prompt = (
            f'Expand this rap bar. Add detail to reach about {target_syllables} syllables.\n'
            f'Keep the meaning and end rhyme.\n\n'
            f'Original: "{line}"\n\n'
            f'Write {n} expanded versions, one per line. Output ONLY the rewritten bars.'
        )
        key = _cache_key(line, "expand", (target_syllables, syllable_range))
        logger.debug("expand line=%r target_syl=%d", line[:40], target_syllables)
        raw = self._call_lm(prompt, key)
        return self._post_process(raw, line, syllable_range)
