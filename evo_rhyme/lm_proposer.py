"""
evo_rhyme/lm_proposer.py

LM-backed bar proposer for evolutionary rap verse generation.
Uses an OpenAI-compatible API to generate candidate bars conditioned on
theme, rhyme targets, syllable targets, and section roles.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

from evo_rhyme.phonetics import (
    extract_rhyme_tail,
    get_pronunciations,
    syllable_count_line,
)

load_dotenv(override=True)
logger = logging.getLogger(__name__)

ROLE_DESCRIPTIONS: Dict[str, str] = {
    "setup": "introduce the topic and set the scene",
    "flex": "brag, boast, or assert dominance",
    "threat": "deliver aggressive or confrontational energy",
    "introspection": "be reflective, vulnerable, or philosophical",
    "punchline": "land a clever twist, wordplay, or surprise ending",
}

_NUM_RE = re.compile(r"^\s*\d+[\.\)\-:]\s*")
_BULLET_RE = re.compile(r"^\s*[-*•]\s*")
_QUOTE_RE = re.compile(r'^["\']|["\']$')


# ---------------------------------------------------------------------------
# Config & request dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProposerConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    bars_per_slot: int = 80
    temperature: float = 0.95
    top_p: float = 0.95
    max_tokens: int = 60
    batch_size: int = 20
    timeout: float = 30.0
    max_retries: int = 3


@dataclass
class BarRequest:
    theme_keywords: List[str]
    rhyme_target: Optional[str] = None
    rhyme_family: Optional[str] = None
    syllable_range: Tuple[int, int] = (8, 14)
    role: str = "flex"
    persona: Optional[str] = None
    avoid_words: Optional[Set[str]] = None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(
    n: int,
    request: BarRequest,
) -> str:
    min_syl, max_syl = request.syllable_range
    theme = ", ".join(request.theme_keywords)
    role_desc = ROLE_DESCRIPTIONS.get(request.role, request.role)

    parts = [
        f"You are a rap lyricist. Generate {n} unique rap bars (one per line).",
        "Each bar must:",
        f"- Be {min_syl}-{max_syl} syllables",
        f"- Fit the theme: {theme}",
    ]

    if request.rhyme_target:
        parts.append(
            f'- End with a word that rhymes with "{request.rhyme_target}"'
        )
    elif request.rhyme_family:
        parts.append(
            f"- End with a word whose rhyme tail matches: {request.rhyme_family}"
        )

    parts.append(f"- Serve as a {request.role} bar ({role_desc})")

    if request.persona:
        parts.append(f"- Write from the persona: {request.persona}")

    if request.avoid_words:
        avoid = ", ".join(sorted(request.avoid_words))
        parts.append(f"- Do NOT use these words: {avoid}")

    parts.append("")
    parts.append("Output ONLY the bars, one per line. No numbering, no explanations.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _clean_line(line: str) -> str:
    """Strip numbering, bullets, quotes, and whitespace from a raw LM line."""
    line = line.strip()
    line = _NUM_RE.sub("", line)
    line = _BULLET_RE.sub("", line)
    line = _QUOTE_RE.sub("", line)
    return line.strip()


def _end_word(line: str) -> str:
    words = re.findall(r"[A-Za-z']+", line.lower())
    return words[-1] if words else ""


def _filter_bars(
    raw_lines: List[str],
    request: BarRequest,
) -> List[str]:
    """Apply syllable, pronunciation, and rhyme filters; deduplicate."""
    min_syl, max_syl = request.syllable_range
    seen: Set[str] = set()
    accepted: List[str] = []

    for raw in raw_lines:
        line = _clean_line(raw)
        if not line:
            continue

        key = line.lower()
        if key in seen:
            continue

        syl = syllable_count_line(line)
        if syl < min_syl or syl > max_syl:
            continue

        end = _end_word(line)
        if not end:
            continue
        if not get_pronunciations(end):
            continue

        if request.rhyme_target:
            target_tail = extract_rhyme_tail(request.rhyme_target)
            bar_tail = extract_rhyme_tail(end)
            if target_tail and bar_tail and target_tail != bar_tail:
                continue

        seen.add(key)
        accepted.append(line)

    return accepted


# ---------------------------------------------------------------------------
# Scheme helpers
# ---------------------------------------------------------------------------

_SCHEME_PAIRS: Dict[str, List[Tuple[int, ...]]] = {
    "AABB": [(0, 1), (2, 3)],
    "ABAB": [(0, 2), (1, 3)],
    "ABBA": [(0, 3), (1, 2)],
    "ABCB": [(1, 3)],
}


def _rhyme_groups(scheme: str, num_lines: int) -> List[Tuple[int, ...]]:
    """Return groups of line indices that must share a rhyme family."""
    canonical = scheme.upper()
    if canonical in _SCHEME_PAIRS:
        return _SCHEME_PAIRS[canonical]

    groups_map: Dict[str, List[int]] = {}
    for idx, ch in enumerate(canonical):
        if idx >= num_lines:
            break
        groups_map.setdefault(ch, []).append(idx)
    return [tuple(v) for v in groups_map.values() if len(v) > 1]


# ---------------------------------------------------------------------------
# BarProposer
# ---------------------------------------------------------------------------

class BarProposer:
    """Generate candidate rap bars via an OpenAI-compatible LM."""

    def __init__(self, config: ProposerConfig | None = None) -> None:
        self.cfg = config or ProposerConfig()
        self._api_key = self.cfg.api_key or os.getenv("OPENAI_API_KEY", "")
        if not self._api_key:
            logger.warning("No API key found – set OPENAI_API_KEY or pass api_key in config")

    # -- async core ---------------------------------------------------------

    async def _call_api(self, system: str, n: int) -> List[str]:
        """Single API call; returns parsed lines from the response."""
        import openai

        client_kwargs: Dict = {"api_key": self._api_key, "timeout": self.cfg.timeout}
        if self.cfg.api_base:
            client_kwargs["base_url"] = self.cfg.api_base

        client = openai.AsyncOpenAI(**client_kwargs)

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Generate {n} bars now."},
                    ],
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_tokens=self.cfg.max_tokens * n,
                )
                text = resp.choices[0].message.content or ""
                return [l for l in text.splitlines() if l.strip()]

            except openai.RateLimitError:
                wait = 2 ** attempt
                logger.warning("Rate-limited, backing off %ds (attempt %d/%d)", wait, attempt, self.cfg.max_retries)
                await asyncio.sleep(wait)

            except openai.APIError as exc:
                wait = 2 ** attempt
                logger.warning("API error: %s – retrying in %ds (attempt %d/%d)", exc, wait, attempt, self.cfg.max_retries)
                await asyncio.sleep(wait)

        logger.error("All %d API attempts exhausted", self.cfg.max_retries)
        return []

    async def _propose_bars_async(self, request: BarRequest) -> List[str]:
        """Generate bars across multiple batched API calls, filter, dedupe."""
        total = self.cfg.bars_per_slot
        batch = self.cfg.batch_size
        num_calls = max(1, (total + batch - 1) // batch)

        system = _build_system_prompt(batch, request)

        tasks = [self._call_api(system, batch) for _ in range(num_calls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        raw_lines: List[str] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("Batch failed: %s", r)
                continue
            raw_lines.extend(r)

        accepted = _filter_bars(raw_lines, request)
        logger.info(
            "Proposed %d raw -> %d accepted bars (role=%s, rhyme=%s)",
            len(raw_lines), len(accepted), request.role, request.rhyme_target,
        )
        return accepted

    async def _propose_verse_pool_async(
        self,
        theme_keywords: List[str],
        scheme: str = "AABB",
        num_lines: int = 4,
        roles: Optional[List[str]] = None,
    ) -> Dict[int, List[str]]:
        if roles is None:
            roles = _default_roles(num_lines)
        if len(roles) != num_lines:
            roles = (roles * ((num_lines // len(roles)) + 1))[:num_lines]

        groups = _rhyme_groups(scheme, num_lines)
        grouped_slots: Set[int] = set()
        for g in groups:
            grouped_slots.update(g)

        slot_rhyme_target: Dict[int, Optional[str]] = {i: None for i in range(num_lines)}

        tasks: List[asyncio.Task] = []
        slot_map: List[int] = []

        for group in groups:
            anchor = group[0]
            for idx in group:
                req = BarRequest(
                    theme_keywords=theme_keywords,
                    rhyme_target=slot_rhyme_target.get(anchor),
                    syllable_range=(8, 14),
                    role=roles[idx],
                )
                tasks.append(asyncio.ensure_future(self._propose_bars_async(req)))
                slot_map.append(idx)

        for idx in range(num_lines):
            if idx not in grouped_slots:
                req = BarRequest(
                    theme_keywords=theme_keywords,
                    syllable_range=(8, 14),
                    role=roles[idx],
                )
                tasks.append(asyncio.ensure_future(self._propose_bars_async(req)))
                slot_map.append(idx)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        pool: Dict[int, List[str]] = {i: [] for i in range(num_lines)}
        for slot_idx, result in zip(slot_map, results):
            if isinstance(result, BaseException):
                logger.warning("Slot %d failed: %s", slot_idx, result)
                continue
            pool[slot_idx].extend(result)

        for idx in pool:
            pool[idx] = list(dict.fromkeys(pool[idx]))

        return pool

    # -- sync wrappers ------------------------------------------------------

    def propose_bars(self, request: BarRequest) -> List[str]:
        """Synchronous wrapper for bar proposal."""
        return _run_async(self._propose_bars_async(request))

    def propose_verse_pool(
        self,
        theme_keywords: List[str],
        scheme: str = "AABB",
        num_lines: int = 4,
        roles: Optional[List[str]] = None,
    ) -> Dict[int, List[str]]:
        """Synchronous wrapper for verse pool proposal.

        Returns ``{slot_index: [candidate_bars]}`` — one pool per line slot.
        """
        return _run_async(
            self._propose_verse_pool_async(theme_keywords, scheme, num_lines, roles)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_roles(num_lines: int) -> List[str]:
    """Sensible role assignments for a verse."""
    defaults_4 = ["setup", "flex", "flex", "punchline"]
    defaults_8 = [
        "setup", "flex", "threat", "flex",
        "introspection", "flex", "threat", "punchline",
    ]
    if num_lines <= 4:
        return defaults_4[:num_lines]
    if num_lines <= 8:
        return defaults_8[:num_lines]
    return (defaults_8 * ((num_lines // 8) + 1))[:num_lines]


def _run_async(coro):
    """Run a coroutine, reusing the current event loop if one is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    return asyncio.run(coro)
