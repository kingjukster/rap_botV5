"""
evo_rhyme/population.py

Population initialization for evolutionary rhyme. Uses a generator to create
seed couplets for the initial population.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evo_rhyme.individual import (
    CoupletIndividual,
    VerseIndividual,
    VerseStructure,
    create_verse_individual,
)

logger = logging.getLogger(__name__)


def create_mixed_population(
    corpus_path: Optional[Path] = None,
    theme_keywords: Optional[List[str]] = None,
    size: int = 50,
    analyze: bool = True,
) -> List[CoupletIndividual]:
    """
    Create initial population with mixed sources:
    - With corpus: 40% template, 40% corpus, 20% random
    - Without corpus: 60% template, 40% random

    Template fill uses existing logic from seed_generator/generator: pick random
    template, fill slots with random words from vocab/tail_to_words.
    Corpus uses SeedGenerator or existing corpus-based generation.
    """
    from evo_rhyme.generator import generate_random_couplets, template_fill_couplets
    from evo_rhyme.seed_generator import SeedGenerator

    corpus_gen = SeedGenerator(corpus_path=corpus_path)
    corpus_lines = corpus_gen._get_lines()
    has_corpus = len(corpus_lines) >= 2

    if has_corpus:
        n_template = int(size * 0.4)
        n_corpus = int(size * 0.4)
        n_random = size - n_template - n_corpus
    else:
        n_template = int(size * 0.6)
        n_corpus = 0
        n_random = size - n_template

    result: List[CoupletIndividual] = []

    if n_template > 0:
        result.extend(
            template_fill_couplets(
                theme_keywords=theme_keywords,
                count=n_template,
                analyze=analyze,
            )
        )

    if n_corpus > 0:
        result.extend(
            corpus_gen.generate_seed_couplets(
                theme_keywords=theme_keywords,
                size=n_corpus,
            )
        )

    if n_random > 0:
        result.extend(
            generate_random_couplets(
                theme_keywords=theme_keywords,
                count=n_random,
                analyze=analyze,
            )
        )

    if len(result) < size:
        extra = template_fill_couplets(
            theme_keywords=theme_keywords,
            count=size - len(result),
            analyze=analyze,
        )
        result.extend(extra)

    return result[:size]


class TemplateGenerator:
    """Generator that produces couplets from templates with rhyme-aware filling."""

    def generate_seed_couplets(
        self,
        theme_keywords: Optional[List[str]] = None,
        size: int = 50,
    ) -> List[CoupletIndividual]:
        from evo_rhyme.generator import template_fill_couplets
        return template_fill_couplets(
            theme_keywords=theme_keywords,
            count=size,
            analyze=True,
        )


class RandomGenerator:
    """Generator that produces couplets from templates with random word filling."""

    def generate_seed_couplets(
        self,
        theme_keywords: Optional[List[str]] = None,
        size: int = 50,
    ) -> List[CoupletIndividual]:
        from evo_rhyme.generator import generate_random_couplets
        return generate_random_couplets(
            theme_keywords=theme_keywords,
            count=size,
            analyze=True,
        )


def create_initial_population(
    generator: object,
    theme_keywords: Optional[List[str]] = None,
    size: int = 50,
) -> List[CoupletIndividual]:
    """
    Create initial population of couplets using the provided generator.

    Args:
        generator: Object with generate_seed_couplets(theme_keywords, size) method.
        theme_keywords: Optional list of theme keywords for semantic bias.
        size: Population size.

    Returns:
        List of CoupletIndividual (unanalyzed, no fitness).
    """
    gen_method = getattr(generator, "generate_seed_couplets", None)
    if gen_method is None:
        raise TypeError(
            "generator must have generate_seed_couplets(theme_keywords, size) method"
        )
    return gen_method(theme_keywords=theme_keywords, size=size)


# ---------------------------------------------------------------------------
# LM-based verse seed generation
# ---------------------------------------------------------------------------


class LMVerseSeedGenerator:
    """Generate verse seeds using LM-based bar proposal."""

    def __init__(
        self,
        theme_keywords: List[str],
        scheme: str = "AABB",
        num_lines: int = 4,
        proposer_config: Optional[Dict] = None,
        roles: Optional[List[str]] = None,
        constraint_config: Optional[Any] = None,
    ):
        """
        Args:
            theme_keywords: Theme words for conditioning.
            scheme: Rhyme scheme string.
            num_lines: Lines per verse (4, 8, or 16).
            proposer_config: Dict of ProposerConfig overrides.
            roles: Per-line role assignments. Auto-assigned if None.
            constraint_config: ConstraintConfig for filtering.
        """
        from evo_rhyme.lm_proposer import BarProposer, ProposerConfig

        cfg = ProposerConfig()
        if proposer_config:
            for k, v in proposer_config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        self.proposer = BarProposer(cfg)
        self.theme_keywords = theme_keywords
        self.scheme = scheme
        self.num_lines = num_lines
        self.roles = roles
        self.constraint_config = constraint_config

    def generate(self, count: int) -> List[VerseIndividual]:
        """Generate *count* verse individuals using LM bar proposal.

        1. Call proposer.propose_verse_pool() to get per-slot candidate pools.
        2. Assemble verses by picking one bar per slot from the pools.
        3. Filter through passes_verse_constraints.
        4. Return up to *count* valid verses.
        """
        pool = self.proposer.propose_verse_pool(
            theme_keywords=self.theme_keywords,
            scheme=self.scheme,
            num_lines=self.num_lines,
            roles=self.roles,
        )

        if not pool or not all(pool.get(i) for i in range(self.num_lines)):
            logger.warning("LM proposer returned empty pools for some slots")
            return []

        from evo_rhyme.constraints import passes_verse_constraints

        verses: List[VerseIndividual] = []
        attempts = 0
        max_attempts = count * 5

        while len(verses) < count and attempts < max_attempts:
            attempts += 1
            lines: List[str] = []
            for i in range(self.num_lines):
                candidates = pool.get(i, [])
                if not candidates:
                    break
                lines.append(random.choice(candidates))

            if len(lines) != self.num_lines:
                continue

            verse = create_verse_individual(
                lines=lines,
                scheme=self.scheme,
                roles=self.roles,
            )

            if not passes_verse_constraints(verse, config=self.constraint_config):
                continue

            verses.append(verse)

        logger.info(
            "LM seed generator: %d verses from %d attempts", len(verses), attempts
        )
        return verses


# ---------------------------------------------------------------------------
# Verse population (4-line)
# ---------------------------------------------------------------------------


class VerseSeedGenerator:
    """
    Generator that produces seed verses (4 lines).
    Uses couplet generator twice: lines 1-2 from first couplet, 3-4 from second.
    Supports "mixed", "random", "template", and "lm" init modes.
    """

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        init_mode: str = "mixed",
        scheme: str = "AABB",
        num_lines: int = 4,
        proposer_config: Optional[Dict] = None,
        roles: Optional[List[str]] = None,
        constraint_config: Optional[Any] = None,
    ):
        self.corpus_path = corpus_path
        self.init_mode = init_mode  # "mixed" | "random" | "template" | "lm"
        self.scheme = scheme
        self.num_lines = num_lines
        self.proposer_config = proposer_config
        self.roles = roles
        self.constraint_config = constraint_config

    def _build_lm_generator(
        self, theme_keywords: List[str]
    ) -> Optional[LMVerseSeedGenerator]:
        """Try to build an LMVerseSeedGenerator; return None on failure."""
        try:
            return LMVerseSeedGenerator(
                theme_keywords=theme_keywords,
                scheme=self.scheme,
                num_lines=self.num_lines,
                proposer_config=self.proposer_config,
                roles=self.roles,
                constraint_config=self.constraint_config,
            )
        except Exception as exc:
            logger.warning("Could not initialise LM proposer: %s", exc)
            return None

    def generate_seed_verses(
        self,
        theme_keywords: Optional[List[str]] = None,
        size: int = 50,
    ) -> List[VerseIndividual]:
        """
        Generate initial population of 4-line verses.
        Each verse = 2 couplets (lines 1-2 from first, 3-4 from second).
        """
        # ---- LM-only mode ------------------------------------------------
        if self.init_mode == "lm":
            if not theme_keywords:
                logger.warning("LM mode requires theme_keywords; falling back to mixed")
            else:
                lm_gen = self._build_lm_generator(theme_keywords)
                if lm_gen is not None:
                    verses = lm_gen.generate(size)
                    if verses:
                        return verses[:size]
                    logger.warning(
                        "LM seed generator produced 0 verses; falling back to mixed"
                    )
                # fall through to mixed if LM init failed

        # ---- Mixed mode with optional LM blend ----------------------------
        if self.init_mode == "mixed" and self.proposer_config and theme_keywords:
            lm_gen = self._build_lm_generator(theme_keywords)
            if lm_gen is not None:
                n_lm = int(size * 0.4)
                n_corpus_couplets = int(size * 0.3) * 2
                n_template_couplets = (size - n_lm - int(size * 0.3)) * 2

                lm_verses = lm_gen.generate(n_lm)

                from evo_rhyme.generator import template_fill_couplets
                from evo_rhyme.seed_generator import SeedGenerator

                corpus_gen = SeedGenerator(corpus_path=self.corpus_path)
                corpus_lines = corpus_gen._get_lines()
                has_corpus = len(corpus_lines) >= 2

                couplet_verses: List[VerseIndividual] = []

                if has_corpus and n_corpus_couplets > 0:
                    corpus_couplets = corpus_gen.generate_seed_couplets(
                        theme_keywords=theme_keywords,
                        size=n_corpus_couplets,
                    )
                    couplet_verses.extend(
                        self._couplets_to_verses(corpus_couplets)
                    )

                if n_template_couplets > 0:
                    tmpl_couplets = template_fill_couplets(
                        theme_keywords=theme_keywords,
                        count=n_template_couplets,
                        analyze=False,
                    )
                    couplet_verses.extend(
                        self._couplets_to_verses(tmpl_couplets)
                    )

                result = list(lm_verses) + couplet_verses
                random.shuffle(result)

                if len(result) < size:
                    extra_couplets = template_fill_couplets(
                        theme_keywords=theme_keywords,
                        count=(size - len(result)) * 2,
                        analyze=False,
                    )
                    result.extend(self._couplets_to_verses(extra_couplets))

                return result[:size]

        # ---- Original modes: template / random / mixed --------------------
        from evo_rhyme.generator import generate_random_couplets, template_fill_couplets
        from evo_rhyme.seed_generator import SeedGenerator

        couplets_needed = size * 2

        if self.init_mode == "template":
            couplets = template_fill_couplets(
                theme_keywords=theme_keywords,
                count=couplets_needed,
                analyze=False,
            )
        elif self.init_mode == "random":
            couplets = generate_random_couplets(
                theme_keywords=theme_keywords,
                count=couplets_needed,
                analyze=False,
            )
        else:
            raw = create_mixed_population(
                corpus_path=self.corpus_path,
                theme_keywords=theme_keywords,
                size=couplets_needed,
                analyze=False,
            )
            couplets = raw

        return self._couplets_to_verses(couplets)[:size]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _couplets_to_verses(
        couplets: List[CoupletIndividual],
    ) -> List[VerseIndividual]:
        """Pair sequential couplets into 4-line VerseIndividuals."""
        verses: List[VerseIndividual] = []
        for i in range(0, len(couplets) - 1, 2):
            c1, c2 = couplets[i], couplets[i + 1]
            lines = [c1.line1, c1.line2, c2.line1, c2.line2]
            verses.append(VerseIndividual(lines=lines))
        return verses


class VersePopulation:
    """Container for verse population with creation helpers."""

    def __init__(
        self,
        individuals: Optional[List[VerseIndividual]] = None,
        size: int = 80,
    ):
        self.individuals = individuals or []
        self.size = size

    @classmethod
    def create(
        cls,
        theme_keywords: Optional[List[str]] = None,
        size: int = 80,
        corpus_path: Optional[Path] = None,
        init_mode: str = "mixed",
        scheme: str = "AABB",
        num_lines: int = 4,
        proposer_config: Optional[Dict] = None,
        roles: Optional[List[str]] = None,
        constraint_config: Optional[Any] = None,
    ) -> "VersePopulation":
        """Create initial verse population using VerseSeedGenerator."""
        gen = VerseSeedGenerator(
            corpus_path=corpus_path,
            init_mode=init_mode,
            scheme=scheme,
            num_lines=num_lines,
            proposer_config=proposer_config,
            roles=roles,
            constraint_config=constraint_config,
        )
        individuals = gen.generate_seed_verses(
            theme_keywords=theme_keywords,
            size=size,
        )
        return cls(individuals=individuals, size=size)


def create_initial_verse_population(
    theme_keywords: Optional[List[str]] = None,
    size: int = 80,
    corpus_path: Optional[Path] = None,
    init_mode: str = "mixed",
    scheme: str = "AABB",
    num_lines: int = 4,
    proposer_config: Optional[Dict] = None,
    roles: Optional[List[str]] = None,
    constraint_config: Optional[Any] = None,
) -> List[VerseIndividual]:
    """
    Create initial population of 4-line verses.

    Args:
        theme_keywords: Optional list of theme keywords.
        size: Population size.
        corpus_path: Optional corpus path for mixed init.
        init_mode: "mixed" | "random" | "template" | "lm"
        scheme: Rhyme scheme string (e.g. "AABB").
        num_lines: Lines per verse.
        proposer_config: Dict of ProposerConfig overrides for LM mode.
        roles: Per-line role assignments.
        constraint_config: ConstraintConfig or dict for filtering.

    Returns:
        List of VerseIndividual (unanalyzed, no fitness).
    """
    pop = VersePopulation.create(
        theme_keywords=theme_keywords,
        size=size,
        corpus_path=corpus_path,
        init_mode=init_mode,
        scheme=scheme,
        num_lines=num_lines,
        proposer_config=proposer_config,
        roles=roles,
        constraint_config=constraint_config,
    )
    return pop.individuals
