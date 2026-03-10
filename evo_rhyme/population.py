"""
evo_rhyme/population.py

Population initialization for evolutionary rhyme. Uses a generator to create
seed couplets for the initial population.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from evo_rhyme.individual import CoupletIndividual, VerseIndividual


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
# Verse population (4-line)
# ---------------------------------------------------------------------------


class VerseSeedGenerator:
    """
    Generator that produces seed verses (4 lines).
    Uses couplet generator twice: lines 1-2 from first couplet, 3-4 from second.
    """

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        init_mode: str = "mixed",
    ):
        self.corpus_path = corpus_path
        self.init_mode = init_mode  # "mixed" | "random" | "template"

    def generate_seed_verses(
        self,
        theme_keywords: Optional[List[str]] = None,
        size: int = 50,
    ) -> List[VerseIndividual]:
        """
        Generate initial population of 4-line verses.
        Each verse = 2 couplets (lines 1-2 from first, 3-4 from second).
        """
        from evo_rhyme.generator import generate_random_couplets, template_fill_couplets
        from evo_rhyme.seed_generator import SeedGenerator

        couplets_needed = size * 2  # 2 couplets per verse

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

        verses: List[VerseIndividual] = []
        for i in range(0, min(len(couplets) - 1, couplets_needed - 1), 2):
            c1, c2 = couplets[i], couplets[i + 1]
            lines = [c1.line1, c1.line2, c2.line1, c2.line2]
            verses.append(VerseIndividual(lines=lines))
            if len(verses) >= size:
                break

        return verses[:size]


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
    ) -> "VersePopulation":
        """Create initial verse population using VerseSeedGenerator."""
        gen = VerseSeedGenerator(corpus_path=corpus_path, init_mode=init_mode)
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
) -> List[VerseIndividual]:
    """
    Create initial population of 4-line verses.

    Args:
        theme_keywords: Optional list of theme keywords.
        size: Population size.
        corpus_path: Optional corpus path for mixed init.
        init_mode: "mixed" | "random" | "template"

    Returns:
        List of VerseIndividual (unanalyzed, no fitness).
    """
    pop = VersePopulation.create(
        theme_keywords=theme_keywords,
        size=size,
        corpus_path=corpus_path,
        init_mode=init_mode,
    )
    return pop.individuals
