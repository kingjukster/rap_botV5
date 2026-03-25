"""Tests for initial population diversity improvements."""

from __future__ import annotations

import random
from unittest.mock import patch

from evo_rhyme.generator import generate_seed_couplets, generate_random_couplets
from evo_rhyme.individual import CoupletIndividual
from evo_rhyme.population import VerseSeedGenerator
from evo_rhyme.seed_generator import SeedGenerator


def _fingerprint(lines: list[str]) -> tuple:
    return tuple(" ".join(l.lower().split()[:3]) for l in lines)


class TestCorpusRandomPairing:
    """SeedGenerator should sample lines independently, not consecutively."""

    def test_no_consecutive_bias(self):
        gen = SeedGenerator()
        lines = gen._get_lines()
        if len(lines) < 10:
            return
        couplets = gen.generate_seed_couplets(size=30)
        consecutive_count = 0
        for c in couplets:
            for i in range(len(lines) - 1):
                if lines[i].strip() == c.line1 and lines[i + 1].strip() == c.line2:
                    consecutive_count += 1
                    break
        assert consecutive_count < len(couplets) * 0.5, (
            f"Too many consecutive pairs: {consecutive_count}/{len(couplets)}"
        )

    def test_fingerprint_dedup(self):
        gen = SeedGenerator()
        couplets = gen.generate_seed_couplets(size=40)
        fps = set()
        for c in couplets:
            fp = (
                " ".join(c.line1.lower().split()[:3]),
                " ".join(c.line2.lower().split()[:3]),
            )
            fps.add(fp)
        assert len(fps) == len(couplets), "Duplicate fingerprints in corpus couplets"


class TestTemplateRoundRobin:
    """Template-based generation should spread across available templates."""

    def test_template_spread(self):
        random.seed(42)
        couplets = generate_seed_couplets(
            theme_keywords=["pressure", "mask"],
            count=30,
            analyze=False,
        )
        assert len(couplets) == 30
        line_starts = set()
        for c in couplets:
            line_starts.add(" ".join(c.line1.split()[:3]))
            line_starts.add(" ".join(c.line2.split()[:3]))
        assert len(line_starts) >= 15, (
            f"Only {len(line_starts)} distinct line openings from 30 couplets "
            f"-- templates are not spread enough"
        )

    def test_random_couplets_spread(self):
        random.seed(42)
        couplets = generate_random_couplets(
            theme_keywords=["flow", "dream"],
            count=30,
            analyze=False,
        )
        assert len(couplets) == 30
        line_starts = set()
        for c in couplets:
            line_starts.add(" ".join(c.line1.split()[:3]))
            line_starts.add(" ".join(c.line2.split()[:3]))
        assert len(line_starts) >= 15


class TestVerseSeedDedup:
    """VerseSeedGenerator._couplets_to_verses should deduplicate."""

    def test_no_duplicate_verses(self):
        dup = CoupletIndividual(line1="same line one here", line2="same line two here")
        couplets = [dup] * 20
        verses = VerseSeedGenerator._couplets_to_verses(couplets)
        assert len(verses) <= 1, f"Expected <=1 unique verse, got {len(verses)}"

    def test_diverse_input_preserved(self):
        words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "golf", "hotel", "india", "juliet", "kilo", "lima",
            "mike", "november", "oscar", "papa", "quebec", "romeo",
            "sierra", "tango",
        ]
        couplets = [
            CoupletIndividual(
                line1=f"{words[i]} walks through the night",
                line2=f"{words[i]} talks through the light",
            )
            for i in range(20)
        ]
        verses = VerseSeedGenerator._couplets_to_verses(couplets)
        assert len(verses) >= 8, f"Too few verses from 20 diverse couplets: {len(verses)}"
        fps = set()
        for v in verses:
            fps.add(_fingerprint(v.lines))
        assert len(fps) == len(verses), "Duplicate fingerprints in verse output"
