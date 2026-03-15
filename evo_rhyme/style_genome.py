"""
Style genome representation for rap-style evolution.

This module keeps style genes compact and mutation-friendly, while exposing
human-readable labels for logging and prompt building.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


STYLE_GENE_VALUES: Dict[str, List[str]] = {
    "tone": ["reflective", "calm", "confident", "aggressive", "chaotic"],
    "narrativity": ["low", "medium", "high"],
    "internal_rhyme_density": ["low", "medium", "high", "very_high"],
    "multisyllabic_rhyme_density": ["none", "light", "medium", "dense"],
    "metaphor_density": ["literal", "some", "rich", "very_rich"],
    "syllable_density": ["sparse", "normal", "dense", "very_dense"],
    "rhyme_scheme": ["AABB", "ABAB", "ABBA", "AAAA", "ABCB", "freeform"],
    "imagery_mode": ["street", "dark", "mythic", "industrial", "personal", "surreal"],
}


@dataclass(frozen=True)
class StyleGenome:
    tone: int
    narrativity: int
    internal_rhyme_density: int
    multisyllabic_rhyme_density: int
    metaphor_density: int
    syllable_density: int
    rhyme_scheme: int
    imagery_mode: int

    def to_labels(self) -> Dict[str, str]:
        return {
            "tone": STYLE_GENE_VALUES["tone"][self.tone],
            "narrativity": STYLE_GENE_VALUES["narrativity"][self.narrativity],
            "internal_rhyme_density": STYLE_GENE_VALUES["internal_rhyme_density"][self.internal_rhyme_density],
            "multisyllabic_rhyme_density": STYLE_GENE_VALUES["multisyllabic_rhyme_density"][self.multisyllabic_rhyme_density],
            "metaphor_density": STYLE_GENE_VALUES["metaphor_density"][self.metaphor_density],
            "syllable_density": STYLE_GENE_VALUES["syllable_density"][self.syllable_density],
            "rhyme_scheme": STYLE_GENE_VALUES["rhyme_scheme"][self.rhyme_scheme],
            "imagery_mode": STYLE_GENE_VALUES["imagery_mode"][self.imagery_mode],
        }

    def to_dict(self) -> Dict[str, int]:
        return {
            "tone": self.tone,
            "narrativity": self.narrativity,
            "internal_rhyme_density": self.internal_rhyme_density,
            "multisyllabic_rhyme_density": self.multisyllabic_rhyme_density,
            "metaphor_density": self.metaphor_density,
            "syllable_density": self.syllable_density,
            "rhyme_scheme": self.rhyme_scheme,
            "imagery_mode": self.imagery_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "StyleGenome":
        return cls(
            tone=int(data.get("tone", 0)),
            narrativity=int(data.get("narrativity", 1)),
            internal_rhyme_density=int(data.get("internal_rhyme_density", 1)),
            multisyllabic_rhyme_density=int(data.get("multisyllabic_rhyme_density", 1)),
            metaphor_density=int(data.get("metaphor_density", 1)),
            syllable_density=int(data.get("syllable_density", 1)),
            rhyme_scheme=int(data.get("rhyme_scheme", 0)),
            imagery_mode=int(data.get("imagery_mode", 0)),
        )


def random_style_genome() -> StyleGenome:
    return StyleGenome(
        tone=random.randrange(len(STYLE_GENE_VALUES["tone"])),
        narrativity=random.randrange(len(STYLE_GENE_VALUES["narrativity"])),
        internal_rhyme_density=random.randrange(len(STYLE_GENE_VALUES["internal_rhyme_density"])),
        multisyllabic_rhyme_density=random.randrange(len(STYLE_GENE_VALUES["multisyllabic_rhyme_density"])),
        metaphor_density=random.randrange(len(STYLE_GENE_VALUES["metaphor_density"])),
        syllable_density=random.randrange(len(STYLE_GENE_VALUES["syllable_density"])),
        rhyme_scheme=random.randrange(len(STYLE_GENE_VALUES["rhyme_scheme"])),
        imagery_mode=random.randrange(len(STYLE_GENE_VALUES["imagery_mode"])),
    )


def mutate_style_genome(genome: StyleGenome, mutation_rate: float = 0.25) -> StyleGenome:
    data = genome.to_dict()
    for key, values in STYLE_GENE_VALUES.items():
        if random.random() < mutation_rate:
            current = int(data[key])
            if len(values) <= 1:
                continue
            candidates = [i for i in range(len(values)) if i != current]
            data[key] = random.choice(candidates)
    return StyleGenome.from_dict(data)


def crossover_style_genome(a: StyleGenome, b: StyleGenome) -> StyleGenome:
    ad = a.to_dict()
    bd = b.to_dict()
    out = {}
    for key in ad:
        out[key] = ad[key] if random.random() < 0.5 else bd[key]
    return StyleGenome.from_dict(out)


def style_to_prompt_directives(genome: StyleGenome) -> Dict[str, str]:
    labels = genome.to_labels()
    syllable_map = {
        "sparse": "8-10",
        "normal": "10-12",
        "dense": "12-14",
        "very_dense": "13-16",
    }
    return {
        "tone": labels["tone"],
        "narrativity": labels["narrativity"],
        "internal_rhyme_density": labels["internal_rhyme_density"],
        "multisyllabic_rhyme_density": labels["multisyllabic_rhyme_density"],
        "metaphor_density": labels["metaphor_density"],
        "syllable_target": syllable_map.get(labels["syllable_density"], "10-13"),
        "rhyme_scheme": labels["rhyme_scheme"],
        "imagery_mode": labels["imagery_mode"],
    }

