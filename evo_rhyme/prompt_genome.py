"""
Prompt genome representation for neuroevolution of prompting strategy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromptGenome:
    strictness: float = 0.5
    novelty_bias: float = 0.5
    metaphor_boost: float = 0.5
    internal_rhyme_boost: float = 0.5
    punchline_bias: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "strictness": float(self.strictness),
            "novelty_bias": float(self.novelty_bias),
            "metaphor_boost": float(self.metaphor_boost),
            "internal_rhyme_boost": float(self.internal_rhyme_boost),
            "punchline_bias": float(self.punchline_bias),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PromptGenome":
        return cls(
            strictness=float(data.get("strictness", 0.5)),
            novelty_bias=float(data.get("novelty_bias", 0.5)),
            metaphor_boost=float(data.get("metaphor_boost", 0.5)),
            internal_rhyme_boost=float(data.get("internal_rhyme_boost", 0.5)),
            punchline_bias=float(data.get("punchline_bias", 0.5)),
        )


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def random_prompt_genome() -> PromptGenome:
    return PromptGenome(
        strictness=random.random(),
        novelty_bias=random.random(),
        metaphor_boost=random.random(),
        internal_rhyme_boost=random.random(),
        punchline_bias=random.random(),
    )


def mutate_prompt_genome(genome: PromptGenome, mutation_scale: float = 0.18) -> PromptGenome:
    d = genome.to_dict()
    for k, v in d.items():
        if random.random() < 0.35:
            d[k] = _clip(v + random.uniform(-mutation_scale, mutation_scale))
    return PromptGenome.from_dict(d)


def crossover_prompt_genome(a: PromptGenome, b: PromptGenome) -> PromptGenome:
    ad = a.to_dict()
    bd = b.to_dict()
    out = {}
    for k in ad:
        out[k] = ad[k] if random.random() < 0.5 else bd[k]
    return PromptGenome.from_dict(out)


def prompt_directives(genome: PromptGenome) -> Dict[str, str]:
    strictness_text = (
        "strictly obey structure and syllable targets"
        if genome.strictness > 0.65
        else "allow moderate flexibility in structure"
    )
    novelty_text = (
        "prioritize fresh phrasing and avoid repeated cliches"
        if genome.novelty_bias > 0.6
        else "keep phrasing clear and direct"
    )
    return {
        "strictness": strictness_text,
        "novelty": novelty_text,
        "metaphor": f"metaphor emphasis {genome.metaphor_boost:.2f}",
        "internal_rhyme": f"internal rhyme emphasis {genome.internal_rhyme_boost:.2f}",
        "punchline": f"punchline emphasis {genome.punchline_bias:.2f}",
    }

