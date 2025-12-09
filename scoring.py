# scoring.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import math
import torch
import torch.nn.functional as F


# =========================
#  Data Structures
# =========================

@dataclass
class ScoreWeights:
    lm: float = 1.0
    rhyme: float = 1.0
    internal_rhyme: float = 0.7
    theme: float = 0.7
    length: float = 0.3
    syllable_match: float = 0.3


@dataclass
class LengthModel:
    """
    Simple discrete length model over number of tokens or words.
    log_probs: dict[length] -> log P(length)
    """
    log_probs: Dict[int, float]
    default_log_prob: float

    def score(self, length: int) -> float:
        return self.log_probs.get(length, self.default_log_prob)


@dataclass
class ThemeContext:
    """
    Holds a precomputed embedding of the 'theme' (seed text, seed song, etc.).
    You can build this once per verse/song using any encoder.
    """
    embedding: torch.Tensor  # shape [d]


@dataclass
class LineFeatures:
    """
    All numeric features we care about for one candidate line.
    """
    text: str
    length: int                     # in tokens or words (your choice, but be consistent)
    last_word: str
    lm_logprob: float               # average logprob per token
    rhyme_score_end: float          # rhyme with previous line's last word
    internal_rhyme_score: float     # internal rhyme density
    theme_similarity: float         # cosine sim with theme embedding
    length_prior_logprob: float     # log P(length)
    syllable_match_score: float     # [0, 1] measure of syllable match vs previous line


# =========================
#  Helper: Length Model
# =========================

def build_length_model(length_counts: Dict[int, int]) -> LengthModel:
    """
    Build a LengthModel from counts of lengths extracted from corpus.
    length_counts: e.g. {9: 1200, 8: 900, 10: 750, ...}
    """
    total = sum(length_counts.values())
    if total == 0:
        # fallback uniform in reasonable range
        log_probs = {l: math.log(1.0 / len(length_counts)) for l in length_counts} or {8: 0.0}
        return LengthModel(log_probs=log_probs, default_log_prob=math.log(1e-4))
    log_probs = {}
    for l, c in length_counts.items():
        p = max(c / total, 1e-8)
        log_probs[l] = math.log(p)
    default_log_prob = math.log(1e-4)
    return LengthModel(log_probs=log_probs, default_log_prob=default_log_prob)


# =========================
#  Helper: LM logprob
# =========================

@torch.no_grad()
def compute_lm_logprob(
    model,
    tokenizer,
    text: str,
    device: str = "cuda"
) -> float:
    """
    Compute average logprob per token for the given text under the LM.
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Standard LM loss uses next-token prediction.
    # We shift labels by 1.
    labels = input_ids.clone()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # out.loss is mean cross-entropy; convert to avg logprob
    neg_log_likelihood = out.loss.item()
    avg_logprob = -neg_log_likelihood  # log p per token (approx)
    return avg_logprob


# =========================
#  Helper: Theme similarity
# =========================

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return float((x * y).sum().item())


def compute_theme_similarity(
    line_embedding: torch.Tensor,
    theme_ctx: Optional[ThemeContext]
) -> float:
    if theme_ctx is None:
        return 0.0
    return cosine_similarity(line_embedding, theme_ctx.embedding)


# =========================
#  Core Hybrid Scoring
# =========================

def hybrid_score(
    features: LineFeatures,
    weights: ScoreWeights
) -> float:
    """
    Combine all features into a single scalar score.
    Higher is better.
    """
    score = 0.0

    # LM fluency (already average logprob)
    score += weights.lm * features.lm_logprob

    # End rhyme with previous line
    score += weights.rhyme * features.rhyme_score_end

    # Internal rhyme density
    score += weights.internal_rhyme * features.internal_rhyme_score

    # Theme similarity
    score += weights.theme * features.theme_similarity

    # Length prior from empirical distribution
    score += weights.length * features.length_prior_logprob

    # Syllable match score [0, 1]
    score += weights.syllable_match * features.syllable_match_score

    return score


# =========================
#  Feature Construction Hooks
# =========================

def build_line_features(
    text: str,
    last_word: str,
    length: int,
    length_model: LengthModel,
    lm_logprob: float,
    rhyme_score_end: float,
    internal_rhyme_score: float,
    theme_similarity: float,
    syllable_match_score: float
) -> LineFeatures:
    """
    Convenience helper to assemble LineFeatures.
    You will usually call this from your generation loop
    after you have computed each sub-score.
    """
    length_prior_logprob = length_model.score(length)
    return LineFeatures(
        text=text,
        length=length,
        last_word=last_word,
        lm_logprob=lm_logprob,
        rhyme_score_end=rhyme_score_end,
        internal_rhyme_score=internal_rhyme_score,
        theme_similarity=theme_similarity,
        length_prior_logprob=length_prior_logprob,
        syllable_match_score=syllable_match_score,
    )
