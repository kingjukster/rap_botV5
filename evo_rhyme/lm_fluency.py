"""
evo_rhyme/lm_fluency.py

Perplexity-based language model fluency scoring. Uses a small transformer (DistilGPT-2)
to score lines by phrase plausibility - nonsense sequences get high perplexity.
Complements corpus n-grams: ngrams catch "appears in rap corpus", LM catches
"grammatically/fluently plausible in general language".
"""

from __future__ import annotations

import math
from typing import Optional

# Perplexity -> [0,1] mapping. General English: ppl ~20-80. Rap/slang: ppl 80-200.
# Nonsense: ppl 200-500+. score = exp(-ppl/scale).
# scale=150: ppl=100->0.51, ppl=200->0.26, ppl=400->0.07 (nonsense)
PPL_SCALE = 150.0
PPL_CAP = 600.0  # cap for numerical stability


def _ppl_to_score(perplexity: float) -> float:
    """Map perplexity to [0,1]. Lower perplexity = higher (more natural) score."""
    ppl = min(float(perplexity), PPL_CAP)
    return math.exp(-ppl / PPL_SCALE)


class LMPerplexityScorer:
    """
    Scores lines by perplexity from a causal LM (DistilGPT-2).
    Lower perplexity = more natural phrase structure.
    """

    def __init__(self, model_id: str = "distilgpt2", device: Optional[str] = None):
        self._model = None
        self._tokenizer = None
        self._model_id = model_id
        self._device = device
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_id)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load LM for fluency scoring: {e}") from e

    def score_line(self, text: str) -> float:
        """
        Score a single line [0,1] by perplexity.
        Returns 0.5 (neutral) for empty or very short input.
        """
        text = (text or "").strip()
        if not text or len(text.split()) < 2:
            return 0.5

        self._ensure_loaded()
        import torch

        with torch.no_grad():
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            input_ids = enc["input_ids"].to(self._model.device)
            if input_ids.size(1) < 2:
                return 0.5
            labels = input_ids.clone()
            outputs = self._model(input_ids, labels=labels)
            loss = outputs.loss.item()
            ppl = math.exp(loss)

        return _ppl_to_score(ppl)

    def score_couplet(self, line1: str, line2: str) -> float:
        """Average score across both lines."""
        s1 = self.score_line(line1)
        s2 = self.score_line(line2)
        return (s1 + s2) / 2.0


# Module-level cache
_CACHED_SCORER: Optional[LMPerplexityScorer] = None


def get_lm_scorer(
    model_id: str = "distilgpt2",
    device: Optional[str] = None,
) -> Optional[LMPerplexityScorer]:
    """
    Get or create cached LM perplexity scorer.
    Returns None if transformers/torch not available.
    """
    global _CACHED_SCORER
    try:
        if _CACHED_SCORER is None:
            _CACHED_SCORER = LMPerplexityScorer(model_id=model_id, device=device)
            _CACHED_SCORER._ensure_loaded()
        return _CACHED_SCORER
    except Exception:
        return None


def clear_lm_cache() -> None:
    """Clear cached scorer (e.g. between runs)."""
    global _CACHED_SCORER
    _CACHED_SCORER = None
