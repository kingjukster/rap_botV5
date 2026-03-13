"""Punchline surprise scoring using LM perplexity contrast."""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_MODEL: Optional[Any] = None
_TOKENIZER: Optional[Any] = None
_LOADED_NAME: Optional[str] = None

TAIL_TOKENS = 3


def _ensure_loaded(model_name: str = "distilgpt2") -> None:
    """Lazy-load causal LM and tokenizer."""
    global _MODEL, _TOKENIZER, _LOADED_NAME
    if _MODEL is not None and _LOADED_NAME == model_name:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModelForCausalLM.from_pretrained(model_name)
    _MODEL.to(device)
    _MODEL.eval()
    _LOADED_NAME = model_name
    logger.info("Loaded punchline model: %s on %s", model_name, device)


def _surprise_ratio(line: str, model_name: str) -> Optional[float]:
    """
    Compute surprise ratio for a single line.

    Returns tail_nll / avg_nll, or None if the line is too short.
    """
    import torch
    import torch.nn.functional as F

    _ensure_loaded(model_name)
    assert _MODEL is not None and _TOKENIZER is not None

    enc = _TOKENIZER(line, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"].to(_MODEL.device)
    seq_len = input_ids.size(1)
    if seq_len < 3:
        return None

    with torch.no_grad():
        outputs = _MODEL(input_ids)
        logits = outputs.logits  # (1, T, V)

    # logits[0, t] predicts token at position t+1
    shift_logits = logits[0, :-1]  # (T-1, V)
    shift_labels = input_ids[0, 1:]  # (T-1,)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    token_nlls = -token_log_probs  # positive values

    n_tokens = token_nlls.size(0)
    if n_tokens < 2:
        return None

    avg_nll = float(token_nlls.mean().item())
    tail_count = min(TAIL_TOKENS, n_tokens)
    tail_nll = float(token_nlls[-tail_count:].mean().item())

    return tail_nll / max(avg_nll, 0.01)


def _sigmoid_scale(ratio: float) -> float:
    """Map surprise ratio to [0, 1] via sigmoid centred at 1.0."""
    return 1.0 / (1.0 + math.exp(-2.0 * (ratio - 1.0)))


def score_punchline(lines: List[str], model_name: str = "distilgpt2") -> float:
    """
    Score punchline strength across a verse.

    For each line, computes per-token log-probabilities. The "surprise score" for a line
    is the ratio of perplexity of the last 3 tokens vs. the full-line average.
    Higher contrast = more surprising ending = stronger punchline.

    Returns mean surprise across all lines, normalized to [0.0, 1.0].
    """
    try:
        ratios: List[float] = []
        for line in lines:
            line = (line or "").strip()
            if not line:
                continue
            r = _surprise_ratio(line, model_name)
            if r is not None:
                ratios.append(r)
        if not ratios:
            return 0.0
        mean_ratio = sum(ratios) / len(ratios)
        return max(0.0, min(1.0, _sigmoid_scale(mean_ratio)))
    except Exception:
        logger.warning("Punchline scoring failed", exc_info=True)
        return 0.0


def score_punchline_per_line(
    lines: List[str], model_name: str = "distilgpt2"
) -> List[float]:
    """Return per-line punchline surprise scores."""
    results: List[float] = []
    try:
        for line in lines:
            line = (line or "").strip()
            if not line:
                results.append(0.0)
                continue
            r = _surprise_ratio(line, model_name)
            if r is None:
                results.append(0.0)
            else:
                results.append(max(0.0, min(1.0, _sigmoid_scale(r))))
    except Exception:
        logger.warning("Punchline per-line scoring failed", exc_info=True)
        return [0.0] * len(lines)
    return results
