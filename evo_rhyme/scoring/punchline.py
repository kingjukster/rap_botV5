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


# Weights for 4-line verse: L1, L2, L3, L4 (last line dominates)
_WEIGHTS_4 = [0.2, 0.2, 0.2, 0.4]


def _get_line_weights(n: int) -> List[float]:
    """Return weights for n lines; last line always gets highest weight."""
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    if n == 2:
        return [0.2, 0.8]
    if n == 3:
        return [0.2, 0.2, 0.6]
    if n == 4:
        return _WEIGHTS_4.copy()
    # 5+ lines: 0.2 for first n-1, 0.4 for last; normalize to sum 1
    weights = [0.2] * (n - 1) + [0.4]
    total = sum(weights)
    return [w / total for w in weights]


def score_punchline(lines: List[str], model_name: str = "distilgpt2") -> float:
    """
    Score punchline strength across a verse.

    Uses weighted mean of per-line surprise scores: 0.2*L1 + 0.2*L2 + 0.2*L3 + 0.4*L4
    so the final line (line 4) dominates. For fewer than 4 lines, weights are applied
    proportionally (last line always gets highest weight). Empty lines get 0.0 from
    score_punchline_per_line and are included in the weighted sum.

    Returns weighted mean surprise, normalized to [0.0, 1.0].
    """
    try:
        per_line = score_punchline_per_line(lines, model_name)
        if not per_line:
            return 0.0
        weights = _get_line_weights(len(per_line))
        if not weights:
            return 0.0
        weighted_sum = sum(w * s for w, s in zip(weights, per_line))
        return max(0.0, min(1.0, weighted_sum))
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


def score_punchline_per_line_batch(
    all_lines: List[str], model_name: str = "distilgpt2", batch_size: int = 64,
) -> List[float]:
    """Batch-score punchline surprise for many lines in fewer GPU passes."""
    import torch
    import torch.nn.functional as F

    _ensure_loaded(model_name)
    assert _MODEL is not None and _TOKENIZER is not None
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    results = [0.0] * len(all_lines)
    valid: List[tuple] = []
    for i, line in enumerate(all_lines):
        line = (line or "").strip()
        if line:
            valid.append((i, line))

    if not valid:
        return results

    for bs in range(0, len(valid), batch_size):
        batch = valid[bs : bs + batch_size]
        batch_texts = [t for _, t in batch]

        enc = _TOKENIZER(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=128, padding=True,
        )
        input_ids = enc["input_ids"].to(_MODEL.device)
        attn = enc["attention_mask"].to(_MODEL.device)

        with torch.no_grad():
            logits = _MODEL(input_ids, attention_mask=attn).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attn[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_nlls = -log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        token_nlls = token_nlls * shift_mask.float()

        for j, (orig_idx, _) in enumerate(batch):
            mask_j = shift_mask[j].bool()
            nlls_j = token_nlls[j][mask_j]
            n_tokens = nlls_j.size(0)
            if n_tokens < 2:
                continue
            avg_nll = float(nlls_j.mean().item())
            tail_count = min(TAIL_TOKENS, n_tokens)
            tail_nll = float(nlls_j[-tail_count:].mean().item())
            ratio = tail_nll / max(avg_nll, 0.01)
            results[orig_idx] = max(0.0, min(1.0, _sigmoid_scale(ratio)))

    return results


def score_punchlines_batch(
    verses_lines: List[List[str]], model_name: str = "distilgpt2",
) -> List[float]:
    """Batch-score punchline for multiple verses at once.

    Args:
        verses_lines: list of verse line-lists, e.g. [[l1,l2,l3,l4], [l1,l2,l3,l4], ...]

    Returns:
        One punchline score per verse.
    """
    all_lines: List[str] = []
    verse_spans: List[tuple] = []
    for vlines in verses_lines:
        start = len(all_lines)
        all_lines.extend(vlines)
        verse_spans.append((start, start + len(vlines)))

    if not all_lines:
        return [0.0] * len(verses_lines)

    per_line_scores = score_punchline_per_line_batch(all_lines, model_name)

    verse_scores: List[float] = []
    for start, end in verse_spans:
        line_scores = per_line_scores[start:end]
        if not line_scores:
            verse_scores.append(0.0)
            continue
        weights = _get_line_weights(len(line_scores))
        weighted = sum(w * s for w, s in zip(weights, line_scores))
        verse_scores.append(max(0.0, min(1.0, weighted)))

    return verse_scores
