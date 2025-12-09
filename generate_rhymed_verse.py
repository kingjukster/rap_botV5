#!/usr/bin/env python
"""
generate_rhymed_verse.py

Generate a multi-bar verse with a rhyme scheme (e.g. AAAA, AABB, ABAB)
using your QLoRA-trained Qwen2.5-7B-Instruct model + LoRA adapter.

Hybrid scoring:
- Original structural score (rhyme groups, internal multis, coherence, etc.).
- PLUS (when enabled): LM logprob, length prior, theme embedding similarity, syllable match.

New enhancements (Hernandez-style):
- Rhyme planning using rhyme_planner.plan_rhyme_endings (anchor end words per line).
- Topic similarity via Word2Vec (TopicScorer).
- N-gram critic for "organic"/non-weird phrasing (NgramCritic).
- Meter scoring via meter_utils.meter_score (syllable control at bar level).

Toggle hybrid LM-related scoring with:
  --no_hybrid    # disables hybrid LM-based scoring and uses structural + Hernandez-style only.
"""

import argparse
import json
import math
import os
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config.settings import load_settings
from rhyme_planner import load_rhyme_groups as load_rhyme_groups_planner, plan_rhyme_endings
from meter_utils import meter_score
from topic_utils import TopicScorer
from ngram_critic import NgramCritic

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModel,
)
from peft import PeftModel

import pronouncing
import pandas as pd

# --- Hybrid scoring imports ---
from scoring import (
    ScoreWeights,
    LengthModel,
    ThemeContext,
    build_length_model,
    build_line_features,
    hybrid_score,
    compute_lm_logprob,
    compute_theme_similarity,
)

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CANDIDATES_PER_BAR = 8
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.05
VERSE_SIAMESE_THRESHOLD = 0.40
MAX_VERSE_ATTEMPTS = 4

# Structural weights
END_CHAIN_WEIGHT = 2.2
END_RHYME_WEIGHT = 1.6
INTERNAL_RHYME_WEIGHT = 2.8
COHERENCE_WEIGHT = 0.7
THEME_SIM_WEIGHT = 0.7
FILLER_PENALTY = 0.8
RHYME_GROUP_PRESENCE_WEIGHT = 0.4

# Bar length control (structural)
MAX_BAR_WORDS = 14
HARD_BAR_WORD_LIMIT = 20

REPEAT_ENDWORD_PENALTY = 0.5
REPEAT_ENDWORD_HARD_CAP = 2

# STRICT_RHYME_MODE controls filter_candidates_by_rhyme_letter.
# ENFORCE_STRICT_IN_SCORING controls whether candidate_score() also hard-rejects by rhyme.
STRICT_RHYME_MODE = True
ENFORCE_STRICT_IN_SCORING = False


PHRASE_BAN_LIST = [
    "demons in the mirror",
    "pressure on my chest",
]

MAX_PHRASE_REUSE = 1
PHRASE_REUSE_FACTOR = 0.05

STOP_PHRASES = [
    "yeah", "yo", "uh", "uhh", "uh-huh", "ya", "ya'll",
    "yeah yeah", "haha", "ha ha", "skrrt", "skrr",
    "i'ma do me", "ima do me", "imma do me",
    "we're the best", "we the best",
    "my nigga, my nigga",
]

STOP_END_WORDS = {
    "yeah", "yo", "uh", "uhh", "yah", "ayy", "ay", "hey",
    "whoa", "wow", "ha", "nah", "okay", "ok",
    "the", "a", "an", "and", "or", "but",
    "i", "me", "you", "ya", "we", "us", "they", "them",
    "he", "she", "it", "this", "that",
    "here", "there", "where",
    "of", "to", "for", "in", "on", "at", "by",
}

META_TOKENS = [
    "intro", "outro", "chorus", "hook", "bridge", "verse",
    "interlude", "refrain", "pre-chorus", "post-chorus",
    "instrumental", "skit", "spoken", "sample",
    "remix", "mix", "edit",
    "feat.", "featuring",
]

META_ARTIST_TAGS = [
    "kendrick lamar", "k-dot", "k dot", "eminem",
    "nas:", "jay-z", "black thought", "mf doom",
]

COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but",
    "i", "me", "you", "ya", "we", "us", "they", "them",
    "he", "she", "it", "this", "that",
    "here", "there", "where",
    "of", "to", "for", "in", "on", "at", "by",
}

META_REGEXES = [
    re.compile(r"\[[^\]]+\]"),
    re.compile(r"\(.*?\)"),
    re.compile(r"\bx\d+\b"),
    re.compile(r"\b(19|20)\d{2}\b"),
]

BAR_SPAN_RE = re.compile(r"\[BAR\]\s*(.*?)\s*\[RHY=", re.DOTALL)

RHYME_GROUPS: Dict[str, int] = {}
SIAMESE_SCORER = None

DEFAULT_LOG_FILENAME = "generated_raw.jsonl"

# -----------------------------------------------------------------------------
# Rhyme groups (word -> group_id mapping, for structural scoring)
# -----------------------------------------------------------------------------

@dataclass
class RhymeGroupEntry:
    word: str
    group: int


def load_word_rhyme_groups(csv_path: str) -> Dict[str, int]:
    """
    Local loader: map word -> integer group id, used by structural scoring.

    This is separate from rhyme_planner.load_rhyme_groups, which returns
    group_id -> list[word] for anchor planning.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] rhyme group CSV not found at {csv_path}, continuing without.")
        return {}
    df = pd.read_csv(csv_path)
    mapping: Dict[str, int] = {}
    for _, row in df.iterrows():
        w = str(row["word"]).strip().lower()
        g = int(row["group"])
        if w:
            mapping[w] = g
    print(f"__ Loaded {len(mapping)} word->group rhyme entries from {csv_path}")
    return mapping


def get_rhyme_group(word: str):
    return RHYME_GROUPS.get(word.lower())


def rhyme_key(word: str, max_len: int = 5) -> str:
    w = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not w:
        return ""
    return w[-max_len:]


def rhyme_similarity(key1: str, key2: str) -> float:
    if not key1 or not key2:
        return 0.0
    max_len = min(len(key1), len(key2))
    if max_len == 0:
        return 0.0
    matches = 0
    for i in range(1, max_len + 1):
        if key1[-i:] == key2[-i:]:
            matches = i
        else:
            break
    return matches / max_len


def get_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_end_word(bar: str) -> str:
    words = get_words(bar)
    return words[-1] if words else ""


def count_internal_hits(bar: str) -> int:
    words = get_words(bar)
    if len(words) < 3:
        return 0
    keys = [rhyme_key(w) for w in words]
    counts = defaultdict(int)
    hits = 0
    for k in keys:
        if not k:
            continue
        counts[k] += 1
    for _, c in counts.items():
        if c > 1:
            hits += (c - 1)
    return hits


def extract_bar_text(decoded: str) -> str:
    m = BAR_SPAN_RE.search(decoded)
    if m:
        bar = m.group(1).strip()
    else:
        bar = ""
        for line in decoded.splitlines():
            line = line.strip()
            if line:
                bar = line
                break
        bar = bar.strip()

    bar = re.split(r"<\|", bar)[0]
    bar = bar.split("<sep>")[0]
    bar = bar.split("</s>")[0]
    # Drop any leftover analysis/meta fragments like "<end rhyme"
    bar = re.split(r"<end rhyme", bar, flags=re.IGNORECASE)[0].rstrip(" ,.;:/")
    bar = re.sub(r"<[^>]+>", "", bar)

    return bar.strip()


def bar_word_count(bar: str) -> int:
    return len(bar.strip().split())


def bar_length_factor(bar: str) -> float:
    n = bar_word_count(bar)
    if n <= MAX_BAR_WORDS:
        return 1.0
    if n >= HARD_BAR_WORD_LIMIT:
        return 0.0
    over = n - MAX_BAR_WORDS
    span = HARD_BAR_WORD_LIMIT - MAX_BAR_WORDS
    return max(0.1, 1.0 - over / span)


def end_chain_hits(bar: str, max_len: int = 5) -> int:
    words = get_words(bar)
    if len(words) < 3:
        return 0
    end = extract_end_word(bar)
    if not end:
        return 0
    end_k = rhyme_key(end, max_len=max_len)
    if not end_k:
        return 0
    hits = 0
    for w in words:
        if w.lower() == end.lower():
            continue
        if rhyme_key(w, max_len=max_len) == end_k:
            hits += 1
    return hits


def phrase_repetition_factor(bar: str, prev_bars: List[str]) -> float:
    if not prev_bars:
        return 1.0
    lower = bar.lower()
    prev_lower = [b.lower() for b in prev_bars if b]
    factor = 1.0
    for phrase in PHRASE_BAN_LIST:
        if phrase in lower:
            count = sum(phrase in pb for pb in prev_lower)
            if count >= MAX_PHRASE_REUSE:
                factor *= PHRASE_REUSE_FACTOR
    return factor


def choose_rhyme_anchor_word(bar: str) -> str:
    words = get_words(bar)
    if not words:
        return ""
    candidates = []
    for idx in range(len(words) - 1, -1, -1):
        w = words[idx]
        if w in COMMON_STOPWORDS or w in STOP_END_WORDS:
            continue
        g = get_rhyme_group(w)
        k = rhyme_key(w)
        score = 0.0
        if g is not None:
            score += 2.0
        score += min(len(k), 5) * 0.3
        score += (idx / max(len(words) - 1, 1)) * 0.5
        candidates.append((score, w))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return extract_end_word(bar)


def phrase_hard_block(bar: str, prev_bars: List[str]) -> bool:
    if not prev_bars:
        return False
    lower = bar.lower()
    prev_lower = [b.lower() for b in prev_bars if b]
    for phrase in PHRASE_BAN_LIST:
        if phrase in lower:
            count = sum(phrase in pb for pb in prev_lower)
            if count >= MAX_PHRASE_REUSE:
                return True
    return False


# --- syllable helpers --------------------------------------------------------

def count_syllables_word(word: str) -> int:
    if not word:
        return 0
    phones = pronouncing.phones_for_word(word.lower())
    if phones:
        return pronouncing.syllable_count(phones[0])
    w = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


def syllable_match_score(curr: int, prev: int) -> float:
    if prev <= 0 or curr <= 0:
        return 0.0
    diff = abs(curr - prev)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7
    if diff == 2:
        return 0.4
    return 0.0


def compute_total_score(
    bar: str,
    *,
    rhyme_score: float,
    topic_score: float,
    meter_score_val: float,
    ngram_score: float,
    w_rhyme: float = 0.4,
    w_topic: float = 0.2,
    w_meter: float = 0.2,
    w_ngram: float = 0.2,
) -> float:
    """
    Small additive "Hernandez-style" score combining:
    - rhyme alignment (end_rhyme_consistency_score),
    - topic coherence (Word2Vec),
    - meter (syllable-based),
    - n-gram critic.
    """
    return (
        w_rhyme * rhyme_score
        + w_topic * topic_score
        + w_meter * meter_score_val
        + w_ngram * ngram_score
    )


from transformers import LogitsProcessor, LogitsProcessorList


class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = scores.clamp(min=-50.0, max=50.0)
        return scores


# -----------------------------------------------------------------------------
# Siamese rhyme / coherence scorer
# -----------------------------------------------------------------------------

class SiameseRhymeScorer:
    def __init__(self, model_dir: str, device: str = "cpu", max_len: int = 64):
        self.model_dir = model_dir
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"
        self.max_len = max_len
        self.enabled = False
        self.tokenizer = None
        self.encoder = None
        self.hidden_size = 768

        if not os.path.isdir(model_dir):
            print(f"[WARN] Siamese model dir '{model_dir}' not found, disabling scorer.")
            return

        try:
            print(f"[INFO] Loading Siamese encoder from {model_dir} on {self.device} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.encoder = AutoModel.from_pretrained(model_dir)
            self.encoder.to(self.device)
            self.encoder.eval()
            self.hidden_size = self.encoder.config.hidden_size
            self.enabled = True
            print("[INFO] Siamese scorer enabled.")
        except Exception as e:
            print(f"[WARN] Failed to load Siamese encoder: {e}")
            self.enabled = False
            self.tokenizer = None
            self.encoder = None

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        if not self.enabled or self.encoder is None or self.tokenizer is None:
            return torch.zeros(self.hidden_size, device=self.device)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.encoder(**enc)
        last_hidden = outputs.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom
        emb = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return emb.squeeze(0)

    @torch.no_grad()
    def score_pair(self, a: str, b: str) -> float:
        if not self.enabled:
            return 0.0
        ea = self.embed(a)
        eb = self.embed(b)
        if ea.numel() == 0 or eb.numel() == 0:
            return 0.0
        return float(torch.dot(ea, eb).item())


def get_siamese_scorer() -> SiameseRhymeScorer:
    if SIAMESE_SCORER is None:
        raise RuntimeError("Siamese rhyme scorer not initialized. Call load_settings() first.")
    return SIAMESE_SCORER


# -----------------------------------------------------------------------------
# Bar filters: meta / filler
# -----------------------------------------------------------------------------

def is_meta_bar(bar: str) -> bool:
    if not bar:
        return True
    s = bar.strip()
    lower = s.lower()
    if "<|response" in lower or "<|bar" in lower or "<sep>" in lower:
        return True
    for rx in META_REGEXES:
        if rx.search(s):
            return True
    for tok in META_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", lower):
            return True
    for tag in META_ARTIST_TAGS:
        if tag in lower:
            return True
    if re.match(r"^[a-z0-9 .'-]+:\s", lower):
        return True
    if s.startswith('"') and s.endswith('"') and len(get_words(s)) <= 6:
        return True
    if s.endswith("]") or s.endswith(")") or s.endswith("..."):
        return True
    if len(get_words(s)) <= 2:
        return True
    return False


def has_repeated_phrase(bar: str, n: int = 2, min_repeats: int = 3) -> bool:
    tokens = get_words(bar)
    if len(tokens) < n * min_repeats:
        return False
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))
    counts = defaultdict(int)
    for ng in ngrams:
        counts[ng] += 1
    for _, c in counts.items():
        if c >= min_repeats:
            return True
    return False


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    total = len(text)
    ascii_chars = sum(1 for ch in text if 32 <= ord(ch) <= 126)
    return 1.0 - (ascii_chars / total)


def is_filler_bar(bar: str) -> bool:
    if not bar:
        return True
    if non_ascii_ratio(bar) > 0.40:
        return True
    lower = bar.strip().lower()
    if "i'm not a player, i just fuck a lot" in lower:
        return True
    if any(p in lower for p in STOP_PHRASES):
        return True
    if has_repeated_phrase(bar, n=2, min_repeats=3) or has_repeated_phrase(bar, n=3, min_repeats=2):
        return True
    end = extract_end_word(bar)
    if end in STOP_END_WORDS:
        return True
    if len(get_words(bar)) < 4:
        return True
    return False


# -----------------------------------------------------------------------------
# Candidate scoring (STRUCTURAL)
# -----------------------------------------------------------------------------

def end_rhyme_consistency_score(
    letter: str,
    bar: str,
    rhyme_memory: Dict[str, List[Dict]],
) -> float:
    anchor = choose_rhyme_anchor_word(bar)
    if not anchor:
        return 0.0
    group = get_rhyme_group(anchor)
    key = rhyme_key(anchor)
    history = rhyme_memory.get(letter, [])
    if not history:
        return 1.0
    base = history[0]
    base_group, base_key = base["group"], base["key"]
    if base_group is not None and group is not None:
        return 1.0 if base_group == group else 0.0
    if base_key and key:
        sim = rhyme_similarity(base_key, key)
        target = 0.9 if letter == "A" else 0.85
        return 1.0 if sim >= target else 0.0
    return 0.5


def coherence_score(
    bar: str,
    prev_bars: List[str],
    seed: str,
    siamese_scorer: SiameseRhymeScorer | None = None,
) -> float:
    scorer = siamese_scorer if siamese_scorer is not None else get_siamese_scorer()
    if not getattr(scorer, "enabled", False):
        return 0.0
    context = seed
    if prev_bars:
        context = seed + " || " + " ".join(prev_bars[-4:])
    return scorer.score_pair(context, bar)


def candidate_score(
    letter: str,
    bar: str,
    rhyme_memory: Dict[str, List[Dict]],
    prev_bars: List[str],
    seed: str,
    siamese_scorer: SiameseRhymeScorer | None = None,
) -> float:
    if not bar.strip():
        return -1e9
    if is_meta_bar(bar):
        return -1e9
    if phrase_hard_block(bar, prev_bars):
        return -1e9

    if is_filler_bar(bar):
        base_penalty = FILLER_PENALTY
    else:
        base_penalty = 1.0

        history = rhyme_memory.get(letter, [])

        # Optional strict rhyme hard-gate
        if ENFORCE_STRICT_IN_SCORING and history:
            base = history[0]
            base_group, base_key = base["group"], base["key"]

            anchor = choose_rhyme_anchor_word(bar)
            if anchor:
                group = get_rhyme_group(anchor)
                key = rhyme_key(anchor)

                # Case 1: both have rhyme groups -> must match
                if base_group is not None and group is not None and group != base_group:
                    return -1e9

                # Case 2: missing group info -> require strong suffix overlap
                if base_group is None or group is None:
                    if base_key and key:
                        sim = rhyme_similarity(base_key, key)
                        overlap_len = min(len(base_key), len(key))
                        min_chars = 4 if letter == "A" else 3
                        min_sim   = 0.85 if letter == "A" else 0.80
                        if not (overlap_len >= min_chars and sim >= min_sim):
                            return -1e9

    scorer = siamese_scorer if siamese_scorer is not None else get_siamese_scorer()
    theme_sim = 0.0
    if getattr(scorer, "enabled", False) and seed:
        theme_sim = scorer.score_pair(seed, bar)

    end_score = end_rhyme_consistency_score(letter, bar, rhyme_memory)
    internal_hits = count_internal_hits(bar)
    internal_score = math.log(1 + internal_hits)
    chain_hits = end_chain_hits(bar)
    # For primary rhyme letters (A/B), softly penalise bars with zero internal hits
    if letter in ("A", "B") and internal_hits == 0:
        internal_score -= 0.5  # tuneable; not a hard reject, just a nudge
    chain_score = math.log(1 + chain_hits)
    end_word = extract_end_word(bar)
    group = get_rhyme_group(end_word) if end_word else None
    group_present_score = 1.0 if group is not None else 0.0
    coh = coherence_score(bar, prev_bars, seed, siamese_scorer=scorer)

    total = (
        END_RHYME_WEIGHT * end_score +
        INTERNAL_RHYME_WEIGHT * internal_score +
        END_CHAIN_WEIGHT * chain_score +
        RHYME_GROUP_PRESENCE_WEIGHT * group_present_score +
        COHERENCE_WEIGHT * coh +
        THEME_SIM_WEIGHT * theme_sim
    )

    length_factor = bar_length_factor(bar)
    repeat_factor = 1.0
    if end_word:
        ew = end_word.lower()
        hist = rhyme_memory.get(letter, [])
        used_count = sum(1 for e in hist if e["word"].lower() == ew)
        if used_count >= REPEAT_ENDWORD_HARD_CAP:
            return -1e9
        elif used_count > 0:
            repeat_factor *= (REPEAT_ENDWORD_PENALTY ** used_count)
    phrase_factor = phrase_repetition_factor(bar, prev_bars)

    return total * base_penalty * length_factor * repeat_factor * phrase_factor


# -----------------------------------------------------------------------------
# Rhyme memory
# -----------------------------------------------------------------------------

def update_rhyme_memory(
    letter: str,
    bar: str,
    rhyme_memory: Dict[str, List[Dict]],
    max_history: int = 6,
):
    anchor = choose_rhyme_anchor_word(bar)
    if not anchor:
        return
    group = get_rhyme_group(anchor)
    entry = {
        "word": anchor,
        "group": group,
        "key": rhyme_key(anchor),
    }
    hist = rhyme_memory[letter]
    if not hist:
        hist.append(entry)
    else:
        if hist[0].get("group") is None and group is not None:
            hist.insert(0, entry)
        else:
            hist.append(entry)
    if len(hist) > max_history:
        rhyme_memory[letter] = hist[-max_history:]


def filter_candidates_by_rhyme_letter(
    letter: str,
    candidates: List[str],
    rhyme_memory: Dict[str, List[Dict]],
) -> List[str]:
    candidates = [
        c for c in candidates
        if c and not is_meta_bar(c) and not is_filler_bar(c)
    ]
    if not candidates:
        return []

    history = rhyme_memory.get(letter, [])
    if not history or not STRICT_RHYME_MODE:
        return candidates

    base = history[0]
    base_group, base_key = base["group"], base["key"]

    kept: List[str] = []
    for bar in candidates:
        anchor = choose_rhyme_anchor_word(bar)
        if not anchor:
            continue
        group = get_rhyme_group(anchor)
        key = rhyme_key(anchor)
        ok = False
        if base_group is not None and group is not None and group == base_group:
            ok = True
        if not ok and base_key and key:
            sim = rhyme_similarity(base_key, key)
            overlap_len = min(len(base_key), len(key))
            if overlap_len >= 3 and sim >= 0.75:
                ok = True
        if ok:
            kept.append(bar)

    if kept:
        return kept

    group_pref = []
    for bar in candidates:
        last = extract_end_word(bar)
        if last and get_rhyme_group(last) is not None:
            group_pref.append(bar)
    if group_pref:
        return group_pref
    return candidates


# -----------------------------------------------------------------------------
# Model loading / generation
# -----------------------------------------------------------------------------

def load_rap_model(adapter_dir: str, tokenizer_dir: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig

    print(f"__ Loading LoRA adapter from: {adapter_dir}")
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    base_model_name = peft_cfg.base_model_name_or_path
    print(f"__ Adapter expects base model: {base_model_name}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU is expected for Qwen2.5-7B.")

    print(f"__ Loading tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("__ Loading base model on GPU 0 (torch_dtype=torch.float16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print("__ Base model loaded on GPU.")

    base_vocab = base_model.get_input_embeddings().weight.shape[0]
    tok_vocab = len(tokenizer)
    print(f"__ Current base vocab size: {base_vocab}")
    print(f"__ Tokenizer vocab size  : {tok_vocab}")

    if base_vocab != tok_vocab:
        print(
            f"__ Resizing base model embeddings from {base_vocab} to {tok_vocab} "
            f"to match tokenizer..."
        )
        base_model.resize_token_embeddings(tok_vocab)
        print("__ Embeddings resized.")

    print(f"__ Attaching LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )
    model.eval()
    print("__ Model + adapter ready.")

    return tokenizer, model


def build_scheme_letters(scheme: str, num_bars: int) -> List[str]:
    scheme = scheme.strip().upper()
    if not scheme:
        scheme = "A"
    letters = list(scheme)
    out = []
    i = 0
    while len(out) < num_bars:
        out.append(letters[i % len(letters)])
        i += 1
    return out


def build_prompt(
    artist_token: str,
    section_token: str,
    seed: str,
    bar_index: int,
    scheme: str,
    end_words_hint: List[str] | None = None,
) -> str:
    hint_block = ""
    if end_words_hint:
        uniq = sorted({w.lower() for w in end_words_hint})
        sample_hints = ", ".join(uniq[:6])
        hint_block = (
            f"Target end rhymes: {sample_hints}. "
            f"Try to end the bar with a word that rhymes with one of these.\n"
        )

    return (
        f"<|artist:{artist_token}|><|section:{section_token}|>\n"
        f"Theme: {seed}\n"
        f"{hint_block}"
        f"Instruction: Write one dense, original rap bar that advances this theme. "
        f"Use a multi-syllable end rhyme (3–5 sounds) and echo that sound inside the bar "
        f"with internal rhymes. Keep it one tight line, around 8–14 words, no run-on "
        f"sentences. Avoid chant-like repetition, listing track numbers or times, album "
        f"credits, or direct quotes from existing songs. Do NOT literally repeat the phrases "
        f"'demons in the mirror' or 'pressure on my chest'; flip the imagery instead. "
        f"Focus on vivid imagery, internal multis, and emotional depth.\n"
        f"Scheme: {scheme}\n"
        f"Bar {bar_index}:"
    )


def generate_bar_candidates(
    artist_token: str,
    section_token: str,
    seed: str,
    tokenizer,
    model,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    bar_index: int = 1,
    scheme: str = "AAAA",
    end_words_hint: List[str] | None = None,
) -> List[str]:
    prompt = build_prompt(
        artist_token=artist_token,
        section_token=section_token,
        seed=seed,
        bar_index=bar_index,
        scheme=scheme,
        end_words_hint=end_words_hint,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    input_len = inputs["input_ids"].shape[1]
    logits_processors = LogitsProcessorList([SafeLogitsProcessor()])

    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_candidates,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processors,
    )

    bars = []
    for seq in outputs:
        gen = seq[input_len:]
        decoded = tokenizer.decode(gen, skip_special_tokens=True)
        bar_text = extract_bar_text(decoded)
        bars.append(bar_text)

    return bars


def select_best_bar(
    letter: str,
    candidates: List[str],
    rhyme_memory,
    prev_bars: List[str],
    seed: str,
    siamese_scorer: SiameseRhymeScorer | None,
    tokenizer,
    model,
    verse_length_model: LengthModel,
    score_weights: ScoreWeights,
    theme_ctx: ThemeContext,
    use_hybrid: bool,
    topic_scorer: TopicScorer | None,
    ngram_critic: NgramCritic | None,
    meter_target_syllables: int,
    meter_sigma: float,
    anchor_words: List[str] | None,
) -> Tuple[str, Dict[str, Any]]:
    filtered = filter_candidates_by_rhyme_letter(
        letter=letter,
        candidates=candidates,
        rhyme_memory=rhyme_memory,
    )
    if not filtered:
        filtered = [c for c in candidates if c and not is_meta_bar(c)]
    if not filtered:
        return "", {}

    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    model_device = next(model.parameters()).device

    # Normalise anchors for comparison
    anchor_set = {a.lower() for a in anchor_words} if anchor_words else set()

    for bar in filtered:
        struct_score = candidate_score(
            letter=letter,
            bar=bar,
            rhyme_memory=rhyme_memory,
            prev_bars=prev_bars,
            seed=seed,
            siamese_scorer=siamese_scorer,
        )
        if struct_score <= -1e8:
            continue

        # Optional hybrid LM + length + theme scoring (your existing pipeline)
        hybrid_component = 0.0
        theme_similarity = 0.0
        if use_hybrid:
            lm_logprob = compute_lm_logprob(
                model=model,
                tokenizer=tokenizer,
                text=bar,
                device=model_device,
            )

            internal_hits = count_internal_hits(bar)
            internal_rhyme_score = math.log(1 + internal_hits)
            line_emb = get_siamese_scorer().embed(bar)
            theme_similarity = compute_theme_similarity(line_emb, theme_ctx)

            length = len(bar.split())
            last_word = extract_end_word(bar)

            curr_syllables = count_syllables_word(last_word) if last_word else 0
            if prev_bars:
                prev_last = extract_end_word(prev_bars[-1])
                prev_syllables = count_syllables_word(prev_last) if prev_last else 0
            else:
                prev_syllables = curr_syllables
            syl_match = syllable_match_score(curr_syllables, prev_syllables)

            feats = build_line_features(
                text=bar,
                last_word=last_word,
                length=length,
                length_model=verse_length_model,
                lm_logprob=lm_logprob,
                rhyme_score_end=0.0,  # rhyme handled by structural + Hernandez part
                internal_rhyme_score=internal_rhyme_score,
                theme_similarity=theme_similarity,
                syllable_match_score=syl_match,
            )
            hybrid_component = hybrid_score(feats, score_weights)

        # Hernandez-style metrics
        # 1) End-rhyme alignment with memory
        rhyme_alignment = end_rhyme_consistency_score(letter, bar, rhyme_memory)

        # 2) Topic similarity via Word2Vec
        topic_score = 0.0
        if topic_scorer is not None:
            topic_score = topic_scorer.similarity(bar, seed)

        # 3) Meter score (syllable control for whole bar)
        meter_score_val = meter_score(
            bar,
            target_syllables=meter_target_syllables,
            sigma=meter_sigma,
        )

        # 4) N-gram critic score
        ngram_score_val = 0.0
        if ngram_critic is not None:
            ngram_score_val = ngram_critic.critic_score(bar)

        hernandez_component = compute_total_score(
            bar,
            rhyme_score=rhyme_alignment,
            topic_score=topic_score,
            meter_score_val=meter_score_val,
            ngram_score=ngram_score_val,
        )

        # 5) Anchor bonus: if we actually hit one of the planned end words, reward it
        anchor_bonus = 0.0
        end_word = extract_end_word(bar)
        if end_word and anchor_set and end_word.lower() in anchor_set:
            anchor_bonus = 0.2  # small but meaningful extra

        final_score = struct_score + hybrid_component + hernandez_component + anchor_bonus
        diagnostics = {
            "bar": bar,
            "letter": letter,
            "struct_score": struct_score,
            "hybrid_component": hybrid_component,
            "hernandez_component": hernandez_component,
            "anchor_bonus": anchor_bonus,
            "rhyme_alignment": rhyme_alignment,
            "topic_score": topic_score,
            "meter_score": meter_score_val,
            "ngram_score": ngram_score_val,
            "final_score": final_score,
            "anchor_hit": bool(end_word and anchor_set and end_word.lower() in anchor_set),
            "theme_similarity": theme_similarity,
            "candidate_pool": len(candidates),
            "filtered_pool": len(filtered),
            "end_word": end_word,
            "length": len(bar.split()),
        }
        scored.append((bar, final_score, diagnostics))

    if not scored:
        return "", {}

    scored.sort(key=lambda x: x[1], reverse=True)
    best_bar, _, diag = scored[0]
    diag["selected"] = True
    return best_bar, diag


def verse_siamese_rhyme_score(
    verse_bars: List[Tuple[str, str]],
    scorer: SiameseRhymeScorer,
) -> float:
    """
    Verse coherence score based on *adjacent* bar pairs only.

    This rewards smooth transitions bar-to-bar instead of forcing
    the entire 12-bar verse to live in one semantic cluster.
    """
    if not scorer.enabled or len(verse_bars) < 2:
        return 0.0

    texts = [b for _, b in verse_bars if b.strip()]
    if len(texts) < 2:
        return 0.0

    embs = [scorer.embed(t) for t in texts]
    embs = [e / (e.norm() + 1e-8) for e in embs]

    sims = []
    for i in range(len(embs) - 1):
        sims.append(float((embs[i] * embs[i + 1]).sum().item()))

    return float(np.mean(sims)) if sims else 0.0


def analyse_verse(verse_bars: List[Tuple[str, str]]):
    print("\n=== RHYME ANALYSIS ===")
    for i, (letter, bar) in enumerate(verse_bars, start=1):
        end = extract_end_word(bar)
        group = get_rhyme_group(end)
        internal_hits = count_internal_hits(bar)
        print(f"[{i:02d}] ({letter}) end='{end}' group={group} internal_hits={internal_hits}")

    print("\n-- Cross-bar rhyme links (shared groups / near-rhymes) --")
    for i, (_, bar_i) in enumerate(verse_bars):
        end_i = extract_end_word(bar_i)
        key_i = rhyme_key(end_i)
        group_i = get_rhyme_group(end_i)
        links = []
        for j, (_, bar_j) in enumerate(verse_bars):
            if j <= i:
                continue
            end_j = extract_end_word(bar_j)
            key_j = rhyme_key(end_j)
            group_j = get_rhyme_group(end_j)
            label = None
            if group_i is not None and group_j is not None and group_i == group_j:
                label = "STRONG_END/INTERNAL_GROUP"
            else:
                sim = rhyme_similarity(key_i, key_j)
                if sim >= 0.85:
                    label = "NEAR_END"
            if label:
                links.append(f"{j+1:02d}({label})")
        print(f"Bar {i+1:02d} links -> " + (", ".join(links) if links else "(no strong rhyme links)"))


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def verse_to_text(verse_bars: List[Tuple[str, str]]) -> str:
    lines = []
    for _, bar in verse_bars:
        if bar:
            lines.append(bar.rstrip())
    lines.append("<END_SONG>")
    return "\n".join(lines)


def bars_payload(verse_bars: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    return [{"letter": letter, "text": bar} for letter, bar in verse_bars]


def resolve_log_path(args, settings) -> Path | None:
    if args.log_json:
        return Path(args.log_json)
    if args.log_dir:
        return Path(args.log_dir) / DEFAULT_LOG_FILENAME
    return settings.generation_log_path


def append_generation_log(log_path: Path | None, entry: Dict[str, Any]):
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate a rhymed rap verse.")
    p.add_argument("--artist", type=str, required=True)
    p.add_argument("--section", type=str, default="VERSE")
    p.add_argument("--seed", type=str, required=True)
    p.add_argument("--scheme", type=str, default="AAAA")
    p.add_argument("--num_bars", type=int, default=12)
    p.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES_PER_BAR)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    p.add_argument("--attempts", type=int, default=4)
    p.add_argument("--verse_accept_threshold", type=float, default=0.35)
    p.add_argument(
        "--log_json",
        type=str,
        default=None,
        help="Optional path to append generation logs (JSONL). Defaults to config path.",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Optional directory to place generated_raw.jsonl (overrides config).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a JSON/YAML config overriding default paths.",
    )

    # Hybrid scoring toggle
    p.add_argument(
        "--no_hybrid",
        action="store_false",
        dest="hybrid",
        help="Disable hybrid LM/length/theme scoring (use structural + Hernandez-style only).",
    )
    p.set_defaults(hybrid=True)

    # Hernandez-style extras
    p.add_argument(
        "--topic_model_path",
        type=str,
        default=None,
        help="Path to Word2Vec topic model for TopicScorer (default from config).",
    )
    p.add_argument(
        "--ngram_path",
        type=str,
        default=None,
        help="Path to n-gram critic data (default from config).",
    )
    p.add_argument(
        "--target_syllables",
        type=int,
        default=13,
        help="Target syllables per bar for meter_score.",
    )
    p.add_argument(
        "--meter_sigma",
        type=float,
        default=2.0,
        help="Sigma for meter_score Gaussian (larger = looser).",
    )
    p.add_argument(
        "--anchor_candidates_per_line",
        type=int,
        default=6,
        help="Number of planned anchor end-words per line (for rhyme planner).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    adapter_dir = str(settings.adapter_dir)
    tokenizer_dir = str(settings.tokenizer_dir)
    rhyme_csv = str(settings.rhyme_groups_csv)
    siamese_dir = str(settings.siamese_model_dir)
    topic_model_path = args.topic_model_path or str(settings.topic_model_path)
    ngram_path = args.ngram_path or str(settings.ngram_output_path)
    log_path = resolve_log_path(args, settings)

    global RHYME_GROUPS, SIAMESE_SCORER
    RHYME_GROUPS = load_word_rhyme_groups(rhyme_csv)
    print(f"__ Using word->group rhyme entries from {rhyme_csv} (count={len(RHYME_GROUPS)})")

    print(f"__ Loading Siamese rhyme model from {siamese_dir} ...")
    SIAMESE_SCORER = SiameseRhymeScorer(siamese_dir, device=DEVICE)
    siamese_scorer = get_siamese_scorer()
    if siamese_scorer.enabled:
        print("__ Siamese rhyme scorer ready.\n")
    else:
        print("__ Siamese rhyme scorer disabled (structural filters only).\n")

    if log_path:
        print(f"__ Generation logs will be appended to: {log_path}")

    artist_token = args.artist.strip().lower().replace(" ", "_")
    section_token = args.section.strip().upper()
    seed = args.seed.strip()
    scheme = args.scheme.strip().upper()
    scheme_letters = build_scheme_letters(scheme, args.num_bars)

    print(f"__ Loading model for artist '{artist_token}' with scheme '{scheme}' ...")
    tokenizer, model = load_rap_model(adapter_dir=adapter_dir, tokenizer_dir=tokenizer_dir)

    # Load group->words mapping for rhyme planner
    try:
        group_to_words = load_rhyme_groups_planner(rhyme_csv)
        print(f"__ Rhyme planner loaded {len(group_to_words)} groups for anchor planning.")
    except Exception as e:
        print(f"[WARN] Failed to load planner rhyme groups: {e}")
        group_to_words = None

    # Topic scorer (Word2Vec)
    try:
        topic_scorer = TopicScorer(topic_model_path)
        print("__ Topic scorer (Word2Vec) ready.")
    except Exception as e:
        print(f"[WARN] Topic model not available: {e}")
        topic_scorer = None

    # N-gram critic
    try:
        ngram_critic = NgramCritic(ngram_path)
        print("__ n-gram critic ready.")
    except Exception as e:
        print(f"[WARN] n-gram critic not available: {e}")
        ngram_critic = None

    # Plan rhyme endings per line (Hernandez-style)
    if group_to_words is not None:
        anchor_plan = plan_rhyme_endings(
            seed=seed,
            scheme=scheme,
            group_to_words=group_to_words,
            num_candidates_per_group=args.anchor_candidates_per_line,
        )
    else:
        anchor_plan = {}

    # Hybrid length prior (your existing prior; can be replaced later)
    VERSE_LENGTH_COUNTS = {
        3: 3987,
        4: 7058,
        5: 7835,
        6: 8797,
        7: 8296,
        8: 7998,
        9: 6748,
        10: 5775,
        11: 3938,
        12: 2629,
    }

    verse_length_model = build_length_model(VERSE_LENGTH_COUNTS)

    score_weights = ScoreWeights(
        lm=1.0,
        rhyme=0.0,           # rhyme handled by structural + Hernandez components
        internal_rhyme=1.0,
        theme=0.9,
        length=0.4,
        syllable_match=0.6,
    )

    theme_emb = siamese_scorer.embed(seed)
    theme_ctx = ThemeContext(embedding=theme_emb)

    print(f"\n=== Target: {args.num_bars} bars ({''.join(scheme_letters)}) ===")
    print(f"Artist   : {artist_token}")
    print(f"Section  : {section_token}")
    print(f"Seed     : {seed}")
    print(f"Scheme   : {''.join(scheme_letters)}")
    print(f"Hybrid   : {'ON' if args.hybrid else 'OFF'}\n")

    best_verse = None
    best_score = -1e9

    for attempt in range(1, args.attempts + 1):
        print(f"\n===== VERSE ATTEMPT {attempt}/{args.attempts} =====")
        rhyme_memory = defaultdict(list)
        verse_bars: List[Tuple[str, str]] = []
        prev_bars_text: List[str] = []
        bar_metrics: List[Dict[str, Any]] = []
        verse_id = uuid.uuid4().hex

        for idx, letter in enumerate(scheme_letters, start=1):
            print(f"[Bar {idx}/{len(scheme_letters)}] Rhyme letter: {letter}")

            line_idx = idx - 1
            line_anchors = anchor_plan.get(line_idx) if anchor_plan else None

            if line_anchors:
                print(f"  Planned anchor end-words (sample): {', '.join(line_anchors[:4])}")

            candidates = generate_bar_candidates(
                artist_token=artist_token,
                section_token=section_token,
                seed=seed,
                tokenizer=tokenizer,
                model=model,
                num_candidates=args.candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                bar_index=idx,
                scheme=scheme,
                end_words_hint=line_anchors,
            )

            bar, bar_diag = select_best_bar(
                letter=letter,
                candidates=candidates,
                rhyme_memory=rhyme_memory,
                prev_bars=prev_bars_text,
                seed=seed,
                siamese_scorer=siamese_scorer,
                tokenizer=tokenizer,
                model=model,
                verse_length_model=verse_length_model,
                score_weights=score_weights,
                theme_ctx=theme_ctx,
                use_hybrid=args.hybrid,
                topic_scorer=topic_scorer,
                ngram_critic=ngram_critic,
                meter_target_syllables=args.target_syllables,
                meter_sigma=args.meter_sigma,
                anchor_words=line_anchors,
            )

            if not bar_diag:
                bar_diag = {"bar": bar, "letter": letter, "final_score": None}
            bar_metrics.append(bar_diag)

            update_rhyme_memory(letter, bar, rhyme_memory)
            verse_bars.append((letter, bar))
            prev_bars_text.append(bar)
            print(f"  -> Selected: {bar}\n")

        verse_score = verse_siamese_rhyme_score(verse_bars, siamese_scorer)
        print(f"--> Verse-level Siamese rhyme score: {verse_score:.3f}")

        log_entry = {
            "log_version": 1,
            "verse_id": verse_id,
            "timestamp": utc_timestamp(),
            "artist": args.artist,
            "artist_token": artist_token,
            "seed": seed,
            "scheme": "".join(scheme_letters),
            "num_bars": len(scheme_letters),
            "attempt_index": attempt,
            "attempts_total": args.attempts,
            "hybrid_mode": args.hybrid,
            "generation_params": {
                "candidates": args.candidates,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "target_syllables": args.target_syllables,
                "meter_sigma": args.meter_sigma,
                "scheme": scheme,
            },
            "bars": bars_payload(verse_bars),
            "bar_metrics": bar_metrics,
            "verse_text": verse_to_text(verse_bars),
            "verse_score": verse_score,
            "verse_accept_threshold": args.verse_accept_threshold,
            "accepted": verse_score >= args.verse_accept_threshold,
            "topic_model_path": topic_model_path,
            "ngram_path": ngram_path,
            "rhyme_groups_csv": rhyme_csv,
            "siamese_model_dir": siamese_dir,
            "settings_snapshot": settings.as_dict(),
        }
        if bar_metrics:
            meter_vals = [bm.get("meter_score") for bm in bar_metrics if isinstance(bm.get("meter_score"), (int, float))]
            final_vals = [bm.get("final_score") for bm in bar_metrics if isinstance(bm.get("final_score"), (int, float))]
            summary = {}
            if meter_vals:
                summary["avg_meter"] = float(np.mean(meter_vals))
            if final_vals:
                summary["avg_final_score"] = float(np.mean(final_vals))
            if summary:
                log_entry["metrics_summary"] = summary
        append_generation_log(log_path, log_entry)

        if verse_score > best_score:
            best_score = verse_score
            best_verse = verse_bars

        if verse_score >= args.verse_accept_threshold:
            print(f"✅ Verse accepted (score {verse_score:.3f} ≥ threshold {args.verse_accept_threshold:.3f})")
            break
        else:
            print(f"❌ Verse rejected (score {verse_score:.3f} < threshold {args.verse_accept_threshold:.3f})")

    if best_verse is None:
        print("[WARN] No verse generated.")
        return

    print("\n=== FINAL VERSE ===")
    for i, (letter, bar) in enumerate(best_verse, start=1):
        print(f"{i:02d} [{letter}] {bar}")

    analyse_verse(best_verse)


if __name__ == "__main__":
    main()
