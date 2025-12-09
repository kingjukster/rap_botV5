"""
rhyme_scorer.py

Minimal helper module used by build_elite_kaggle_corpus_multi_stage.py
for rhyme-group metadata and Siamese coherence scoring.

- Loads rhyme groups from rhymes_grouped.csv
- Provides a SiameseRhymeScorer with embed / embed_batch / score_pair
"""

import os
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

RHYME_CSV_PATH = "/workspace/rap-botV4/rhymes_grouped.csv"
SIAMESE_MODEL_DIR = "/workspace/rap-botV4/rhyme_siamese"
DEVICE = "cpu"  # keep this on CPU to avoid fighting Qwen for VRAM


# ---------------------------------------------------------------------
# Rhyme groups
# ---------------------------------------------------------------------

def load_rhyme_groups(csv_path: str) -> Dict[str, int]:
    """
    Load a mapping word -> group_id from rhymes_grouped.csv.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] Rhyme CSV not found at {csv_path}, continuing with empty mapping.")
        return {}
    df = pd.read_csv(csv_path)
    mapping: Dict[str, int] = {}
    for _, row in df.iterrows():
        w = str(row["word"]).strip().lower()
        g = int(row["group"])
        if w:
            mapping[w] = g
    print(f"[INFO] Loaded {len(mapping)} rhyme-group entries from {csv_path}")
    return mapping


RHYME_GROUPS: Dict[str, int] = load_rhyme_groups(RHYME_CSV_PATH)


def get_rhyme_group(word: str):
    return RHYME_GROUPS.get(str(word).lower(), None)


def rhyme_key(word: str, max_len: int = 4) -> str:
    """
    Simple fallback rhyme key = lowercased alpha tail, up to max_len chars.
    """
    import re

    w = re.sub(r"[^a-zA-Z]", "", str(word).lower())
    if not w:
        return ""
    return w[-max_len:]


def rhyme_similarity(k1: str, k2: str) -> float:
    """
    Suffix-based similarity in [0,1] between two rhyme keys.
    """
    if not k1 or not k2:
        return 0.0
    max_len = min(len(k1), len(k2))
    if max_len == 0:
        return 0.0
    matches = 0
    for i in range(1, max_len + 1):
        if k1[-i:] == k2[-i:]:
            matches = i
        else:
            break
    return matches / max_len


# ---------------------------------------------------------------------
# Siamese rhyme / coherence scorer
# ---------------------------------------------------------------------

class SiameseRhymeScorer:
    """
    Lightweight wrapper around a sentence-level encoder saved at SIAMESE_MODEL_DIR.

    Used for:
      - Scoring coherence / similarity between bars
      - Computing embeddings for Pass B scoring
    """

    def __init__(self, model_dir: str, device: str = "cuda", batch_size: int = 64, max_len: int = 16):
        import torch
        from transformers import AutoTokenizer, AutoModel

        # Decide device once and use it everywhere
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"[INFO] Loading Siamese encoder from {model_dir} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.encoder = AutoModel.from_pretrained(model_dir)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.batch_size = batch_size
        self.max_len = max_len
        self.enabled = True
        self.model_dir = model_dir
        
    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        """
        Encode a single string into a normalized embedding [H].
        """
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
        last_hidden = outputs.last_hidden_state      # [1, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)   # [1, T, 1]

        summed = (last_hidden * mask).sum(dim=1)     # [1, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)      # [1, 1]
        pooled = summed / denom                      # [1, H]

        emb = torch.nn.functional.normalize(pooled, p=2, dim=-1)  # [1, H]
        return emb.squeeze(0)  # [H]

    @torch.no_grad()
    def embed_batch(self, texts, batch_size: int = None):
        if not self.enabled or not texts:
            return torch.empty(0, 768)
    
        if batch_size is None:
            batch_size = self.batch_size
    
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
    
                enc = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                # 🔑 move inputs to same device as encoder
                enc = {k: v.to(self.device) for k, v in enc.items()}
    
                outputs = self.encoder(**enc)
                hidden = outputs.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
    
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                emb = summed / denom
    
                # store CPU tensors to avoid GPU blow-up
                all_embs.append(emb.cpu())
    
        return torch.cat(all_embs, dim=0)



    @torch.no_grad()
    def score_pair(self, a: str, b: str) -> float:
        """
        Cosine similarity between embeddings of two texts, in [-1,1].
        """
        if not self.enabled:
            return 0.0
        ea = self.embed(a)
        eb = self.embed(b)
        if ea.numel() == 0 or eb.numel() == 0:
            return 0.0
        return float(torch.dot(ea, eb).item())
