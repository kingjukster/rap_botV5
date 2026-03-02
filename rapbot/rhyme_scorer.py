"""
rhyme_scorer.py

Minimal helper module used by build_elite_kaggle_corpus_multi_stage.py
for rhyme-group metadata and Siamese coherence scoring.

- Loads rhyme groups from rhymes_grouped.csv
- Provides a SiameseRhymeScorer with embed / embed_batch / score_pair
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def _load_settings_paths() -> Tuple[Path, Path]:
    """
    Try to pull canonical paths from config/settings so every script
    agrees on where rhyme assets live. Falls back to repo-relative paths
    if the config module is unavailable (e.g., lightweight tooling).
    """
    try:
        from config.settings import load_settings

        cfg = load_settings(os.environ.get("RAPBOT_CONFIG"))
        return Path(cfg.rhyme_groups_csv), Path(cfg.siamese_model_dir)
    except Exception:
        # Repo-relative defaults keep the legacy behaviour but avoid
        # hard-coded /workspace paths that break on other machines.
        return ROOT / "data" / "rhymes_grouped.csv", ROOT / "rhyme_siamese"


_DEFAULT_RHYME_CSV, _DEFAULT_SIAMESE_DIR = _load_settings_paths()

RHYME_CSV_PATH = Path(os.environ.get("RAPBOT_RHYME_CSV", str(_DEFAULT_RHYME_CSV)))
SIAMESE_MODEL_DIR = Path(os.environ.get("RAPBOT_SIAMESE_DIR", str(_DEFAULT_SIAMESE_DIR)))

DEVICE = "cpu"  # keep this on CPU to avoid fighting Qwen for VRAM


# ---------------------------------------------------------------------
# Rhyme groups
# ---------------------------------------------------------------------

def load_rhyme_groups(csv_path: str | Path, validate: bool = True) -> Dict[str, int]:
    """
    Load a mapping word -> group_id from rhymes_grouped.csv.
    
    Args:
        csv_path: Path to rhymes_grouped.csv
        validate: If True, perform basic validation checks
        
    Returns:
        Dictionary mapping word -> group_id
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[WARN] Rhyme CSV not found at {csv_path}, continuing with empty mapping.")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read rhyme CSV {csv_path}: {e}")
        return {}
    
    # Validation checks
    if validate:
        if "word" not in df.columns or "group" not in df.columns:
            print(f"[WARN] Rhyme CSV missing required columns (word, group). Found: {list(df.columns)}")
            return {}
        
        # Check for empty dataframe
        if len(df) == 0:
            print(f"[WARN] Rhyme CSV is empty")
            return {}
    
    mapping: Dict[str, int] = {}
    duplicate_count = 0
    invalid_count = 0
    
    for _, row in df.iterrows():
        w = str(row["word"]).strip().lower()
        if not w:
            continue
        
        try:
            g = int(row["group"])
            if g < 0:
                invalid_count += 1
                continue
            
            # Handle duplicates by keeping the last entry
            if w in mapping and mapping[w] != g:
                duplicate_count += 1
            mapping[w] = g
        except (ValueError, KeyError):
            invalid_count += 1
            continue
    
    if validate and (duplicate_count > 0 or invalid_count > 0):
        print(f"[WARN] Found {duplicate_count} duplicate words and {invalid_count} invalid entries in rhyme CSV")
    
    print(f"[INFO] Loaded {len(mapping)} rhyme-group entries from {csv_path}")
    return mapping


# Load rhyme groups with validation
# If loading fails, empty dict is returned and system falls back to phonetic analysis
RHYME_GROUPS: Dict[str, int] = load_rhyme_groups(RHYME_CSV_PATH, validate=True)


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

    def __init__(self, model_dir: str | Path, device: str = "cuda", batch_size: int = 64, max_len: int = 16):
        import torch
        from transformers import AutoTokenizer, AutoModel

        # Decide device once and use it everywhere
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        model_dir = Path(model_dir)
        print(f"[INFO] Loading Siamese encoder from {model_dir} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.encoder = AutoModel.from_pretrained(str(model_dir))
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
    
    @torch.no_grad()
    def score_pairs_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score multiple pairs in batch for efficiency.
        
        Args:
            pairs: List of (text1, text2) tuples
            
        Returns:
            List of similarity scores
        """
        if not self.enabled or not pairs:
            return [0.0] * len(pairs)
        
        texts1 = [p[0] for p in pairs]
        texts2 = [p[1] for p in pairs]
        
        embs1 = self.embed_batch(texts1)
        embs2 = self.embed_batch(texts2)
        
        if embs1.shape[0] != embs2.shape[0]:
            return [0.0] * len(pairs)
        
        # Compute cosine similarities
        scores = []
        for i in range(len(pairs)):
            e1 = embs1[i]
            e2 = embs2[i]
            if e1.numel() == 0 or e2.numel() == 0:
                scores.append(0.0)
            else:
                # Cosine similarity
                dot_product = torch.dot(e1, e2).item()
                norm1 = torch.norm(e1).item()
                norm2 = torch.norm(e2).item()
                if norm1 > 0 and norm2 > 0:
                    score = dot_product / (norm1 * norm2)
                    scores.append(float(score))
                else:
                    scores.append(0.0)
        
        return scores
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the encoder."""
        if hasattr(self.encoder, 'config'):
            return self.encoder.config.hidden_size
        # Fallback
        return 768