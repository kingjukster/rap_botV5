"""
Siamese rhyme / coherence scorer for evo_rhyme.

Lightweight wrapper around a sentence-level encoder for scoring
coherence / similarity between bars and computing embeddings.
"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]


def get_siamese_model_dir() -> Path:
    """
    Resolve Siamese model directory: config.settings, then env vars, else default.

    Priority:
      1. config.settings load_settings().siamese_model_dir
      2. RAPBOT_SIAMESE_DIR or EVO_RHYME_SIAMESE_DIR
      3. ROOT / rhyme_siamese
    """
    try:
        from config.settings import load_settings

        cfg = load_settings(os.environ.get("RAPBOT_CONFIG"))
        return Path(cfg.siamese_model_dir)
    except Exception:
        pass

    for key in ("RAPBOT_SIAMESE_DIR", "EVO_RHYME_SIAMESE_DIR"):
        val = os.environ.get(key)
        if val:
            return Path(val)

    return ROOT / "rhyme_siamese"


SIAMESE_MODEL_DIR = get_siamese_model_dir()


class SiameseRhymeScorer:
    """
    Lightweight wrapper around a sentence-level encoder saved at model_dir.

    Used for:
      - Scoring coherence / similarity between bars
      - Computing embeddings for Pass B scoring
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        batch_size: int = 64,
        max_len: int = 16,
    ):
        model_dir = Path(model_dir)

        # Decide device once and use it everywhere
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"[INFO] Loading Siamese encoder from {model_dir} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.encoder = AutoModel.from_pretrained(str(model_dir))
        self.encoder.to(self.device)
        self.encoder.eval()

        self.batch_size = batch_size
        self.max_len = max_len
        self.enabled = True
        self.model_dir = model_dir

        # Store hidden_size for use when encoder may be None (e.g. disabled path)
        self._hidden_size = (
            self.encoder.config.hidden_size
            if self.encoder and hasattr(self.encoder, "config")
            else 768
        )

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        """
        Encode a single string into a normalized embedding [H].
        """
        if not self.enabled or self.encoder is None or self.tokenizer is None:
            return torch.zeros(self._hidden_size, device=self.device)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.encoder(**enc)
        last_hidden = outputs.last_hidden_state  # [1, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [1, T, 1]

        summed = (last_hidden * mask).sum(dim=1)  # [1, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)  # [1, 1]
        pooled = summed / denom  # [1, H]

        emb = torch.nn.functional.normalize(pooled, p=2, dim=-1)  # [1, H]
        return emb.squeeze(0)  # [H]

    @torch.no_grad()
    def embed_batch(self, texts, batch_size: int = None):
        if not self.enabled or not texts:
            return torch.empty(0, self._hidden_size)

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
        if self.encoder is not None and hasattr(self.encoder, "config"):
            return self.encoder.config.hidden_size
        return self._hidden_size


__all__ = ["SiameseRhymeScorer", "get_siamese_model_dir", "SIAMESE_MODEL_DIR"]
