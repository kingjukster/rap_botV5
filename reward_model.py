#!/usr/bin/env python
"""
reward_model.py

Local reward model (critic) built on top of the Siamese encoder.
Predicts critic scores (overall/depth/coherence/originality) from verse text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
import torch.nn as nn

from rhyme_scorer import SiameseRhymeScorer

SCORE_KEYS = ["overall_score", "depth_score", "coherence_score", "originality_score"]


class CriticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LocalCriticConfig:
    head_path: Path
    siamese_model_dir: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LocalCritic:
    """
    Thin wrapper: embed verse text via Siamese encoder and apply critic head.
    """

    def __init__(self, config: LocalCriticConfig):
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.siamese = SiameseRhymeScorer(str(config.siamese_model_dir), device=str(self.device))

        if not config.head_path.exists():
            raise FileNotFoundError(f"Local critic head not found at {config.head_path}")

        checkpoint = torch.load(config.head_path, map_location=self.device)
        hidden_dim = checkpoint["hidden_dim"]
        input_dim = checkpoint["input_dim"]
        self.score_keys: List[str] = checkpoint.get("score_keys", SCORE_KEYS)
        output_dim = len(self.score_keys)

        self.head = CriticHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        self.head.load_state_dict(checkpoint["state_dict"])
        self.head.eval()

    @torch.no_grad()
    def score(self, text: str) -> Dict[str, float]:
        emb = self.siamese.embed(text)
        emb = emb.to(self.device)
        preds = self.head(emb.unsqueeze(0)).squeeze(0).cpu().tolist()
        return {key: float(value) for key, value in zip(self.score_keys, preds)}


def save_head(
    head: CriticHead,
    path: Path,
    input_dim: int,
    hidden_dim: int,
    score_keys: Optional[List[str]] = None,
):
    payload = {
        "state_dict": head.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "score_keys": score_keys or SCORE_KEYS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_head(path: Path, device: torch.device) -> CriticHead:
    payload = torch.load(path, map_location=device)
    head = CriticHead(
        input_dim=payload["input_dim"],
        hidden_dim=payload["hidden_dim"],
        output_dim=len(payload.get("score_keys", SCORE_KEYS)),
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device)
    return head
