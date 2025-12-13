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

import torch
import torch.nn as nn

from rapbot.rhyme_scorer import SiameseRhymeScorer

SCORE_KEYS = ["overall_score", "depth_score", "coherence_score", "originality_score"]

ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if not hidden_layers:
            hidden_layers = [input_dim]
        act_name = activation.lower()
        act_cls = ACTIVATIONS.get(act_name, nn.GELU)

        layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

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
        input_dim = checkpoint["input_dim"]
        self.score_keys: List[str] = checkpoint.get("score_keys", SCORE_KEYS)
        output_dim = len(self.score_keys)
        hidden_layers = checkpoint.get("hidden_layers")
        if isinstance(hidden_layers, list):
            hidden_layers = [int(dim) for dim in hidden_layers if isinstance(dim, (int, float))]
        if not hidden_layers:
            legacy_hidden = checkpoint.get("hidden_dim")
            hidden_layers = [int(legacy_hidden)] if legacy_hidden else [input_dim]
        dropout = float(checkpoint.get("dropout", 0.0))
        activation = checkpoint.get("activation", "gelu")

        self.head = CriticHead(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
        ).to(self.device)
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
    hidden_layers: List[int],
    score_keys: Optional[List[str]] = None,
    activation: str = "gelu",
    dropout: float = 0.0,
):
    payload = {
        "state_dict": head.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_layers[0] if hidden_layers else input_dim,
        "hidden_layers": hidden_layers,
        "score_keys": score_keys or SCORE_KEYS,
        "activation": activation,
        "dropout": dropout,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_head(path: Path, device: torch.device) -> CriticHead:
    payload = torch.load(path, map_location=device)
    input_dim = payload["input_dim"]
    hidden_layers = payload.get("hidden_layers")
    if isinstance(hidden_layers, list):
        hidden_layers = [int(dim) for dim in hidden_layers if isinstance(dim, (int, float))]
    if not hidden_layers:
        hidden_layers = [int(payload.get("hidden_dim", input_dim))]
    activation = payload.get("activation", "gelu")
    dropout = float(payload.get("dropout", 0.0))
    output_dim = len(payload.get("score_keys", SCORE_KEYS))
    head = CriticHead(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
    )
    head.load_state_dict(payload["state_dict"])
    head.to(device)
    return head
