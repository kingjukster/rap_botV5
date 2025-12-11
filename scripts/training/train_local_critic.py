#!/usr/bin/env python
"""
Train a lightweight local critic (reward model) on scored_dataset.jsonl.

Uses frozen Siamese encoder embeddings + MLP regression head.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config.settings import load_settings
from rapbot.reward_model import CriticHead, save_head, SCORE_KEYS
from rapbot.rhyme_scorer import SiameseRhymeScorer


class CriticDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, targets: torch.Tensor):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the local critic reward head.")
    parser.add_argument(
        "--scored_dataset",
        type=str,
        default=None,
        help="JSONL file with combined generation logs + critic scores (default from config).",
    )
    parser.add_argument(
        "--siamese_model_dir",
        type=str,
        default=None,
        help="Directory containing Siamese encoder (default from config).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store trained reward head (default config.local_critic_dir).",
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit hidden layer sizes (e.g., --layers 768 512 256).",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=["gelu", "relu", "silu", "tanh"],
        help="Activation used between critic head layers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.9, help="Fraction of data for training.")
    parser.add_argument(
        "--kaggle_enriched",
        type=str,
        default=None,
        help="Optional enriched Kaggle JSONL (from enrich_kaggle_corpus.py) for pseudo-labeled samples.",
    )
    parser.add_argument(
        "--kaggle_limit",
        type=int,
        default=5000,
        help="Maximum Kaggle pseudo-samples to include.",
    )
    parser.add_argument(
        "--kaggle_score_floor",
        type=float,
        default=1.2,
        help="Minimum critic score assigned to Kaggle pseudo-samples.",
    )
    parser.add_argument(
        "--kaggle_score_ceiling",
        type=float,
        default=3.2,
        help="Maximum critic score assigned to Kaggle pseudo-samples.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON/YAML config file for defaults.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_scored_dataset(path: Path) -> List[Tuple[str, Dict[str, float]]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            critic = record.get("critic", {})
            if not all(k in critic for k in SCORE_KEYS):
                continue
            verse_text = record.get("verse_text")
            if not verse_text:
                bars = record.get("bars") or []
                verse_text = "\n".join(bar.get("text", "") for bar in bars if bar.get("text"))
            if not verse_text:
                continue
            target = {k: float(critic.get(k, 0.0)) for k in SCORE_KEYS}
            records.append((verse_text, target))
    return records


def heuristic_scores(entry: Dict[str, Any], floor: float, ceiling: float) -> Dict[str, float]:
    density = min(entry.get("unique_rhymes_window", 0) / 4.0, 1.0)
    dense_bonus = min(entry.get("dense_bars_window", 0) / 2.0, 1.0)
    syllables = min((entry.get("avg_syllables_window", 0.0) or 0.0) / 14.0, 1.0)
    quality = max(0.0, min(1.0, 0.5 * density + 0.3 * dense_bonus + 0.2 * syllables))
    score = floor + (ceiling - floor) * quality
    return {key: score for key in SCORE_KEYS}


def read_kaggle_enriched(path: Path, limit: int, floor: float, ceiling: float) -> List[Tuple[str, Dict[str, float]]]:
    records: List[Tuple[str, Dict[str, float]]] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and seen >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not entry.get("english_like", False):
                continue
            if entry.get("section_hint") == "hook":
                continue
            unique = entry.get("unique_rhymes_window", 0)
            if unique < 1:
                continue
            text = entry.get("text", "").strip()
            if not text:
                continue
            target = heuristic_scores(entry, floor=floor, ceiling=ceiling)
            records.append((text, target))
            seen += 1
    return records


def build_embeddings(records: List[Tuple[str, Dict[str, float]]], scorer: SiameseRhymeScorer):
    embs = []
    targets = []
    for text, target in records:
        emb = scorer.embed(text)
        embs.append(emb.cpu())
        targets.append([target[k] for k in SCORE_KEYS])
    embeddings = torch.stack(embs, dim=0)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return embeddings, target_tensor


def train_loop(model, train_loader, val_loader, epochs, device, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for emb, target in train_loader:
            emb = emb.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(emb)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * emb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb, target in val_loader:
                emb = emb.to(device)
                target = target.to(device)
                pred = model(emb)
                loss = criterion(pred, target)
                val_loss += loss.item() * emb.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")


def main():
    args = parse_args()
    settings = load_settings(args.config)
    set_seed(args.seed)

    scored_path = Path(args.scored_dataset or settings.scored_dataset_path)
    siamese_dir = Path(args.siamese_model_dir or settings.siamese_model_dir)
    output_dir = Path(args.output_dir or settings.local_critic_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scored_path.exists():
        raise FileNotFoundError(f"Scored dataset not found at {scored_path}")

    print(f"[INFO] Loading scored dataset from {scored_path}")
    records = read_scored_dataset(scored_path)
    if len(records) < 10:
        raise RuntimeError("Not enough scored records to train the local critic.")
    print(f"[INFO] Loaded {len(records):,} records with critic scores.")

    kaggle_path = Path(args.kaggle_enriched) if args.kaggle_enriched else None
    if kaggle_path and kaggle_path.exists():
        kaggle_records = read_kaggle_enriched(
            kaggle_path,
            limit=args.kaggle_limit,
            floor=args.kaggle_score_floor,
            ceiling=args.kaggle_score_ceiling,
        )
        records.extend(kaggle_records)
        print(f"[INFO] Added {len(kaggle_records):,} Kaggle pseudo-labeled samples (total={len(records):,}).")
    elif kaggle_path:
        print(f"[WARN] Kaggle enriched file not found: {kaggle_path}")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = cache_dir / "embeddings.pt"
    meta_cache = cache_dir / "embeddings.meta.json"
    dataset_sig = {
        "scored_path": str(scored_path.resolve()),
        "mtime": scored_path.stat().st_mtime,
        "size": scored_path.stat().st_size,
        "count": len(records),
        "siamese_dir": str(siamese_dir.resolve()),
    }
    embeddings = targets = None
    if emb_cache.exists() and meta_cache.exists():
        try:
            with open(meta_cache, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta == dataset_sig:
                blob = torch.load(emb_cache)
                embeddings = blob["embeddings"]
                targets = blob["targets"]
                print(f"[INFO] Loaded cached Siamese embeddings from {emb_cache}")
        except Exception as exc:
            print(f"[WARN] Failed to load embedding cache ({emb_cache}): {exc}")
            embeddings = targets = None

    if embeddings is None or targets is None:
        print(f"[INFO] Embedding verses with Siamese encoder from {siamese_dir}")
        scorer = SiameseRhymeScorer(str(siamese_dir))
        embeddings, targets = build_embeddings(records, scorer)
        torch.save({"embeddings": embeddings, "targets": targets}, emb_cache)
        with open(meta_cache, "w", encoding="utf-8") as f:
            json.dump(dataset_sig, f)
        print(f"[INFO] Saved embedding cache to {emb_cache}")

    num_samples = embeddings.size(0)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split_idx = int(num_samples * args.train_split)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_dataset = CriticDataset(embeddings[train_idx], targets[train_idx])
    val_dataset = CriticDataset(embeddings[val_idx], targets[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = embeddings.size(1)
    hidden_layers = [int(dim) for dim in (args.layers or []) if dim is not None]
    if not hidden_layers:
        hidden_layers = [args.hidden_dim]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CriticHead(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=len(SCORE_KEYS),
        activation=args.activation,
        dropout=args.dropout,
    ).to(device)

    train_loop(model, train_loader, val_loader, args.epochs, device, args.lr)

    head_path = output_dir / "reward_head.pt"
    save_head(
        model.cpu(),
        head_path,
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        score_keys=SCORE_KEYS,
        activation=args.activation,
        dropout=args.dropout,
    )
    meta = {
        "scored_dataset": str(scored_path),
        "siamese_model_dir": str(siamese_dir),
        "score_keys": SCORE_KEYS,
        "hidden_layers": hidden_layers,
        "hidden_dim": hidden_layers[0],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "activation": args.activation,
        "dropout": args.dropout,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] Local critic head saved to {head_path}")


if __name__ == "__main__":
    main()
