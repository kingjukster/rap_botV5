#!/usr/bin/env python
"""
train_siamese_rhyme_model.py

Train a Siamese encoder that scores rhyme similarity between two bars.

- Uses your rap_lyrics_writers.csv (same corpus as rap-botV4).
- Uses rhymes_grouped.csv (word,group) to build positive rhyme pairs.
- Produces a small model dir: /workspace/rap-botV4/rhyme_siamese

Run (example):

    python train_siamese_rhyme_model.py \
        --lyrics_csv /workspace/rap_lyrics_writers.csv \
        --rhyme_csv /workspace/rap-botV4/rhymes_grouped.csv \
        --out_dir /workspace/rap-botV4/rhyme_siamese

You can adjust epochs / batch size if GPU allows.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import math
import os
import random
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW          # ← FIXED: AdamW now comes from torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup


# -------------------------
# Dataset + helpers
# -------------------------

def get_last_word(text: str) -> str:
    import re
    m = re.findall(r"[a-zA-Z']+", str(text).lower())
    return m[-1] if m else ""

def load_rhyme_groups(rhyme_csv: str):
    df = pd.read_csv(rhyme_csv)
    if "word" not in df.columns or "group" not in df.columns:
        raise ValueError("Expected columns 'word' and 'group' in rhyme CSV.")
    mapping = {}
    for _, row in df.iterrows():
        w = str(row["word"]).strip().lower()
        if not w:
            continue
        try:
            g = int(row["group"])
        except Exception:
            continue
        mapping[w] = g
    print(f"[rhyme] Loaded {len(mapping)} word→group entries.")
    return mapping

def build_bar_groups(lyrics_csv: str, rhyme_groups: dict, max_per_group: int = 4000):
    """
    Group bars by rhyme group of their last word, so we can make positive pairs.
    """
    df = pd.read_csv(lyrics_csv)
    if "lyric" not in df.columns:
        raise ValueError("Expected a 'lyric' column in the lyrics CSV.")

    df = df.dropna(subset=["lyric"]).copy()
    df["lyric"] = df["lyric"].astype(str).str.strip()
    df = df[df["lyric"] != ""]

    group_to_bars = defaultdict(list)
    skipped_no_group = 0

    for _, row in df.iterrows():
        line = row["lyric"]
        last = get_last_word(line)
        g = rhyme_groups.get(last)
        if g is None:
            skipped_no_group += 1
            continue
        group_to_bars[g].append(line)

    # Clip to avoid huge groups dominating training
    for g in list(group_to_bars.keys()):
        if len(group_to_bars[g]) > max_per_group:
            random.shuffle(group_to_bars[g])
            group_to_bars[g] = group_to_bars[g][:max_per_group]

    print(f"[data] Built {len(group_to_bars)} rhyme-groups with bars.")
    print(f"[data] Skipped {skipped_no_group} lines with no rhyme-group mapping.")
    return group_to_bars

class RhymePairsDataset(Dataset):
    """
    Produces (text1, text2, label) pairs.
    label = 1 for rhyming, -1 for non-rhyming.
    """

    def __init__(self, group_to_bars, num_pairs_per_group=3000, neg_ratio=1.0):
        self.pairs = []
        self.labels = []

        groups = list(group_to_bars.keys())
        random.shuffle(groups)

        # Positive pairs
        for g in groups:
            bars = group_to_bars[g]
            if len(bars) < 2:
                continue
            n = min(num_pairs_per_group, len(bars) * (len(bars) - 1) // 2)
            for _ in range(n):
                a, b = random.sample(bars, 2)
                self.pairs.append((a, b))
                self.labels.append(1)

        num_pos = len(self.pairs)
        num_neg = int(num_pos * neg_ratio)

        # Negative pairs (bars from different rhyme groups)
        print(f"[pairs] Positive pairs: {num_pos}, generating {num_neg} negative pairs.")
        all_bars = [(g, b) for g, bars in group_to_bars.items() for b in bars]
        for _ in range(num_neg):
            (g1, b1), (g2, b2) = random.sample(all_bars, 2)
            while g1 == g2:
                (g2, b2) = random.choice(all_bars)
            self.pairs.append((b1, b2))
            self.labels.append(-1)

        c = list(zip(self.pairs, self.labels))
        random.shuffle(c)
        self.pairs, self.labels = zip(*c)
        print(f"[pairs] Total pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]


# -------------------------
# Siamese encoder model
# -------------------------

class SiameseRhymeEncoder(nn.Module):
    """
    Wrap a transformer encoder into a Siamese network.

    Outputs cosine similarity between two encoded texts.
    """

    def __init__(self, base_model_name: str = "distilroberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1)      # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom
        emb = self.activation(self.proj(pooled))
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    def forward(self, batch):
        # batch: dict with input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels
        emb1 = self.encode(batch["input_ids_1"], batch["attention_mask_1"])
        emb2 = self.encode(batch["input_ids_2"], batch["attention_mask_2"])
        return emb1, emb2


def collate_fn(batch, tokenizer, max_len=64):
    texts1, texts2, labels = zip(*batch)
    enc1 = tokenizer(list(texts1), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc2 = tokenizer(list(texts2), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {
        "input_ids_1": enc1["input_ids"],
        "attention_mask_1": enc1["attention_mask"],
        "input_ids_2": enc2["input_ids"],
        "attention_mask_2": enc2["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float),
    }


# -------------------------
# Training loop
# -------------------------

def train(
    lyrics_csv: str,
    rhyme_csv: str,
    out_dir: str,
    base_encoder: str = "distilroberta-base",
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.05,
    max_len: int = 64,
):

    os.makedirs(out_dir, exist_ok=True)

    rhyme_groups = load_rhyme_groups(rhyme_csv)
    group_to_bars = build_bar_groups(lyrics_csv, rhyme_groups)

    dataset = RhymePairsDataset(group_to_bars, num_pairs_per_group=2000, neg_ratio=1.0)

    tokenizer = AutoTokenizer.from_pretrained(base_encoder)
    model = SiameseRhymeEncoder(base_encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_len=max_len),
    )

    total_steps = len(loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CosineEmbeddingLoss(margin=0.1)

    model.train()
    step = 0
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            emb1, emb2 = model(batch)
            # CosineEmbeddingLoss expects labels in {-1, 1}
            loss = criterion(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % 100 == 0:
                avg = running_loss / 100
                print(f"[epoch {epoch}] step {step}/{total_steps} - loss {avg:.4f}")
                running_loss = 0.0

        # Save checkpoint per epoch
        ckpt_dir = os.path.join(out_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.encoder.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "siamese_state.pt"))
        print(f"[save] Saved epoch {epoch} to {ckpt_dir}")

    # Save final
    model.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "siamese_state.pt"))
    print(f"[done] Final model saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lyrics_csv", type=str, required=True)
    p.add_argument("--rhyme_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--base_encoder", type=str, default="distilroberta-base")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        lyrics_csv=args.lyrics_csv,
        rhyme_csv=args.rhyme_csv,
        out_dir=args.out_dir,
        base_encoder=args.base_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
    )
