#!/usr/bin/env python
"""
train_siamese_rhyme_model_pairs.py

End-to-end script that:

  1) Loads a text corpus (by default: your Pass A top-tier bars CSV).
  2) Builds a vocabulary with CountVectorizer (like download_data.ipynb).
  3) For each vocab word, calls the Datamuse rhyme API to build rhyme pairs:
        rhyme_df:  (word_a, word_b, rhyme=1, rhyme_group_id, rhyme_id)
  4) Builds non-rhyme pairs by sampling word_b from *other* rhyme groups:
        non_rhyme_df: (word_a, word_b, rhyme=0, rhyme_group_id)
  5) Trains a Siamese encoder using CosineEmbeddingLoss on these pairs.
  6) Saves the encoder + tokenizer to --out_dir so SiameseRhymeScorer can load it.

Example usage:

  python train_siamese_rhyme_model_pairs.py \
    --bars_csv /workspace/rap-botV4/data/elite_kaggle_lines_clean_top_tier_bars.csv \
    --text_column line_text \
    --pairs_out_dir /workspace/rap-botV4/data/rhymes \
    --out_dir /workspace/rap-botV4/rhyme_siamese \
    --base_encoder distilroberta-base \
    --max_vocab_words 5000 \
    --epochs 1 \
    --batch_size 32
"""

import argparse
import itertools
import math
import os
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# 1. Vocabulary & rhyme pair construction (pulled from download_data.ipynb)
# ---------------------------------------------------------------------------

def get_vocab(corpus: List[str]) -> List[str]:
    """
    Build a vocabulary from a corpus of strings using CountVectorizer,
    then filter to words with length > 2 and non-digit.
    Mirrors notebook logic.
    """
    vectorizer = CountVectorizer()
    _ = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    vocab = [w for w in vocab if len(w) > 2 and not w.isdigit()]
    return vocab


def get_rhymes(word: str) -> List[dict]:
    """
    Datamuse API: words that rhyme with `word`.

    Same logic as notebook:
      https://api.datamuse.com/words?rel_rhy=<word>
    """
    url = f"https://api.datamuse.com/words?rel_rhy={word.lower()}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_rhyme_and_nonrhyme_pairs_from_vocab(
    vocab: List[str],
    pairs_out_dir: str,
    save_every: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-creates the rhyme_df / non_rhyme_df logic from download_data.ipynb:

      - For each vocab word:
          - Call Datamuse to get rhyming words.
          - Build all pair combinations within that rhyme set.
          - Add to rhyme_data with columns:
                rhyme_id, rhyme_group_id, word_a, word_b, rhyme=1

      - After collecting all groups:
          - Build non_rhyme_df by, for each rhyme_group_id:
              - Sample the same number of word_b's from OTHER groups.
              - Mark rhyme=0.

      - Saves rhyme_df.csv and non_rhyme_df.csv to pairs_out_dir.
    """
    os.makedirs(pairs_out_dir, exist_ok=True)
    rhyme_data = []
    rhyme_id = 1
    rhyme_group_id = 1

    print(f"[rhyme] Building rhyme pairs for {len(vocab)} vocab words...")
    for idx, word in enumerate(tqdm(vocab, desc="Datamuse rhyme fetch")):
        try:
            rhyme_response = get_rhymes(word)
        except Exception as e:
            print(f"[WARN] Failed to fetch rhymes for '{word}': {e}")
            continue

        if len(rhyme_response) > 0:
            # list all words returned by the response
            rhyming_words = [r["word"] for r in rhyme_response] + [word]
            # Create all pair combinations within this rhyme set
            all_rhyme_combinations = list(itertools.combinations(rhyming_words, 2))

            for rhyme_pair in all_rhyme_combinations:
                rhyme_data.append(
                    {
                        "rhyme_id": rhyme_id,
                        "rhyme_group_id": rhyme_group_id,
                        "word_a": rhyme_pair[0],
                        "word_b": rhyme_pair[1],
                        "rhyme": 1,
                    }
                )
                rhyme_id += 1

            rhyme_group_id += 1

        # Periodic checkpoint (like notebook: every 500 groups)
        if rhyme_group_id % save_every == 0 and rhyme_group_id != 0:
            tmp_df = pd.DataFrame(rhyme_data)
            tmp_path = os.path.join(pairs_out_dir, "rhyme_df_partial.pkl")
            tmp_df.to_pickle(tmp_path)
            print(f"[rhyme] Saved partial rhyme_df to {tmp_path} (groups: {rhyme_group_id})")

        # Be a bit nice to Datamuse
        time.sleep(0.05)

    rhyme_df = pd.DataFrame(rhyme_data)
    if rhyme_df.empty:
        raise RuntimeError("No rhyme pairs built from vocab; rhyme_df is empty.")

    # Drop duplicate pairs (notebook logic)
    rhyme_df = rhyme_df.drop_duplicates(subset=["word_a", "word_b"], keep="first")

    rhyme_csv = os.path.join(pairs_out_dir, "rhyme_df.csv")
    rhyme_df.to_csv(rhyme_csv, index=False)
    print(f"[rhyme] Final rhyme_df saved to {rhyme_csv} (rows={len(rhyme_df)})")

    # --- Build non-rhyme_df (notebook logic) ---
    non_rhyme_df = rhyme_df.copy()
    print("[non-rhyme] Building negative (non-rhyme) pairs...")
    for rhyme_group in tqdm(
        list(rhyme_df["rhyme_group_id"].drop_duplicates()),
        desc="Non-rhyme sampling",
    ):
        words_in_group = len(
            rhyme_df.loc[rhyme_df["rhyme_group_id"] == rhyme_group]
        )

        # sample same number of word_b's from *other* groups
        other_samples = list(
            non_rhyme_df.loc[
                non_rhyme_df["rhyme_group_id"] != rhyme_group, "word_b"
            ].sample(words_in_group, replace=True)
        )

        non_rhyme_df.loc[
            non_rhyme_df["rhyme_group_id"] == rhyme_group, "word_b"
        ] = other_samples

    non_rhyme_df["rhyme"] = 0
    non_rhyme_df = non_rhyme_df.drop_duplicates(
        subset=["word_a", "word_b"], keep="first"
    )

    non_rhyme_csv = os.path.join(pairs_out_dir, "non_rhyme_df.csv")
    non_rhyme_df.to_csv(non_rhyme_csv, index=False)
    print(
        f"[non-rhyme] Final non_rhyme_df saved to {non_rhyme_csv} "
        f"(rows={len(non_rhyme_df)})"
    )

    return rhyme_df, non_rhyme_df


# ---------------------------------------------------------------------------
# 2. Dataset for Siamese training
# ---------------------------------------------------------------------------

class RhymePairDataset(Dataset):
    """
    Dataset that combines rhyme_df (label=+1) and non_rhyme_df (label=-1)
    into (word_a, word_b, label) tuples.
    """

    def __init__(
        self,
        rhyme_df: pd.DataFrame,
        non_rhyme_df: pd.DataFrame,
        max_pos: int | None = None,
        neg_ratio: float = 1.0,
    ):
        self.pairs: List[Tuple[str, str]] = []
        self.labels: List[int] = []

        # --- positive pairs ---
        pos_df = rhyme_df.dropna(subset=["word_a", "word_b"]).copy()
        pos_df["word_a"] = pos_df["word_a"].astype(str).str.strip()
        pos_df["word_b"] = pos_df["word_b"].astype(str).str.strip()
        pos_df = pos_df[(pos_df["word_a"] != "") & (pos_df["word_b"] != "")]

        if max_pos is not None and max_pos > 0 and len(pos_df) > max_pos:
            pos_df = pos_df.sample(max_pos, random_state=42)

        for _, row in pos_df.iterrows():
            self.pairs.append((row["word_a"], row["word_b"]))
            self.labels.append(1)

        num_pos = len(pos_df)
        print(f"[dataset] Positive (rhyme) pairs: {num_pos}")

        # --- negative pairs ---
        neg_df = non_rhyme_df.dropna(subset=["word_a", "word_b"]).copy()
        neg_df["word_a"] = neg_df["word_a"].astype(str).str.strip()
        neg_df["word_b"] = neg_df["word_b"].astype(str).str.strip()
        neg_df = neg_df[(neg_df["word_a"] != "") & (neg_df["word_b"] != "")]

        target_neg = int(num_pos * neg_ratio)
        if target_neg > 0 and len(neg_df) > target_neg:
            neg_df = neg_df.sample(target_neg, random_state=123)

        for _, row in neg_df.iterrows():
            self.pairs.append((row["word_a"], row["word_b"]))
            self.labels.append(-1)

        print(f"[dataset] Negative (non-rhyme) pairs: {len(self.pairs) - num_pos}")

        # Shuffle
        combined = list(zip(self.pairs, self.labels))
        random.shuffle(combined)
        self.pairs, self.labels = zip(*combined)
        print(f"[dataset] Total pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        label = self.labels[idx]
        return a, b, label


def collate_fn(batch, tokenizer, max_len=32):
    texts1, texts2, labels = zip(*batch)
    enc1 = tokenizer(
        list(texts1),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc2 = tokenizer(
        list(texts2),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "input_ids_1": enc1["input_ids"],
        "attention_mask_1": enc1["attention_mask"],
        "input_ids_2": enc2["input_ids"],
        "attention_mask_2": enc2["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float),
    }


# ---------------------------------------------------------------------------
# 3. Siamese training (encoder-only, compatible with SiameseRhymeScorer)
# ---------------------------------------------------------------------------

def train_siamese_pairs(
    rhyme_df: pd.DataFrame,
    non_rhyme_df: pd.DataFrame,
    out_dir: str,
    base_encoder: str = "distilroberta-base",
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.05,
    max_len: int = 32,
    max_pos: int | None = None,
    neg_ratio: float = 1.0,
):
    os.makedirs(out_dir, exist_ok=True)

    dataset = RhymePairDataset(
        rhyme_df=rhyme_df,
        non_rhyme_df=non_rhyme_df,
        max_pos=max_pos,
        neg_ratio=neg_ratio,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_encoder)
    encoder = AutoModel.from_pretrained(base_encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_len=max_len),
    )

    total_steps = epochs * math.ceil(len(dataset) / batch_size)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(encoder.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CosineEmbeddingLoss(margin=0.1)

    print(
        f"[train] device={device}, base_encoder={base_encoder}, "
        f"epochs={epochs}, steps={total_steps}"
    )

    encoder.train()
    step = 0
    running_loss = 0.0

    for epoch in range(1, epochs + 1):
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            # Encode both sides and do mean pooling
            outputs1 = encoder(
                input_ids=batch["input_ids_1"],
                attention_mask=batch["attention_mask_1"],
            )
            outputs2 = encoder(
                input_ids=batch["input_ids_2"],
                attention_mask=batch["attention_mask_2"],
            )

            def mean_pool(last_hidden, mask):
                mask = mask.unsqueeze(-1)  # [B, T, 1]
                summed = (last_hidden * mask).sum(dim=1)  # [B, H]
                denom = mask.sum(dim=1).clamp(min=1e-6)   # [B, 1]
                return summed / denom

            emb1 = mean_pool(outputs1.last_hidden_state, batch["attention_mask_1"])
            emb2 = mean_pool(outputs2.last_hidden_state, batch["attention_mask_2"])

            # CosineEmbeddingLoss expects labels in {-1, 1}
            loss = criterion(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % 100 == 0:
                avg = running_loss / 100
                print(f"[epoch {epoch}] step {step}/{total_steps} - loss {avg:.4f}")
                running_loss = 0.0

        # Save checkpoint per epoch
        epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        encoder.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"[save] Saved epoch {epoch} encoder+tokenizer to {epoch_dir}")

    # Final: also save to the root out_dir so SiameseRhymeScorer can point there
    encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[save] Final encoder+tokenizer saved to {out_dir}")


# ---------------------------------------------------------------------------
# 4. CLI & orchestration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build rhyme/non-rhyme pairs from corpus and train a Siamese rhyme encoder."
    )
    p.add_argument(
        "--bars_csv",
        type=str,
        required=True,
        help="Path to Pass A bars CSV (e.g. elite_kaggle_lines_clean_top_tier_bars.csv).",
    )
    p.add_argument(
        "--text_column",
        type=str,
        default="line_text",
        help="Column in bars_csv containing the bar text.",
    )
    p.add_argument(
        "--pairs_out_dir",
        type=str,
        default="./data/rhymes",
        help="Directory to save rhyme_df.csv and non_rhyme_df.csv.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for the Siamese encoder (for SiameseRhymeScorer).",
    )
    p.add_argument(
        "--base_encoder",
        type=str,
        default="distilroberta-base",
        help="HuggingFace encoder model name.",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument(
        "--max_vocab_words",
        type=int,
        default=5000,
        help="Max number of vocab words to use for Datamuse rhyme queries (for runtime control).",
    )
    p.add_argument(
        "--max_pos_pairs",
        type=int,
        default=None,
        help="Optional cap on positive pairs for training.",
    )
    p.add_argument(
        "--neg_ratio",
        type=float,
        default=1.0,
        help="Negative-to-positive ratio for training dataset.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --- Paths for rhyme pair files ---
    pairs_out_dir = args.pairs_out_dir
    os.makedirs(pairs_out_dir, exist_ok=True)
    rhyme_csv = os.path.join(pairs_out_dir, "rhyme_df.csv")
    non_rhyme_csv = os.path.join(pairs_out_dir, "non_rhyme_df.csv")

    # --- Build corpus from bars_csv (always needed for vocab if we rebuild pairs) ---
    print(f"[corpus] Loading bars from {args.bars_csv} (text_column={args.text_column})")
    df = pd.read_csv(args.bars_csv)
    if args.text_column not in df.columns:
        raise ValueError(
            f"text_column='{args.text_column}' not in CSV columns: {list(df.columns)}"
        )

    text_series = df[args.text_column].dropna().astype(str).str.strip()
    text_series = text_series[text_series != ""]
    corpus = text_series.tolist()
    print(f"[corpus] Loaded {len(corpus)} bars for vocab building.")

    # --- Either load existing rhyme/non-rhyme pairs or build them ---
    if os.path.exists(rhyme_csv) and os.path.exists(non_rhyme_csv):
        print(f"[pairs] Found existing rhyme/non-rhyme files in {pairs_out_dir}, loading...")
        rhyme_df = pd.read_csv(rhyme_csv)
        non_rhyme_df = pd.read_csv(non_rhyme_csv)
        print(f"[pairs] Loaded rhyme_df ({len(rhyme_df)} rows) and non_rhyme_df ({len(non_rhyme_df)} rows).")
    else:
        # Build vocab
        vocab = get_vocab(corpus)
        print(f"[vocab] Raw vocab size: {len(vocab)}")

        if args.max_vocab_words is not None and args.max_vocab_words > 0:
            vocab = vocab[: args.max_vocab_words]
            print(f"[vocab] Truncated to first {len(vocab)} words for rhyme pair building.")

        # Build rhyme_df and non_rhyme_df using notebook logic
        rhyme_df, non_rhyme_df = build_rhyme_and_nonrhyme_pairs_from_vocab(
            vocab=vocab,
            pairs_out_dir=pairs_out_dir,
            save_every=500,
        )

    # --- Train Siamese on these pairs ---
    train_siamese_pairs(
        rhyme_df=rhyme_df,
        non_rhyme_df=non_rhyme_df,
        out_dir=args.out_dir,
        base_encoder=args.base_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_len=args.max_len,
        max_pos=args.max_pos_pairs,
        neg_ratio=args.neg_ratio,
    )



if __name__ == "__main__":
    main()
