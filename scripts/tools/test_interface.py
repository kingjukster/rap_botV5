#!/usr/bin/env python
"""
test.py – structure-aware generation sanity check

Modes:
  --mode base  : Qwen2.5-7B-Instruct base only (no LoRA)
  --mode lora  : Base + LoRA from configured adapter directory
  --mode both  : Run base first, then LoRA, same prompt

Prompt pattern matches training corpus:
  <ARTIST=...> <TITLE=...> <SOURCE=...> <COHERENCE=...>
  [BAR] first bar [RHY=...] [SYL_...] [INT_...]
  [BAR]
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config.settings import load_settings

DEVICE = "cuda:0"

def load_model(use_lora: bool, base_model: str, adapter_dir: str, tokenizer_dir: str):
    print("[STEP 1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] pad_token_id = {tokenizer.pad_token_id}")

    print("[STEP 2] Loading 4-bit base model...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": DEVICE},
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    # Ensure embeddings match tokenizer size
    model.resize_token_embeddings(len(tokenizer))
    print(f"[INFO] Resized base embeddings to {len(tokenizer)}")

    if use_lora:
        print("[STEP 3] Attaching LoRA adapter...")
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            device_map={"": DEVICE},
            torch_dtype=torch.float16,
        )
        print("[INFO] LoRA adapter attached.")
    else:
        print("[STEP 3] Skipping LoRA (BASE-ONLY mode).")

    model.eval()
    print(f"[INFO] Model is on device: {DEVICE}")

    # --- Wire EOS to include <END_SONG> if present ---
    eos_ids = []
    if model.config.eos_token_id is not None:
        if isinstance(model.config.eos_token_id, int):
            eos_ids.append(model.config.eos_token_id)
        else:
            eos_ids.extend(model.config.eos_token_id)

    # our structural EOS
    if "<END_SONG>" in tokenizer.get_vocab():
        end_song_id = tokenizer.convert_tokens_to_ids("<END_SONG>")
        eos_ids.append(end_song_id)

    if eos_ids:
        model.generation_config.eos_token_id = eos_ids
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("[GEN CONFIG] eos_token_id =", model.generation_config.eos_token_id)
    print("[GEN CONFIG] pad_token_id =", model.generation_config.pad_token_id)

    return tokenizer, model


# --------------------------------------------------------------------
# Single run (for either base or LoRA)
# --------------------------------------------------------------------
def run_once(label: str, tokenizer, model):
    artist = "Kendrick Lamar"
    seed_text = "Demons in the mirror and pressure on my chest"

    header = (
        f"<ARTIST={artist}> <TITLE=FREESTYLE> "
        f"<SOURCE=custom> <COHERENCE=0.500>"
    )
    first_bar = (
        f"[BAR] {seed_text} [RHY=A] [SYL_11_12] [INT_MED]"
    )

    prompt = header + "\n" + first_bar + "\n[BAR]"

    print(f"\n==================== {label} ====================")
    print("=== PROMPT ===")
    print(prompt)
    print("==============\n")

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attn_mask = enc["attention_mask"].to(DEVICE)

    prompt_len = input_ids.shape[1]
    max_new = 96

    print(f"[INFO] Prompt token length: {prompt_len}")
    print(f"[GEN] max_new_tokens={max_new}")
    print(
        f"[GEN] eos_token_id={model.generation_config.eos_token_id}, "
        f"pad_token_id={model.generation_config.pad_token_id}"
    )

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new,
            do_sample=True,
            top_p=0.90,
            temperature=0.8,
            repetition_penalty=1.05,
        )

    new_ids = out_ids[0, prompt_len:]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=False)

    # Strip Qwen chat end token if it shows up
    new_text_clean = new_text.replace("<|im_end|>", "").strip()

    print("\n=== RAW CONTINUATION (after [BAR]) ===")
    print(repr(new_text_clean))
    print("=====================================\n")

    split = new_text_clean.split("<END_SONG>")[0]
    print("=== CONTINUATION UNTIL <END_SONG> ===")
    print(split)
    print("=====================================\n")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        choices=["base", "lora", "both"],
        default="lora",
        help="Which model(s) to test: base (no LoRA), lora, or both.",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to JSON/YAML config describing paths.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    base_model = settings.base_model_name
    adapter_dir = str(settings.adapter_dir)
    tokenizer_dir = str(settings.tokenizer_dir)

    if args.mode in ("base", "both"):
        tokenizer_base, model_base = load_model(
            use_lora=False,
            base_model=base_model,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
        )
        run_once("BASE MODEL", tokenizer_base, model_base)

    if args.mode in ("lora", "both"):
        tokenizer_lora, model_lora = load_model(
            use_lora=True,
            base_model=base_model,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
        )
        run_once("BASE + LoRA", tokenizer_lora, model_lora)


if __name__ == "__main__":
    main()
