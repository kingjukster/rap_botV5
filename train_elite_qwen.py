import os
from typing import List

import torch
import bitsandbytes as bnb  # needed for 4-bit, but we don't call Linear4bit directly
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# === DEFAULT CONFIG (can be overridden by CLI) ===============================

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CORPUS_PATH = "/workspace/rap-botV4/data/elite_kaggle_corpus_clean.txt"
OUTPUT_DIR = "/workspace/rap-botV4/checkpoints/elite_qwen_siamese"
TOKENIZER_SAVE_DIR = "/workspace/rap-botV4/elite_tokenizer"

MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 16
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
SAVE_STEPS = 2000

# Special structural tokens for rhyme / syllables / bars / structure
SPECIAL_TOKENS: List[str] = (
    ["[BAR]", "<END_SONG>"]
    + [f"[RHY={chr(c)}]" for c in range(ord("A"), ord("Z") + 1)]
    + [
        "[SYL_0_6]",
        "[SYL_7_8]",
        "[SYL_9_10]",
        "[SYL_11_12]",
        "[SYL_13_14]",
        "[SYL_15_PLUS]",
        "[INT_NONE]",
        "[INT_MED]",
        "[INT_DENSE]",
    ]
)


# === CLI ARGS ================================================================

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=BASE_MODEL_NAME,
        help="Base Qwen model to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Where to save the LoRA adapter.",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default=CORPUS_PATH,
        help="Path to cleaned elite corpus text file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=PER_DEVICE_BATCH,
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=GRAD_ACCUM_STEPS,
    )
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=NUM_TRAIN_EPOCHS,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MAX_SEQ_LENGTH,
    )

    return parser.parse_args()


# === TOKENIZER + MODEL (4-BIT QLoRA) ========================================

def load_tokenizer_and_model(base_model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"[INFO] Loading tokenizer from {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False,
        trust_remote_code=True,
    )

    # Ensure a pad token exists (needed for Trainer)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pad_token_id = tokenizer.pad_token_id

    # Add our structural tokens if they are not already there
    vocab = tokenizer.get_vocab()
    to_add = [t for t in SPECIAL_TOKENS if t not in vocab]
    if to_add:
        print(f"[INFO] Adding {len(to_add)} special tokens: {to_add}")
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    else:
        print("[INFO] No new special tokens to add.")

    # Save tokenizer so inference and future training use the same vocab
    os.makedirs(TOKENIZER_SAVE_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
    print(f"[INFO] Tokenizer saved to {TOKENIZER_SAVE_DIR}")

    # 4-bit quantization config for QLoRA
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"[INFO] Loading 4-bit model from {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    # Resize embeddings for newly added tokens (including pad + structural)
    print(f"[INFO] Resizing model embeddings to match tokenizer size = {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # Prepare model for k-bit training (QLoRA)
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration: tune attention and MLP projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Ensure pad token is ignored in loss
    model.config.pad_token_id = pad_token_id

    print("[INFO] Model and tokenizer ready.")
    return tokenizer, model


# === DATA PREP ==============================================================

def load_rap_dataset(tokenizer: AutoTokenizer, text_path: str, max_seq_length: int):
    """
    Load the Siamese-annotated corpus as a text dataset and tokenize it
    into contiguous chunks of max_seq_length for causal LM.
    """
    print(f"[INFO] Loading text dataset from: {text_path}")
    raw_dataset = load_dataset(
        "text",
        data_files={"train": text_path},
    )

    def tokenize_function(examples):
        # Just tokenize each line; no truncation here.
        return tokenizer(examples["text"])

    tokenized = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    # Group into blocks of max_seq_length tokens for causal LM
    block_size = max_seq_length

    def group_texts(examples):
        concatenated = {}
        for key in examples.keys():
            concatenated[key] = sum(examples[key], [])
        total_length = len(concatenated["input_ids"])
        # Drop remainder to simplify
        total_length = (total_length // block_size) * block_size
        result = {}
        for key, tokens in concatenated.items():
            tokens = tokens[:total_length]
            result[key] = [
                tokens[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized["train"].map(
        group_texts,
        batched=True,
        batch_size=1000,
    )

    print("[INFO] Tokenized dataset ready.")
    return lm_dataset


# === MAIN TRAINING ENTRYPOINT ==============================================

def main():
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model = load_tokenizer_and_model(args.model_name_or_path)
    train_dataset = load_rap_dataset(
        tokenizer,
        text_path=args.text_path,
        max_seq_length=args.max_seq_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,

        do_eval=False,

        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,

        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),

        load_best_model_at_end=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training complete. Saving final adapter...")
    trainer.save_model(output_dir)

    # Save tokenizer alongside the adapter as well
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Done. LoRA + tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    main()
