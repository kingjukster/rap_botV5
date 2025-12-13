# Qwen2.5‑14B Upgrade Guide

Modern Rap Bot runs best on **Qwen/Qwen2.5-14B-Instruct**. This document walks through the hardware expectations, repository settings, and verification steps so you can switch every stage (training, evaluation, generation) over to the larger base model without guesswork. The defaults in `config/rapbot.yaml` and `config/settings.py` already point at 14B, so you only need an override when intentionally comparing against the legacy 7B build.

---

## 1. Hardware & Dependencies

- **GPU**: 48 GB VRAM class (RunPod RTX A6000 or higher). Training relies on 4-bit QLoRA, but inference attaches the adapter to a full-precision base model, so budget ~32–36 GB of VRAM per run.
- **Storage**: ~30 GB free for the base weights plus another ~10 GB for tokenizer/adapters and cached datasets.
- **Software**: Existing environment with PyTorch 2.x, bitsandbytes, transformers, peft. Nothing new is required beyond ensuring `huggingface_hub` is logged in.

Authenticate once so `transformers` can download the private Qwen weights:

```bash
huggingface-cli login --token <your_hf_token>
```

> If your pod only has 24 GB VRAM (A5000 class), stay on 7B or use CPU/NVMe offload; the generation script expects CUDA for the 14B footprint.

---

## 2. Verify Repository Defaults

1. `config/rapbot.yaml` → `base_model_name: Qwen/Qwen2.5-14B-Instruct`
2. `config/settings.py` → `DEFAULTS["base_model_name"] = "Qwen/Qwen2.5-14B-Instruct"`
3. Any CLI call that uses `load_settings()` (training, generation, test utilities, Stage‑3 pipeline) will now resolve the 14B model unless overridden.

No other knobs are required for inference because `scripts/generation/generate_rhymed_verse.py` inspects the LoRA adapter metadata and automatically loads the matching base.

---

## 3. Refresh the LoRA Adapter for 14B

You must retrain the adapter so it is aligned with the new base model. The process matches the previous two-phase recipe; only the base checkpoint changes.

```bash
python scripts/training/train_elite_qwen.py \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --text_path data/elite_kaggle_corpus_clean.txt \
  --phase2_text_path data/weighted_corpus_stage3.txt \
  --output_dir checkpoints/elite_qwen_siamese/runs/refresh_14b \
  --tokenizer_save_dir elite_tokenizer \
  --phase1_epochs 2 \
  --phase2_epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 16
```

Key notes:

- The tokenizer is saved once under `elite_tokenizer`, so downstream scripts will keep the new structural tokens.
- The resulting adapter directory embeds `base_model_name_or_path` inside `adapter_config.json`; generation reads that field, so never mix adapters trained on 7B with a 14B base.
- Expect ~14 hours for a full two-phase run on an A6000 with the default hyperparameters above.

---

## 4. Smoke Tests After Training

Run the thin interface harness to confirm both the base and the adapter can generate bars without OOMs:

```bash
python scripts/tools/test_interface.py --mode both
```

- `--mode base` uses `base_model_name` from the shared config (14B).
- `--mode lora` attaches the refreshed adapter directory.

For a full hybrid-scoring pass:

```bash
python scripts/generation/generate_rhymed_verse.py \
  --config config/rapbot.yaml \
  --scheme AABB \
  --num_bars 16
```

This script enforces CUDA availability for the 14B adapter stack and logs accepted verses to `data/generated_raw.jsonl`.

---

## 5. Stage‑3 / Pipeline Considerations

- No YAML or CLI changes are required for `scripts/pipeline/run_stage3_pipeline.py`; `load_settings()` will propagate the 14B base to every subprocess.
- Keep an eye on GPU temperature and VRAM util when Stage‑3 mixes hybrid scoring, Siamese filtering, and multi-candidate sampling. Increase `stage3.parallel_workers` only if VRAM headroom exceeds ~6 GB per worker.
- If you queue OpenAI critic jobs, nothing changes—the upgrade is isolated to the local LM.

---

## 6. Temporarily Comparing Against 7B

When you need A/B data or have to run on a smaller GPU:

### One-off CLI overrides

```bash
python scripts/training/train_elite_qwen.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --output_dir checkpoints/elite_qwen_siamese/runs/refresh_7b

RAPBOT_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
python scripts/tools/test_interface.py --mode base --config config/rapbot.yaml

RAPBOT_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
python scripts/generation/generate_rhymed_verse.py --scheme ABAB --num_bars 12
```

You can provide the `--model_name_or_path` flag to any training script; generation honors environment variables.

### Environment variable shortcut

```
export RAPBOT_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
```

- Applies to every script that relies on `load_settings()`.
- Useful when running the entire Stage‑3 pipeline on an A5000 pod.
- Remember to `unset RAPBOT_BASE_MODEL` when you are done so the repo reverts to the 14B default.

> The adapter’s metadata still needs to match the selected base. Keep separate adapter directories for 7B and 14B results (e.g., `runs/refresh_7b`, `runs/refresh_14b`) and switch `paths.adapter_dir` in `config/rapbot.yaml` or via `RAPBOT_ADAPTER_DIR` when hopping between them.

---

## 7. Troubleshooting Checklist

- **`CUDA is not available, but GPU is expected for Qwen2.5-14B.`** — Ensure the pod exposes a GPU and `torch.cuda.is_available()` returns true. You cannot run the FP16 generation path on CPU.
- **OOM during generation** — Drop `generation.candidates` or `generation.attempts` in `config/rapbot.yaml`, or run the `test_interface` tool in base-only mode to isolate memory pressure from the LoRA stack.
- **Mismatched tokenizer vocab size** — Delete and rebuild `elite_tokenizer` by rerunning the training script; the repo relies on a shared tokenizer between 7B and 14B adapters.

Once these checks pass, the Rap Bot stack uses Qwen2.5-14B end-to-end by default.
