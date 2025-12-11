# Project Improvement Roadmap

Detailed action plan to elevate rhyme coverage, data quality, and downstream model performance. Tasks are prioritized roughly in execution order so we can validate each layer before retraining the LoRA adapters.

## 1. Rhyme Inventory & Anchor Planner

Goal: Broaden rhyme coverage from the cleaned Kaggle corpus while retaining quality signals so the planner can sample varied anchors.

Steps:

1. **Corpus Mining Enhancements**
   - Update `scripts/tools/update_rhyme_groups.py` to normalize bar endings (lowercase, strip punctuation, enforce minimum length) before counting frequencies.
   - Keep multiple variants per ending rather than collapsing to a single canonical form; capture confidence per assignment based on Siamese/phonetic agreement.
   - Introduce optional heuristics to drop extremely rare endings (count=1) unless they appear in Stage‑3 logs, so we avoid anchor clutter.
2. **Confidence-Aware CSV**
   - Ensure the rhyme CSV stores `confidence`, `method`, and `source_count` for every row so downstream planners can filter low-trust entries.
   - Emit summary stats (unique words/groups, quartile sizes) for quick sanity checks.
3. **Planner Diversification**
   - Extend `rapbot/rhyme_planner.py` to sample multiple group pools per rhyme letter and rotate across them, favoring high-confidence entries.
   - Allow CLI knobs (`--anchor_groups_per_letter`, `--anchor_random_seed`) to tune diversity.

Validation:
- Regenerate the rhyme CSV from scratch, inspect summary stats, and run verse generation to confirm anchors vary more than the old “wait/await” loops.

## 2. Seed & Log Filtering + Stage‑3 Consolidation

Goal: Reduce duplicate anchors and filler bars before they enter the weighted corpus.

Steps:

1. Build a seed miner that tags Kaggle bars by internal rhyme density (≥2 matching endings) and export a refreshed manifest ordered by richness.
2. Extend `consolidate_stage3.py` to penalize verses that recycle the same stem/anchor more than N times (using stemming or phonetic codes).
3. Add dedup + novelty checks when appending to `data/generated_raw.jsonl` so identical verses aren’t rescored.

Validation:
- Compare `stage3_summary.json` metrics before/after to confirm higher critic scores and lower duplicate counts.

## 3. Kaggle Corpus Cleaning & Tagging

Goal: Retain more songs without sacrificing style.

Steps:

1. Create a preprocessing pipeline (new script/notebook) that:
   - Runs language detection and removes obvious non-English entries.
   - Splits tracks into verse/chorus/interlude sections and filters hooks.
   - Computes rhyme density, syllable counts, and per-bar metadata.
2. Persist outputs as `data/elite_kaggle_corpus_clean_plus.jsonl` (bars + tags) and a summary report.
3. Configure weighting logic for Stage‑3 to upsample dense bars and downsample trivial ones instead of hard deletion.

Validation:
- Spot-check samples, ensure metadata aligns with rhyme planner expectations, and verify that the corpus size increases relative to the previous “elite” subset.

## 4. Critic & Model Training Improvements _(in progress)_

Goal: Only retrain once the upstream data is materially stronger.

Steps:

1. **Local critic refresh**
   - Augment the scored Stage‑3 dataset with pseudo-labeled Kaggle samples (derived from rhyme density stats in `data/elite_kaggle_corpus_clean_plus.jsonl`).
   - Retrain `train_local_critic.py` with the new mix and save both the reward head and calibration JSON (so consolidate + generation can fall back to offline scoring).
2. **Two-phase LoRA refresh**
   - Phase A: fine-tune on the filtered Kaggle verse corpus (`data/phaseA_kaggle_verse.txt`).
   - Phase B: continue training on `data/weighted_corpus_stage3.txt`.
   - Save the resulting adapter/tokenizer under `checkpoints/elite_qwen_siamese/runs/<run>/` and optionally promote it.
3. **Evaluation loop**
   - Re-run Stage‑3 generation using only the local critic to score new verses.
   - Compare Siamese/critic averages, acceptance rate, and sample quality vs. the previous adapter.

Validation checklist:
- Confirm the offline critic reproduces overall/depth/coherence/originality trajectories after calibration.
- Ensure `run_stage3_pipeline.py` can run end-to-end with `--train_local_critic` + `--train_lora_refresh` without requesting OpenAI scores.
- Log before/after verse samples for human spot checks prior to adapter promotion.

---

We’ll iterate task-by-task, validating each layer before moving on. Once Task 1 is stable, we’ll notify before kicking off the heavy retraining (Task 4).
