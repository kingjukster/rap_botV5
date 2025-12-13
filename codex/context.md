# CONTEXT_RAP_BOT.md
# Rap Bot / Blacklight – Ultra‑Expanded Project Context File
(Optimized for Codex, Cursor, and high‑context LLM agents)

---

## 0. Purpose of This Document
This document is the **master context specification** for the Rap Bot / Blacklight generative rap system.  
It is designed so that any AI coding assistant fully understands:

- The complete pipeline architecture  
- How each file in the repository contributes to the system  
- The training, scoring, and generation processes  
- All essential conventions (tokens, rhyme enforcement, scoring rules, etc.)  
- All dependencies and expectations when generating or modifying code  

This version contains **significant expansion beyond the original**:  
More structure, more diagrams, more definitions, more workflow details, and more formal descriptions for Codex‑style reasoning.

---

# 1. System Overview

## 1.1 Vision
Rap Bot / Blacklight is a **multi‑stage expert rap generation engine** focused on:

- Dense multisyllabic rhymes  
- Complex internal rhyming structures  
- Thematic coherence and storytelling  
- Artist‑style emulation (Nas, MF DOOM, Kendrick, etc.)  
- Rhyme‑scheme obedience (AAAA, AABB, ABAB, ABAC, etc.)  
- Advanced scoring loops  
- Reinforcement from an external critic model  
- Pipeline‑driven refinement across generations  

The system combines:
1. **Fine‑tuned Qwen 14B model**  
2. **Rhyme enforcement engine (CSV + Siamese model)**  
3. **Structure‑aware generation loops**  
4. **External critic scoring → weighted dataset → Stage‑3 model training**  

---

# 2. Repository Structure (EXPLICIT & EXPANDED)

```
rap_botV5/
│
├── scripts/
│   ├── generation/                   # Verse + story generators with rhyme + meter control
│   │   ├── generate_rhymed_verse.py
│   │   └── generate_story_verse.py
│   ├── pipeline/
│   │   ├── run_stage3_pipeline.py    # End-to-end Stage-3 orchestration + LoRA refresh hook
│   │   └── consolidate_stage3.py     # Merge generation logs + (local/OpenAI) critic scores
│   ├── tools/                        # CLI utilities (audit, reports, seed mining, tmux starter, etc.)
│   │   ├── update_rhyme_groups.py
│   │   ├── audit_rhyme_groups.py
│   │   ├── score_with_openai.py
│   │   ├── audition_verses.py
│   │   ├── run_stage3_report.py
│   │   ├── run_stage3_cycle.sh
│   │   ├── build_phase_corpus.py
│   │   ├── enrich_kaggle_corpus.py
│   │   ├── mine_seed_density.py
│   │   ├── calibrate_local_critic.py
│   │   └── test_interface.py
│   └── training/
│       ├── build_elite_kaggle_corpus_multi_stage.py
│       ├── clean_elite_corpus.py
│       ├── train_elite_qwen.py
│       ├── train_local_critic.py
│       ├── train_ngram_critic.py
│       ├── train_topic_embeddings.py
│       ├── train_siamese_rhyme_model.py
│       └── train_siamese_rhyme_model_pairs.py
│
├── rapbot/                           # Python helpers imported by generators/pipelines
│   ├── meter_utils.py                # CMUdict syllable estimates + Gaussian meter score
│   ├── ngram_critic.py               # Lightweight n-gram novelty critic
│   ├── reward_model.py               # Local critic head on top of Siamese encoder
│   ├── rhyme_planner.py              # Anchor planner for rhyme schemes
│   ├── rhyme_scorer.py               # Siamese rhyme model wrapper + rhyme groups loader
│   ├── scoring.py                    # Hybrid scoring utilities + offline critic API
│   └── topic_utils.py                # Word2Vec-based TopicScorer
│
├── config/                           # Shared portable config loader + YAML defaults
│   ├── __init__.py
│   ├── rapbot.yaml                   # Reference path + parameter overrides
│   └── settings.py                   # load_settings() with env + config overrides
│
├── data/                             # Corpora, manifests, stats, rhyme CSVs
│   ├── rhymes_grouped.csv + backups
│   ├── elite_kaggle_corpus_clean*.*
│   ├── generated_raw.jsonl
│   ├── critic_scores.jsonl
│   ├── scored_dataset.jsonl
│   ├── weighted_corpus_stage3.txt
│   ├── seeds/ (kaggle + mf doom manifests)
│   ├── critic_manifests/             # run/nightly manifest bookkeeping
│   ├── rhymes/                       # siamese training pairs
│   └── stats/                        # stage3_summary.json + histograms/per-seed CSV
│
├── docs/                             # Human planning docs (improvement_plan.md, etc.)
├── notebooks/                        # Stage-3 stats + rhyme model experiments
├── codex/                            # This context + pod info
├── checkpoints/                      # LoRA adapters + tokenizer snapshots
├── rhyme_siamese/                    # Local rhyme scorer (encoder + tokenizer)
├── rap_structure_qwen.py             # Song structure experiments
├── requirements.txt
└── group                             # Placeholder file used for quick notes
```

Each file below is described in detail.

---

# 3. Models and Learning Components

## 3.1 Base Model: Qwen 2.5–14B
Chosen for:
- Strong multilingual tokenization  
- High instruction fidelity  
- Stable fine‑tuning behavior  
- Excellent narrative and stylistic modeling  

Loaded with:

```python
AutoTokenizer.from_pretrained(model, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(
    model,
    device_map={"": "cuda:0"},
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
```

## 3.2 Added Special Tokens (CRITICAL)
The tokenizer is extended with:

```
[BAR]         # Marks start of bar
[RHY=A-Z]     # Rhyme label for each bar
<END_SONG>    # Marks end of verse/song
```

These tokens **must appear in all training datasets and prompts.**

---

# 4. Training Pipeline (THREE FULL STAGES)

## Stage 1 — Elite Corpus Training (LoRA)
**Goal:** Teach the base model elite rap linguistic patterns.

Uses:
- `elite_kaggle_corpus_clean.txt`

Trains:
- Dense multis  
- Internal rhyme habits  
- Flow and cadence patterns  
- Formal verse structure  

---

## Stage 2 — Rhyme-Aware Generation Model
Builds on Stage 1 with:

- `[BAR]` + `[RHY=X]` token control  
- Rhyme group lookup  
- Siamese rhyme similarity scoring  
- Structured prompts describing rhyme scheme  

This enables **enforced rhyme schemes** during generation.

---

## Stage 3 — Critic-Driven Reinforcement Loop
**Purpose:**  
Push the model beyond imitation into *consistently high-quality lyricism*, guided by an external critic model.

Pipeline:

1. **Generate thousands of verses**  
   - `scripts/generation/generate_rhymed_verse.py` now supports `--log_json` / `--log_dir` (defaults to `data/generated_raw.jsonl`) so every attempt is captured with rhyme/meter metrics, candidate scores, and scheme metadata.
2. **Score each verse using an OpenAI-based harsh critic**  
   - Depth  
   - Coherence  
   - Originality  
   - Line-by-line quality  
3. **Optional: Train the local critic head (`scripts/training/train_local_critic.py`)** using `data/scored_dataset.jsonl` to distill OpenAI judgments into a Siamese+MLP reward model stored under `models/local_critic`.
4. **Run `scripts/pipeline/consolidate_stage3.py`** to merge generation logs + critic outputs *or* local-critic predictions into:
   - `data/scored_dataset.jsonl`
   - `data/weighted_corpus_stage3.txt` (verses repeated per critic-normalized weights)  
5. **Train Stage‑3 refined model**  

This is similar to:
- RLHF
- RLAIF
- Critic-driven reinforcement

But implemented manually via dataset weighting.

---

# 5. The Scoring Model (OpenAI Critic)

## 5.1 Inputs
Raw verse text.

## 5.2 Outputs (MUST BE STRICT JSON)
```json
{
  "overall_score": 0.0,
  "depth_score": 0.0,
  "coherence_score": 0.0,
  "originality_score": 0.0,
  "line_scores": [...],
  "tags": ["dense multis", "weak theme", ...],
  "theme": "revenge, pressure, spiritual conflict",
  "notes": "Short, harsh critique."
}
```

## 5.3 Scoring Philosophy
Harsh, draconian, no undeserved praise.  
Model should:
- Penalize clichés  
- Penalize short bars  
- Reward multis / chains  
- Reward consistent themes  
- Reward narrative clarity  

---

# 6. Weighted Corpus Construction

## 6.1 Weight Formula
Example generalized:

```
score_norm = mean(
  normalize(overall_score),
  normalize(depth_score),
  normalize(coherence_score),
  normalize(originality_score)
)

weight = (score_norm ** 2.4) * scaling_factor
```

High scores lead to exponential repetition in the output dataset.

## 6.2 Output Format
`weighted_corpus_stage3.txt`

Contains many `<END_SONG>` separated verses.  
High-scoring verses appear 10–200+ times.

---

# 7. Rhyme Enforcement System

## Components:
1. **`rhymes_grouped.csv`**  
   Maps words → rhyme cluster ID  
2. **Siamese model**  
   Computes similarity between bar-ending strings  
3. **Rhyme label tags**  
   `[RHY=A]`, `[RHY=B]`, etc.

The generation engine:
- Extracts bar endings  
- Normalizes text  
- Maps to rhyme ID  
- Scores with Siamese model  
- Accepts/rejects candidate bars accordingly  

---

# 8. Generation Engine (Detailed)

## File: `scripts/generation/generate_rhymed_verse.py`

### CLI Parameters
```
--artist
--seed
--scheme
--num_bars
--candidates
--max_new_tokens
--temperature
--top_p
--repetition_penalty
--verse_accept_threshold
```

### Full Algorithm
1. Build prompt  
2. For each bar:
   - Determine rhyme label from scheme  
   - Generate N candidates  
   - Extract rhyme endings  
   - Score rhyme with:
     - rhyme group mapping  
     - Siamese scorer  
   - Rank candidates  
   - Select best  
3. Construct final verse  
4. Apply quality thresholds  
5. Output + log diagnostics  

---

# 9. Story Generator (`generate_story_verse.py`)

Adds:
- Multi-bar thematic tracking  
- Optional story arcs  
- Penalizes incoherent digressions  

Still obeys rhyme structure.

---

# 10. Song Structure Generator (`rap_structure_qwen.py`)

Experiments with generating:
```
[INTRO]
[HOOK]
[VERSE 1]
[VERSE 2]
[BRIDGE]
[OUTRO]
```

Verifies:
- Section coherency  
- Transitions  
- Style retention  

---

# 11. Known Issues & Future Improvements

## Overfitting
Lower LR or smaller steps recommended.

## Repetition
Need n-gram blocking.

## Rhyme Gaps
Improve rhyme-group coverage for slang.
Run `scripts/tools/update_rhyme_groups.py` (optionally via `scripts/training/build_elite_kaggle_corpus_multi_stage.py --auto_expand_rhyme_groups`)
to mine the latest corpus endings and append confidence-weighted entries.

## Critic Bottleneck
Mitigated via `scripts/training/train_local_critic.py` + `reward_model.py`: train a Siamese+MLP head on the scored dataset and run `scripts/pipeline/consolidate_stage3.py --local_critic_head models/local_critic/reward_head.pt` to score verses offline. Use OpenAI critic periodically for calibration.

---

# 12. Config Loader & Defaults

- `config/settings.py` exposes `load_settings(config_path=None)` which merges repo-relative defaults, optional YAML/JSON configs (e.g., `config/rapbot.yaml`), and any `RAPBOT_*` environment overrides before returning a `Settings` dataclass with resolved `Path` objects.
- Every path-like field (adapter dir, rhyme CSV, Siamese dir, logs, stats paths, etc.) is normalized relative to the repo root, so scripts stay portable across machines.
- Nested sections capture runtime knobs: `generation_defaults` (scheme/temperature/candidates/log path), `stage3` (samples per seed, worker count, seed manifests), `critic` (model choice, concurrency, retry policy, token cost estimates), and `stats` (summary JSON, per-seed CSV, histogram JSON). All CLI entrypoints should prefer these settings over hard-coded constants.

---

# 13. Rapbot Helper Modules

- `rapbot/rhyme_planner.py` load-bar logic returns `group -> words` mappings and rotates multiple rhyme groups per scheme letter so anchors do not repeat the same cluster. It honors deterministic seeds and samples per-letter pools to diversify `[RHY=X]` endings.
- `rapbot/meter_utils.py` relies on `pronouncing` + CMUdict to count syllables, falls back to vowel-group heuristics, and scores each bar against a Gaussian target (default 13 syllables) to enforce flow.
- `rapbot/topic_utils.py` wraps a Gensim `Word2Vec` model (trained via `scripts/training/train_topic_embeddings.py`) and computes cosine similarity between generated bars and topic/seed strings for semantic steering.
- `rapbot/ngram_critic.py` trains/loads a small n-gram frequency table (n ≤ 3) from elite corpora. During generation it reports the fraction of seen n-grams to punish bizarre word orders.
- `rapbot/rhyme_scorer.py` centralizes rhyme CSV loading, fallback suffix keys, and the Siamese encoder (auto-tokenizer/model with pooling + normalization). It exposes `SiameseRhymeScorer.embed/embed_batch/score_pair` for reuse in reward modeling and dataset enrichment.
- `rapbot/scoring.py` defines the hybrid scoring stack (`ScoreWeights`, `LengthModel`, `ThemeContext`, `LineFeatures`, LM logprob helper, theme similarity) and exposes `OfflineCritic` which ensembles multiple local critic heads, applies optional calibration JSON, and provides consistent CPU/GPU handling.
- `rapbot/reward_model.py` implements the local critic head (`CriticHead` MLP + LayerNorm), serialization helpers, and `LocalCritic` wrapper that feeds Siamese embeddings through the head to emit overall/depth/coherence/originality estimates.

---

# 14. Stage-3 Orchestration & Logging

- `scripts/pipeline/run_stage3_pipeline.py` chains rhyme CSV refresh, batched verse generation, optional local critic training, consolidation, and even a LoRA refresh/promotion step. It supports JSON/JSONL `--seed_manifest` files (with tags/persona/syllable maps), automatic rhyme refresh detection, in-process generation (model caching) vs subprocess runs, `--parallel_workers` for concurrent sampling, and CLI flags for local critic head/calibration files.
- Seed metadata is normalized via the `SeedSpec` dataclass (tags, persona, theme/style/topic/vocab hints, per-seed sample counts, target syllables, verse acceptance thresholds, etc.), so seeds can drive custom schemes or meter settings.
- `scripts/tools/run_stage3_cycle.sh` wires the whole rhythm: Stage-3 generation, OpenAI critic scoring (with local verse score floor), consolidation + local critic retrain, and stats/HTML refresh.
- `scripts/tools/run_stage3_report.py` executes `notebooks/stage3_stats.ipynb` via Papermill, timestamps the executed notebook, and optionally exports HTML dashboards for quick regression tracking.

### Standard Command Cadence
- `python scripts/generation/generate_rhymed_verse.py --log_json data/generated_raw.jsonl` (default) records every attempt with bar-level metrics, rhyme/meter stats, Siamese verse score, and scheme metadata.
- After critic scoring, run `python scripts/pipeline/consolidate_stage3.py --critic_scores data/critic_scores.jsonl` (plus any custom `--log_path` inputs). You may also pass `--local_critic_head models/local_critic/reward_head.pt` to fill in missing scores entirely offline.
- The consolidator applies the documented weighting formula `(score_norm ** 2.4) * scaling_factor`, ensuring high-scoring verses dominate the Stage-3 corpus without manual spreadsheet work.

---

# 15. Tooling & Diagnostics

- `scripts/tools/score_with_openai.py` streams generation logs through the harsh JSON-only critic with bounded concurrency, retry backoff, manifest logging (tokens/cost/duration), resumable runs (skips verse_ids already in the output), and optional verse_score floors.
- `scripts/tools/audit_rhyme_groups.py` inspects `data/rhymes_grouped.csv` for suffix mismatches per group (majority-vote rhyme keys) so we can purge noisy Siamese assignments quickly.
- `scripts/tools/audition_verses.py` filters `data/scored_dataset.jsonl` across metrics, schemes, personas, tags, or seed IDs and prints formatted verse excerpts; optionally exports a JSONL of curated records for human review.
- `scripts/tools/enrich_kaggle_corpus.py` parses the cleaned Kaggle corpus into a JSONL (`data/elite_kaggle_corpus_clean_plus.jsonl`) with per-bar metadata such as ASCII ratio, section hints, sliding rhyme density, profanity flags, and writes aggregate summaries in `data/stats`.
- `scripts/tools/build_phase_corpus.py` converts the enriched JSONL into Phase-A training sequences by concatenating high-quality bars (filtering by rhyme density and density flags) and appending `<END_SONG>`, ensuring LoRA refreshes can leverage meta-aware slices.
- `scripts/tools/mine_seed_density.py` slides windows across the Kaggle corpus to surface dense rhyming segments with metadata (unique rhymes, dense bar counts, rhyme sequences) and emits ranked JSONL manifests under `data/seeds/`.
- `scripts/tools/calibrate_local_critic.py` scores verses with the trained local critic head, fits per-dimension linear calibration curves against OpenAI critic scores, and saves the resulting JSON for use in `OfflineCritic`.
- `scripts/tools/run_stage3_report.py` (Papermill + nbconvert) and `scripts/tools/run_stage3_cycle.sh` keep nightly refresh + reporting deterministic.
- `scripts/tools/test_interface.py` provides a smoke test for tokenizer/base/QLoRA alignment (ensures `<END_SONG>` is in eos_token_id, tests base vs LoRA outputs, verifies 4-bit loading).
- `scripts/tools/start.sh` bootstraps a tmux session, Python venv, and installs core dependencies (torch/transformers/peft/etc.) so remote pods can be prepared with one command.

---

# 16. Data Assets & Reports

- `data/rhymes_grouped.csv` is continuously backed up (`data/rhymes_grouped.backup.*`) after each refresh; `data/rhymes/` holds positive/negative Siamese training pairs so we can retrain rhyme scorers offline.
- Stage-3 generation logs live in `data/generated_raw.jsonl`, critic outputs in `data/critic_scores.jsonl`, and consolidated data in `data/scored_dataset.jsonl` + `data/weighted_corpus_stage3.txt`.
- `data/seeds/` stores mined Kaggle candidates and MF DOOM manifests; each manifest row tracks window size, rhyme stats, tags, and seed prompts for Stage-3.
- `data/critic_manifests/` maintains per-run JSONL logs that capture API usage (timestamps, token counts, estimated costs) so budgets remain auditable.
- `data/stats/` (`stage3_summary.json`, `stage3_per_seed.csv`, `stage3_histograms.json`, etc.) powers the Stage-3 notebook; the executed outputs are archived via `run_stage3_report.py` under `reports/`.

---

# 17. Improvement Roadmap Snapshot (`docs/improvement_plan.md`)

1. **Rhyme inventory & anchor planner** – normalize endings before frequency counts, store confidence/method/source_count columns in the CSV, drop one-off anchors unless Stage-3 logs need them, and extend `rapbot/rhyme_planner.py` to rotate multiple group pools per rhyme letter (tunable via CLI knobs).
2. **Seed/log filtering + consolidation** – mine Kaggle bars for internal rhyme density, refresh manifests ordered by richness, penalize Stage-3 verses that recycle the same stems/anchors more than N times, and deduplicate `data/generated_raw.jsonl` before critic scoring.
3. **Kaggle corpus cleaning & tagging** – add language detection, section segmentation, rhyme density/metadata tables, export `data/elite_kaggle_corpus_clean_plus.jsonl`, and feed that metadata into Stage-3 weighting instead of hard deletions.
4. **Critic & model training improvements** – refresh the local critic with pseudo-labeled Kaggle samples, run a two-phase LoRA refresh (Phase A = Kaggle verses, Phase B = weighted Stage-3 corpus), then rerun Stage-3 using offline critics for scoring and compare Siamese/critic metrics before promoting adapters.

Each task lists validation steps (stats diffs, rhyme coverage reports, offline critic calibration) so we only retrain LoRA adapters once the upstream signals improve.

---

# 18. Codex Instructions (Critical Behavioral Rules)

### Codex MUST:
- Treat this document as the top-level architectural spec  
- Preserve all special tokens  
- Maintain compatibility with all rhyme systems  
- Keep CLI options stable  
- Use GPU-friendly model loading  
- Resolve filesystem paths via `config/settings.py` (or `--config` overrides) instead of hard-coded locations  

### Codex SHOULD:
- Improve modularity  
- Add validation steps  
- Add diagnostics logs  

### Codex MUST NOT:
- Remove rhyme logic  
- Break tokenizer/token embedding alignment  
- Introduce hard-coded local paths  
- Remove structural tokens `[BAR]`, `[RHY=X]`, `<END_SONG>`  

---

# END OF FILE
