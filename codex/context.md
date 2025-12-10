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
1. **Fine‑tuned Qwen 7B model**  
2. **Rhyme enforcement engine (CSV + Siamese model)**  
3. **Structure‑aware generation loops**  
4. **External critic scoring → weighted dataset → Stage‑3 model training**  

---

# 2. Repository Structure (EXPLICIT & EXPANDED)

```
rap-botV4/
│
├── scripts/
│   ├── generation/
│   │   ├── generate_rhymed_verse.py   # Main structured generation engine
│   │   └── generate_story_verse.py    # Narrative & story-focused generator
│   ├── pipeline/
│   │   ├── run_stage3_pipeline.py     # End-to-end Stage-3 orchestration
│   │   └── consolidate_stage3.py      # Merge generation logs + critic scores into Stage-3 datasets
│   ├── tools/
│   │   ├── update_rhyme_groups.py     # Auto expands rhyme CSV via pronouncing + Siamese
│   │   └── score_with_openai.py       # Queries OpenAI for critic scores
│   └── training/
│       ├── build_elite_kaggle_corpus_multi_stage.py
│       ├── clean_elite_corpus.py
│       ├── train_elite_qwen.py
│       ├── train_local_critic.py
│       ├── train_ngram_critic.py
│       ├── train_topic_embeddings.py
│       ├── train_siamese_rhyme_model.py
│       └── train_siamese_rhyme_model_pairs.py
├── rap_structure_qwen.py            # Song structure experiments
├── config/                          # Shared portable config loader
│   └── settings.py                  # Resolves model/data paths via env/config files
├── reward_model.py                  # Local reward model (Siamese encoder + MLP head)
├── test_interface.py                # Model load / smoke test
│
├── rhyme_siamese/                   # Local rhyme scorer (Siamese model)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
│
├── rhymes_grouped.csv               # Word → rhyme-group lookup
│
├── data/
│   ├── elite_kaggle_corpus_clean.txt
│   ├── original_lyrics.txt
│   ├── generated_raw.jsonl
│   ├── scored_dataset.jsonl
│   └── weighted_corpus_stage3.txt
│
├── lora_elite/
├── lora_elite_v2/
│
└── rap_bot_dev.ipynb                # Stage-3 pipeline n̲o̲t̲e̲b̲o̲o̲k̲ (critic + weighting + training)
```

Each file below is described in detail.

---

# 3. Models and Learning Components

## 3.1 Base Model: Qwen 2.5–7B
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

# 12. Codex Instructions (Critical Behavioral Rules)

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
### Stage-3 Logging & Consolidation
- `python scripts/generation/generate_rhymed_verse.py --log_json data/generated_raw.jsonl` (default) records every attempt with bar-level metrics, rhyme/meter stats, Siamese verse score, and scheme metadata.
- After critic scoring, run `python scripts/pipeline/consolidate_stage3.py --critic_scores data/critic_scores.jsonl` (plus any custom `--log_path` inputs). You may also pass `--local_critic_head models/local_critic/reward_head.pt` to fill in missing scores entirely offline.
- The consolidator applies the documented weighting formula `(score_norm ** 2.4) * scaling_factor`, ensuring high-scoring verses dominate the Stage-3 corpus without manual spreadsheet work.
