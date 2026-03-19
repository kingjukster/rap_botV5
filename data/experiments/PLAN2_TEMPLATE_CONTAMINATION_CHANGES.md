# Plan 2: Template Contamination – Changes Summary

**Date:** 2026-03-19  
**Goal:** Reduce incoherent line mixing (e.g. empire/crown + "shot in the leg") in high-fitness verses.

## Characterization (top_candidates.json)

From `qd_20260318_201102/top_candidates.json`:
- **Recurring orphan patterns:** "They played around, and she caught a shot in the leg" (unrelated to empire/crown)
- **Garbled lines:** "crown the empire with a po sharp as a peg", "flow lai a tempest my shave never dose"
- **Mixed-topic verses** often had coherence 0.39–0.55; coherent verses ~0.69–0.71
- `template_penalty` was 0.0 for many contaminated verses

## Changes Made

### 1. Coherence scoring (`evo_rhyme/scoring/coherence.py`)

- **Added `_min_line_centroid_similarity`** – penalizes verses where any line is a semantic outlier (low cosine similarity to verse centroid). Catches orphan lines.
- **Updated `score_coherence_with_embeddings`** – new blend:
  - consecutive similarity: 0.35 (was 0.5)
  - information gain: 0.25 (was 0.3)
  - structural coherence: 0.15 (was 0.2)
  - **min_line_centroid_similarity: 0.25** (new)

### 2. Fitness weights (`evo_rhyme/fitness.py`)

- **coherence:** 0.18 → **0.22**
- **template_penalty:** -0.15 → **-0.25**
- **`_score_verse_template_penalty`:** Added orphan-line penalty when 2+ lines have theme keywords but 1+ lines have none (penalty 0.4).

### 3. Constraint config (`evo_rhyme/constraints.py`)

- **`DEFAULT_ORPHAN_PHRASES`:** `{"shot in the leg", "played around", "caught a shot"}`
- **`_check_verse_orphan_lines`:** Rejects verses where a line contains an orphan phrase but has no theme keyword (when `prompt_keywords` is set).
- **`ConstraintConfig.orphan_phrases`:** Optional override for orphan phrases.

### 4. Crossover / evolution (`evo_rhyme/verse_evolution.py`, `evo_rhyme/emitters.py`)

- **`constraint_config`** now includes `prompt_keywords` when `theme_keywords` exist, so the orphan check runs on all offspring.
- Offspring that fail the orphan check are rejected; the evolution loop retries.
- **Emitters:** Added `_constraint_config()` helper that includes `prompt_keywords` for orphan check; all emitters use it.

### 5. Audit script (`scripts/audit_template_contamination.py`)

- Samples verses from `top_candidates.json` or `archive.json`.
- Classifies as: coherent, mixed_topic, garbled, borderline.
- Reports % coherent vs mixed-topic.
- Usage: `python scripts/audit_template_contamination.py <path> [--theme crown,empire] [--min-fitness 0.80]`

## Validation

- **Constraint tests:** All 24 tests pass, including new `test_passes_verse_constraints_reject_orphan_line`.
- **Audit baseline:** On existing `top_candidates.json` (fitness≥0.80): 73.5% mixed-topic, 23.5% garbled, 0% coherent.
- **Short run:** `python scripts/run_verse_qd.py --theme "crown,empire" --generations 2 --population 40` completes without regression.
