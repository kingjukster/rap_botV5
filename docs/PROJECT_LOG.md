# PROJECT LOG — rap_botV5 Evolution System

## Overview

This document consolidates:

* System framing
* Documentation architecture
* Research paper outline
* Experimental design

Purpose:
Transform rap_botV5 into a **reproducible, research-grade evolutionary creativity system**.

---

# 1. SYSTEM FRAMING

## What This Project Is

rap_botV5 is:

> An evolutionary optimization system for structured creative text (rap lyrics)

Core idea:

* Treat rap generation as **search in creative space**
* Optimize across multiple objectives:

  * rhyme quality
  * fluency
  * semantic coherence
  * novelty
  * structure

---

# 2. DOCUMENTATION SYSTEM

## Required Folder

```
docs/
```

## Core Files

### CONTEXT.md (MASTER CONTROL)

Defines:

* system identity
* rules
* references

Key principle:

> Cursor must always anchor to this file

---

### ARCHITECTURE.md

Defines:

* pipeline (generation → scoring → evolution)
* module responsibilities
* system boundaries

---

### DATA_FLOW.md

Defines:

* inputs → transformations → outputs
* randomness points
* reproducibility structure

---

### EVOLUTION_STRATEGY.md

Defines:

* genome representation
* mutation operators
* crossover
* selection
* fitness function

---

### EXPERIMENT_TRACKING.md

Defines:

* run logging structure
* metrics
* experiment naming

---

### CODING_STANDARDS.md

Defines:

* config-driven design
* no hardcoding
* logging requirements
* naming conventions

---

### RESEARCH_LOG.md

Daily log format:

* goal
* change
* result
* insight
* next step

---

### TODO_ROADMAP.md

Defines:

* phases of project evolution
* short-term and long-term goals

---

# 3. RESEARCH PAPER OUTLINE

## Title Options

* Evolving Flow: A Multi-Objective Evolutionary System for Rap Lyric Generation
* Search Over Style: Evolutionary Optimization of Structured Creative Text

---

## Sections

### 1. Abstract

* problem
* method
* results
* contribution

---

### 2. Introduction

* challenges in creative text
* limitations of LLMs
* framing as optimization problem

---

### 3. Related Work

* text generation
* computational creativity
* evolutionary language systems
* MAP-Elites / novelty search

---

### 4. Methodology

#### Pipeline

Initialize → Score → Select → Mutate → Repeat

#### Genome

* structured verse
* lines, tokens, rhyme endings

#### Fitness Function

F = w1*R + w2*Fl + w3*S + w4*N

Components:

* rhyme
* fluency
* semantic similarity
* novelty

#### Operators

* mutation (lexical, rhyme, structural)
* crossover (line-level, phrase-level)

#### Optional

* MAP-Elites
* policy-guided mutation

---

### 5. Experimental Setup

* dataset
* hyperparameters
* baselines
* evaluation metrics

---

### 6. Results

* quantitative metrics
* qualitative examples
* ablation studies

---

### 7. Discussion

* strengths
* limitations
* insights

---

### 8. Future Work

* evolve existing verses
* beat-aware generation
* audio → evolution
* learned fitness

---

### 9. Conclusion

* evolution as viable creative method
* benefits: control, structure, interpretability

---

# 4. EXPERIMENT DESIGN

## Global Evaluation Metrics

### Automatic

* acceptance rate
* mean fitness
* rhyme density
* internal rhyme
* diversity
* duplicate rate
* theme similarity
* syllable deviation

### Human

* quality preference
* coherence
* rhyme strength
* originality

---

## Experiment 0: Baseline

Goal:

* establish reproducible baseline

Setup:

* fixed seeds
* current config
* no changes

Outputs:

* metrics
* logs
* samples

---

## Experiment 1: Rhyme Inventory / Planner

Test:

* improved rhyme grouping + diversity

Conditions:

* current
* normalized
* confidence-aware
* full planner upgrade

Metrics:

* rhyme diversity
* repetition rate
* acceptance

---

## Experiment 2: Seed Filtering / Dedup

Test:

* removing duplicates + repetition

Conditions:

* baseline
* dedup only
* repetition penalty
* full filtering

Metrics:

* duplicates
* lexical diversity
* human freshness rating

---

## Experiment 3: Corpus Cleaning / Tagging

Test:

* better data vs more data

Conditions:

* current corpus
* expanded corpus
* weighted corpus

Metrics:

* rhyme quality
* coherence
* acceptance

---

## Experiment 4: Critic Calibration

Test:

* local vs external scoring

Conditions:

* external only
* local only
* calibrated local
* hybrid

Metrics:

* correlation with external
* ranking agreement
* runtime

---

## Experiment 5: Two-Phase LoRA Training

Test:

* staged adaptation

Conditions:

* current model
* phase A only
* phase B only
* A → B

Metrics:

* quality
* diversity
* human preference

---

## Experiment 6: Scoring Ablation

Test:

* importance of each fitness component

Conditions:

* full
* remove each component individually

Metrics:

* quality drop
* human preference

---

## Experiment 7: Decoding Sensitivity

Test:

* generation parameters vs system improvements

Variables:

* temperature
* top_p
* repetition penalty
* candidates
* attempts

Metrics:

* diversity
* acceptance
* runtime

---

## Experiment 8: Final System Comparison

Compare:

* baseline
* improved pipeline
* full optimized system

Metrics:

* all metrics
* human ranking

---

# 5. RUN ORDER

1. Baseline
2. Rhyme planner
3. Seed filtering
4. Corpus improvements
5. Critic calibration
6. Scoring ablation
7. LoRA training
8. Decoding tuning
9. Final comparison

---

# 6. REPRODUCIBILITY RULES

* fixed seed sets
* log config per run
* log git commit hash
* separate dev vs eval seeds
* store outputs per run

---

# 7. KEY INSIGHT

This system is not:

"a text generator"

It is:

> a search process over structured creative space

That distinction is the foundation of:

* your architecture
* your experiments
* your paper

---

# 8. NEXT STEPS

1. Add docs/ folder
2. Add all markdown files
3. Run baseline experiment
4. Run Experiment 1 (rhyme planner)
5. Log results
6. Iterate

---

# END OF LOG
