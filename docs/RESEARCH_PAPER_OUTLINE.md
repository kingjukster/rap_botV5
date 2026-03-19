# Research Paper Outline
(See docs/CONTEXT.md)

## Title Options

- Evolving Flow: A Multi-Objective Evolutionary System for Rap Lyric Generation
- Search Over Style: Evolutionary Optimization of Structured Creative Text

---

## Sections

### Abstract

Problem: Creative text generation; limitations of pure LLM approaches. Method: Evolutionary search over structured verse space with multi-objective fitness. Results: High-quality verses with explicit rhyme, fluency, coherence. Contribution: Configurable, interpretable system; MAP-Elites for diversity.

### Introduction

- Challenges in creative text (structure, rhyme, coherence)
- Limitations of LLMs (no explicit control, no search)
- Framing as optimization problem

### Related Work

- Text generation (LLMs, templates)
- Computational creativity
- Evolutionary language systems
- MAP-Elites / novelty search

### Methodology

- **Pipeline:** Initialize → Score → Select → Mutate → Repeat
- **Genome:** VerseIndividual (lines, tokens, rhyme endings, syllable structure)
- **Fitness:** F = w1*R + w2*Fl + w3*S + w4*N (rhyme, fluency, semantic, novelty)
- **Operators:** Lexical, rhyme, structural mutation; line-level and phrase-slice crossover
- **Optional:** MAP-Elites, policy-guided mutation

### Experiments

- Dataset: elite corpus, rhyme groups
- Hyperparameters: population, generations, init
- Baselines: random, template-only
- Metrics: fitness, diversity, human preference

### Results

- Quantitative: fitness curves, archive coverage
- Qualitative: example verses
- Ablation: fitness component removal

### Discussion

- Strengths: control, structure, interpretability
- Limitations: corpus dependence, compute cost

### Future Work

- Evolve existing verses
- Beat-aware generation
- Audio → evolution
- Learned fitness

### Conclusion

Evolution as viable creative method; benefits of search over sampling.
