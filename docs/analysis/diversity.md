# Diversity / QD Behavior Analysis

## 1) Is MAP-Elites used?
Yes.

The verse-QD pipeline explicitly uses a MAP-Elites archive (`MAPElitesArchive`) with behavior-space dimensions selected by `archive_mode`.
An individual occupies a niche coordinate and replaces incumbent only if fitness is higher.

This is canonical MAP-Elites behavior:
- quality within niche,
- diversity across niches.

---

## 2) Behavior-space design

`archive_mode` (CLI / config) selects dimensions; these map to Python helpers in [evo_rhyme/archive.py](evo_rhyme/archive.py):

| `archive_mode` value | Dimension helper |
|---------------------|------------------|
| `default` (or unknown → `None` dims) | `default_verse_dimensions()` — high niche count |
| `style_chain` | `style_chain_dimensions()` — extended style/rhyme-chain space |
| `compact_style` | `compact_style_dimensions()` — 6-axis compact space |
| `ultra_compact` | `ultra_compact_dimensions()` — 729 niches |
| `curriculum_compact` | starts `default_verse_dimensions()`, switches to `compact_style_dimensions()` mid-run |

Legacy doc names like `default_verse_dimensions` refer to these functions, not separate string tokens.

Representative axes include:
- rhyme density,
- chain length / graph structure,
- style tone / narrativity,
- metaphor density,
- sentiment polarity,
- line-length variance, intensity.

This is a meaningful creativity-space decomposition (form, affect, style, structural complexity).

---

## 3) Archive coverage

### What is measured in code
Coverage = `occupied_niches / total_niches`.
Logged each generation via QD run logger (`archive_coverage`, `occupied_niches`).

### What is observable in this checkout
- Local run CSVs are not materialized (LFS pointers), so direct recomputation is not possible here.
- Experiment documentation (`fitness_gap_investigation.md`) reports final archive coverage near **31.8%** for run `qd_20260318_201102`.

Interpretation:
- Coverage is substantial for high-dimensional creative space.
- It also implies significant unfilled behavior regions (expected in expensive language-generation QD).

---

## 4) Niche utilization quality

Positive signs in implementation:
- Archive stores elite per niche (quality pressure).
- Parent sampling from occupied niches creates exploitation signal.
- Emitters include random exploration + directed mutation + structural and LM-rewrite strategies.
- Curriculum and compact archive modes are available for coverage shaping.

Potential limitation:
- `sample_parents` currently samples uniformly from occupied niches (random choices of occupants), which may underemphasize sparse/frontier niches unless emitters explicitly target them.

---

## 5) Redundancy analysis

Within-niche redundancy is controlled by MAP-Elites replacement rule (single elite per cell).
Cross-niche redundancy risks remain if descriptors are weakly discriminative for lexical content.

Existing anti-redundancy mechanisms:
- novelty scores in fitness,
- repetition/template/corpus-overlap penalties,
- near-duplicate penalties,
- optional cross-verse repetition penalties,
- style/prompt genome diversity.

Expected failure mode:
- semantic-near duplicates may occupy different stylistic/structural niches if descriptors are primarily form-based.

---

## 6) QD research assessment

### Strengths
- Real MAP-Elites core implementation (not just random archive logging).
- Multiple descriptor spaces for runtime/coverage tradeoff.
- Emitter architecture is appropriate for expensive domains (LLM + symbolic hybrids).

### Weaknesses
- Coverage and redundancy evaluation depends heavily on DB/log completeness.
- No built-in entropy/evenness metric across dimensions in core loop (coverage alone can hide skewed occupancy).
- Descriptor axes include hand-crafted lexical proxies (e.g., intensity keyword ratio), which may bias creative space.

### Bottom line
The QD subsystem is research-grade in architecture and clearly beyond a vanilla EA, but needs stronger standardized diversity analytics (coverage + occupancy distribution + semantic redundancy metrics) for robust publication-quality evaluation.

