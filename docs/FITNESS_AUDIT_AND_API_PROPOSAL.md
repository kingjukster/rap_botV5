# Fitness Audit and Revised API Proposal

## Scope

This audit covers the current scoring paths in `evo_rhyme/fitness.py` plus dependent subscore modules:

- Coherence (`evo_rhyme/scoring/coherence.py`)
- Punchline (`evo_rhyme/scoring/punchline.py`)
- Rhyme chain (`evo_rhyme/scoring/rhyme_chain.py`)
- Rhyme graph network (`evo_rhyme/scoring/rhyme_graph_network.py`)
- Line/cliche penalties (`evo_rhyme/scoring/line_penalty.py`)
- N-gram fluency (`evo_rhyme/ngram_fluency.py`)
- Beat-fit (`evo_rhyme/beat/scoring.py`)

---

## 1) Current fitness (couplet): computation, normalization, weights, penalties

### 1.1 Component formulas and ranges

All couplet components are intended to be in `[0,1]` before weighting.

**Positive components**

- `end_rhyme`: `min(phonetic_similarity(end1,end2)^0.8, 0.92)`.
- `internal_rhyme`: proportion of internal-tail pairs above similarity threshold `0.6`, then multiplied by `2.0` and capped at `1.0`.
- `multisyllabic`: discrete bucket from end-tail overlap: `{0->0.0,1->0.2,2->0.4,3->0.7,>=4->1.0}`.
- `syllable_balance`: `max(0, 1 - abs(s1-s2)/6)`.
- `stress_alignment`: token-position stress match ratio across shared length.
- `semantic`: keyword overlap score (scaled by `*2`) optionally blended with embedding score mapped from cosine `[-1,1]` to `[0,1]`.
- `fluency`: `0.7*(0.6*in_range + 0.4*syllable_balance) + 0.3*valid_word_ratio`.
- `lexical_validity`: in-vocab content ratio; linear to full score at 0.8 ratio.
- `ngram_fluency`: corpus n-gram plausibility score (or blended with LM fluency if enabled).
- `novelty`: `1 - repetition_penalty_raw`.
- `rhyme_graph`: mean of per-line rhyme graph score.

**Penalty components (stored as positive magnitudes in `[0,1]`, then multiplied by negative weights)**

- `weak_tail_penalty`: +0.5 per line ending with unstressed tail.
- `repetition_penalty`: from max repeated content token count: `2->0.5, 3+->1.0`.
- `theme_word_repetition_penalty`: repeated theme token count: `2->0.5, 3+->1.0`.
- `rhyme_family_repetition_penalty`: if consecutive same rhyme family run `>=3`, returns `min(1.0, 0.3 + (run-3)*0.25)`.
- `identical_line_penalty`: 1.0 if lines equal.
- `near_duplicate_penalty`: overlap penalty starts above 0.75 token-set overlap.
- `template_penalty`: 0.5 when same end word and near-equal line length.
- `corpus_overlap_penalty`: max of word-overlap and SequenceMatcher penalties (thresholds 0.68 and 0.85).
- `theme_penalty`: 0.8 when prompt keywords exist but none appear in text.

**Additional diversity penalties added directly in `compute_fitness`**

- `_rhyme_family_diversity_penalty`: negative value in `[-0.5,0]` based on population tail overuse.
- `_repeated_shell_penalty`: negative value up to about `-0.35` when many individuals share same content skeleton.

### 1.2 Couplet weighting and aggregation

Current default weighted sum:

```text
fitness = Σ(weight[k] * score[k])
          + rhyme_family_diversity_penalty(pop)
          + repeated_shell_penalty(pop)
          + style_weight * score_style_similarity(optional)
```

Then:

- Hard gate: if `ngram_fluency < 0.2`, fitness is forced to `0.0`.
- Cap: final fitness is clipped to `<= 0.95`.

Current base weights (`DEFAULT_WEIGHTS`) prioritize rhyme stack plus fluency with substantial negative penalties.

---

## 2) Current fitness (4-line verse): computation, normalization, weights, penalties

### 2.1 Verse components and ranges

Verse scores include:

- Rhyme structure: `rhyme_scheme_score`, `internal_rhyme`, `rhyme_chain_density`, `global_rhyme_chain_score`, `internal_chain_score`, graph metrics.
- Cadence/flow: `syllable_balance`, `flow_alignment`, `flow_continuity_score`, `beat_fit`.
- Language quality: `fluency`, `lm_fluency`, `lexical_validity`.
- Meaning quality: `semantic`, `coherence`, `punchline`, `novelty` (typically injected upstream in pipeline).
- Penalties: duplicates, template rigidity, filler lines, banned phrases, corpus overlap, garbling, cliche, structural repetition, cross-verse repetition.
- Adherence overlays: `style_adherence`, `prompt_adherence`.

Notable normalization:

- Beat aligner raw total is mapped with `tanh(raw_total/12)` then to `[0,1]`.
- Coherence blends four bounded components (consecutive sim, info gain sigmoid, structural coherence gaussian, min centroid similarity).
- Punchline uses tail perplexity ratio scaled by sigmoid around ratio `1.0`.
- Cliche is special-cased in aggregate: `cliche_scaled = min(1.0, cliche_penalty * 80)` before applying negative weight.

### 2.2 Verse weighting and aggregation

`compute_verse_fitness` performs simple weighted sum over available keys using `VERSE_DEFAULT_WEIGHTS`.

There is no explicit post-sum cap here in current implementation. Most components are normalized, but penalties and special scalings can still dominate aggregate behavior.

---

## 3) Where current scoring can reward low-quality but high-scoring lyrics

1. **Set-based overlap for duplication can miss word-order clones.**
   `near_duplicate_penalty` uses token sets; lines with shuffled words can evade strong penalties while sounding repetitive.

2. **Keyword-semantic loophole with weak discourse quality.**
   Couplet semantic score can be lifted by sparse keyword overlap or embedding similarity even when syntax is awkward, especially if fluency floor is barely passed.

3. **Rhyme-dense gibberish can still clear if n-gram model is weakly matched.**
   Internal rhyme + rhyme graph + multisyllabic can stack strongly; if corpus is narrow or repetitive, odd phrases may still get tolerable n-gram scores.

4. **Binary template penalties under-penalize near-template variants.**
   Couplet `template_penalty` is a fixed 0.5 for one heuristic (same end word + similar length), leaving many formulaic patterns insufficiently penalized.

5. **Cliche handling is asymmetric between couplet and verse.**
   Verse has explicit cliche penalty with aggressive scaling in aggregate; couplet path lacks this parallel signal, enabling clichéd couplets to rank relatively high.

6. **Novelty definition is narrow in couplets.**
   `novelty = 1 - repetition_penalty_raw` measures only token repetition, not archive distance, semantic freshness, or phrase rarity.

7. **Hard floors can create cliff behavior.**
   `ngram_fluency < 0.2 -> 0` is effective for blocking nonsense but can be brittle; tiny changes around threshold drastically alter selection probability.

8. **Some penalties are max-based rather than cumulative.**
   For corpus overlap and certain template checks, single-threshold heuristics can miss multi-small issues that together indicate poor quality.

---

## 4) Proposed revised fitness API

## 4.1 Design goals

- Make every component explicit and typed.
- Separate **base quality components** from **penalties**.
- Use consistent normalization contracts.
- Preserve interpretability with a detailed breakdown object.
- Support both couplet and verse through one API.

## 4.2 Proposed data model

```python
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

Unit = Literal["normalized_0_1", "raw", "penalty_0_1"]

@dataclass
class MetricValue:
    name: str
    value: float
    unit: Unit
    weight: float
    weighted: float
    notes: Optional[str] = None

@dataclass
class PenaltyValue:
    name: str
    magnitude: float          # always in [0,1]
    weight: float             # negative
    weighted: float
    trigger: Optional[str] = None

@dataclass
class FitnessBreakdown:
    mode: Literal["couplet", "verse"]
    aggregate: float
    aggregate_capped: float
    gates_passed: Dict[str, bool]
    components: Dict[str, MetricValue]
    penalties: Dict[str, PenaltyValue]
    diagnostics: Dict[str, float] = field(default_factory=dict)
```

## 4.3 Canonical component groups (explicit)

Required top-level components:

- `rhyme`
  - end-rhyme quality
  - internal rhyme density
  - chain/network richness
- `cadence`
  - syllable balance
  - stress/flow alignment
  - beat-fit
- `semantics`
  - theme relevance
  - semantic embedding alignment
- `coherence`
  - local transitions
  - global structure
  - orphan-line suppression
- `novelty`
  - archive embedding distance
  - phrase rarity / anti-cliche
  - anti-template diversity
- `quality_language`
  - lexical validity
  - ngram fluency
  - LM fluency

Explicit penalty namespace:

- `penalties.duplication` (identical, near-duplicate, structural repetition)
- `penalties.template`
- `penalties.corpus_overlap`
- `penalties.garbled`
- `penalties.filler`
- `penalties.theme_stuffing`
- `penalties.rhyme_family_collapse`
- `penalties.cross_archive_repetition`

## 4.4 Revised compute contract

```python
def compute_fitness_v2(
    *,
    candidate,
    mode: Literal["couplet", "verse"],
    context,
    weights,
    gates,
    caps,
) -> FitnessBreakdown:
    ...
```

Behavior:

1. Compute all raw metrics.
2. Normalize each metric with named normalizers (`linear`, `sigmoid`, `gaussian_target`, `piecewise`).
3. Compute weighted component subtotal.
4. Compute weighted penalty subtotal.
5. Apply soft gates (multipliers) before hard gates.
6. Apply optional hard gates (e.g., minimum language quality).
7. Return full `FitnessBreakdown` with both pre/post-cap aggregate.

## 4.5 Recommended weighting template (starting point)

- Base components (sum ~`1.0`)
  - rhyme: `0.20`
  - cadence: `0.16`
  - semantics: `0.14`
  - coherence: `0.18`
  - novelty: `0.17`
  - quality_language: `0.15`

- Penalties (negative budget around `-0.60` worst-case)
  - duplication: `-0.15`
  - template: `-0.10`
  - corpus_overlap: `-0.10`
  - garbled/filler: `-0.12`
  - theme_stuffing: `-0.06`
  - rhyme_family_collapse: `-0.04`
  - cross_archive_repetition: `-0.03`

## 4.6 Anti-gaming changes (key)

- Replace set-overlap duplication checks with sequence-aware metrics (LCS/edit similarity + token n-gram overlap).
- Split `quality_language` into two gates: grammar plausibility and corpus plausibility; require both.
- Upgrade novelty to archive embedding distance + cliche rarity (not just repetition inverse).
- Make template penalty continuous (not single binary trigger), combining end-word repetition, syntactic-template collisions, and opener pattern reuse.
- Convert hard cliff gates (like single `ngram_floor`) into soft decay + final hard safety floor.
- Add consistency penalty when rhyme density is high but language quality is low (cross-term anti-exploit).

---

## 5) Migration plan

1. Add `compute_fitness_v2` alongside existing API.
2. Mirror current metrics into `FitnessBreakdown` without changing behavior (parity mode).
3. Introduce new novelty/coherence penalties and sequence-aware duplicate checks.
4. Run A/B selection tests against current champions and human preference probes.
5. Make v2 default after stability and quality deltas are validated.

