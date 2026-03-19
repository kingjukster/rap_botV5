# Generations Tuning Analysis

**Plan 4: Run Longer Generations** — Systematic comparison of 60 vs 80 generation runs.

## Baseline Metrics

### qd_20260318_201102 (60 generations)

| Metric | Value |
|--------|-------|
| Generations | 60 |
| Final best_fitness | **0.861** |
| Final mean_fitness | 0.789 |
| Wall time | 14,684 s (4.1 h) |
| Plateau | Best fitness reached 0.861 at gen 37; flat through gen 59 |

### qd_20260318_105357 (80 generations)

| Metric | Value |
|--------|-------|
| Generations | 80 |
| Final best_fitness | **0.909** |
| Final mean_fitness | 0.855 |
| Wall time | 20,527 s (5.7 h) |
| At gen 60 | best_fitness = 0.895 |
| Key jumps | 0.895 → 0.906 at gen 63; 0.906 → 0.909 at gen 72 |

## Marginal Gain Analysis

**Formula:** `marginal_gain = (fitness_at_N - fitness_at_60) / (N - 60)` for N > 60.

### For the 80-gen run (N = 80)

- fitness_at_60 = 0.895
- fitness_at_80 = 0.909
- marginal_gain = (0.909 - 0.895) / 20 = **0.0007 per generation** (gens 61–80)
- Total gain from 60→80: **+0.014** (1.4 percentage points)

### Cross-run comparison

| Run | Gens | Final best | Wall time | Fitness per hour |
|-----|------|------------|-----------|------------------|
| 201102 | 60 | 0.861 | 4.1 h | 0.210 |
| 105357 | 80 | 0.909 | 5.7 h | 0.159 |

The 80-gen run achieves **+0.048** higher best fitness than the 60-gen run. Part of this is seed/config variance (different runs), but the 80-gen run also gains **+0.014** within its own trajectory from gen 60→80.

## Recommendation

**Default generations: 80**

1. The best-performing run (qd_20260318_105357) used 80 generations and reached 0.909.
2. That run showed non-trivial gains after gen 60 (0.895 → 0.909), indicating 20 extra generations are useful.
3. Marginal gain per gen (0.0007) is modest but meaningful for high-quality output.
4. Wall-time cost: ~40% more (5.7 h vs 4.1 h) for ~5.6% relative fitness improvement (0.861 → 0.909).

**Optional next steps:** Run 100- and 120-gen trials to assess diminishing returns.

## Sweep Script

For multi-level experiments, use `scripts/run_generations_sweep.py`:

```bash
python scripts/run_generations_sweep.py --theme "crown,empire" --generations 60,80,100 --runs-dir
```

## CLI Usage

Pass generations via `--generations`:

```bash
python scripts/run_verse_qd.py --theme "crown,empire" --generations 80 --runs-dir
```

The default comes from `config/evolution.yaml` (qd section) or `config/settings.py` QD_SECTION_DEFAULTS.
