# Convergence & Overfitting Analysis Report

**Generated:** 2026-03-23  
**Data source:** Database + artifacts/learned_policy.json  

**Running in Docker (no host Python/DB):**
```bash
docker compose --profile evolution run --rm evolution python scripts/analyze_convergence.py --limit 600 --output reports/convergence_report.json
```

---

## Summary

| Question | Answer | Confidence |
|----------|--------|------------|
| **Converging?** | **YES** (plateaued) | Medium |
| **Overfitting?** | **EARLY** | Medium |

---

## Metrics (from notebook execution 2026-03-23)

| Metric | Value |
|--------|-------|
| **Total runs** | 494 |
| **Completed** | 464 (93.9%) |
| **Failed** | 27 (5.5%) |
| **Running** | 3 (0.6%) |
| **Fitness (completed)** | n=494, min=0.0000, max=0.9435, mean=0.7858 |

---

## Policy Behavior

| Field | Value |
|-------|-------|
| **Policy source** | `no_data` |
| **top_configs** | [] (empty) |
| **recommended_controls** | {} |

**Interpretation:** The learned policy has not been populated. `update_learned_policy.py` requires at least 15 runs (default `--min-runs`) and will not write top_configs when `source == "no_data"`. Despite 494 runs existing, the policy file shows no data—this may mean the policy was reset or the update script has not been run recently with sufficient data.

---

## Key Findings

1. **High completion rate (93.9%)** – Most runs complete successfully; failure rate is moderate (5.5%).
2. **Fitness spread** – Mean fitness 0.786, max 0.944. The gap between mean and max suggests room for improvement but also some consistency.
3. **Policy inactive** – Learned policy has no top_configs; elite replay and policy-driven exploration are effectively disabled.
4. **Config diversity** – Without running the full analysis script against the DB, we cannot quantify unique configs or entropy. The script computes:
   - Unique configs in last 50 vs first 50 runs
   - Entropy of config distribution
   - Top 5 configs by fitness and by count

---

## Recommendations

1. **Run the analysis script with DB access** to get precise convergence metrics. Docker:
   ```bash
   docker compose --profile evolution run --rm evolution python scripts/analyze_convergence.py --limit 600 --output reports/convergence_report.json
   ```

2. **Update the learned policy** so elite replay can take effect. When DB runs in Docker, use the evolution container:
   ```bash
   docker compose --profile evolution run --rm evolution python scripts/update_learned_policy.py --force --limit 300
   ```
   Or bootstrap before run_continuous: `--bootstrap-policy` or `--bootstrap-if-empty`.

3. **If diversity is low** (unique configs in last 50 < 10):
   - Increase `--elite-replay-fraction` or epsilon in `run_continuous.py`
   - Add more config arms in `config/continuous_runs.yaml`

4. **If runs since last improvement > 30**:
   - Consider adjusting scoring weights or adding a novelty bonus
   - Re-run `update_learned_policy.py` with `--failure-penalty` to downweight failing configs

5. **If failure rate > 15%** in recent runs:
   - Run `update_learned_policy.py` with a higher `--failure-penalty` (e.g. 1.0)

---

## Analysis Script Usage

```bash
# Full analysis in Docker (DB lives in Docker)
docker compose --profile evolution run --rm evolution python scripts/analyze_convergence.py --limit 600 --output reports/convergence_report.json

# Options
#   --limit N         Max runs to load (default 2000)
#   --output PATH     Write JSON report
#   --rolling-window  Window for failure rate (default 80)
```

## Policy Bootstrap (Docker-only)

When the host has no DB access, policy updates must run in Docker:

```bash
# One-time bootstrap: populate top_configs from DB
docker compose --profile evolution run --rm evolution python scripts/update_learned_policy.py --force --limit 300

# Or bootstrap before run_continuous loop (use --no-docker so evolution runs inline in container)
docker compose --profile evolution run --rm evolution python scripts/run_continuous.py --no-docker --bootstrap-policy --config config/continuous_runs.yaml

# Auto-bootstrap when policy is empty (before first run)
docker compose --profile evolution run --rm evolution python scripts/run_continuous.py --no-docker --bootstrap-if-empty --config config/continuous_runs.yaml
```

The script computes:
- Run-level: total, completed/failed/running, recent failure rate
- Fitness trends: mean/max over time, early vs late halves
- Policy: top_configs, diversity
- Config convergence: unique configs, top by fitness and count
- Diversity: unique last 50/100 vs first 50/100, entropy
- Stagnation: runs since last max fitness improvement
- Failure analysis: failed config hashes and counts
