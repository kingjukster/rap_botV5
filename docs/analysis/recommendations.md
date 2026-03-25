# Prioritized High-Impact Recommendations

Below are prioritized improvements (high → low) tied to specific files/functions.

## P0 — Critical (research validity + output quality)

1. **Make run reproducibility mandatory (config + seed + git SHA + policy hash in every run artifact).**
   - **Why:** Missing snapshots currently block reliable attribution of performance differences.
   - **Where:** `evo_rhyme/verse_evolution.py` (`VerseQDRunLogger.write_config`), `evo_rhyme/evolution.py` (`EvolutionRunLogger.write_config`), `scripts/run_verse_qd.py`, `scripts/run_couplet_evolution.py`.

2. **Add hard semantic-coherence gate for elites and archive insertion.**
   - **Why:** Prevents fluent-but-incoherent candidates from dominating.
   - **Where:** `evo_rhyme/verse_evolution.py` (before adding to next population/archive), `evo_rhyme/evolution.py` (elite filtering), `evo_rhyme/fitness.py` (coherence threshold helper).

3. **Switch verse optimization from pure weighted sum to constrained multi-objective selection.**
   - **Why:** Reduces metric compensation (rhyme/fluency hiding coherence deficits).
   - **Where:** `evo_rhyme/verse_evolution.py`, `evo_rhyme/selection.py`, `evo_rhyme/fitness.py` (`score_vector`).

4. **Implement semantic-distance-aware crossover pairing.**
   - **Why:** Directly targets mixed-topic contamination.
   - **Where:** `evo_rhyme/verse_evolution.py` (`verse_crossover` parent selection logic), `evo_rhyme/emitters.py` (`MutationEmitter.emit`).

5. **Materialize run metrics redundantly (DB + local JSONL/CSV not under LFS).**
   - **Why:** Prevents total observability loss when DB/LFS unavailable.
   - **Where:** `evo_rhyme/verse_evolution.py` (`VerseQDRunLogger.flush`), `evo_rhyme/evolution.py` logger, `config/evolution.yaml` output policy.

## P1 — High (QD behavior + robustness)

6. **Add occupancy evenness/entropy and redundancy metrics to QD logger.**
   - **Why:** Coverage alone is insufficient for diversity evaluation.
   - **Where:** `evo_rhyme/archive.py` (`summary`), `evo_rhyme/verse_evolution.py` generation logging, `webapp/services/analysis_service.py`.

7. **Introduce semantic novelty archive (embedding-based) alongside structural MAP-Elites niches.**
   - **Why:** Prevents semantically similar outputs filling different structural niches.
   - **Where:** `evo_rhyme/scoring/novelty.py`, `evo_rhyme/verse_evolution.py`, `evo_rhyme/archive.py`.

8. **Adaptive mutation portfolio via bandit credit assignment per operator.**
   - **Why:** Current fixed weights are brittle across themes and run phases.
   - **Where:** `evo_rhyme/mutation.py`, `evo_rhyme/emitters.py`, `evo_rhyme/policy_runtime.py`.

9. **Two-phase curriculum: exploration-first then coherence-tightening schedule.**
   - **Why:** Balances novelty search early and quality refinement late.
   - **Where:** `scripts/run_verse_qd.py` (schedule args), `evo_rhyme/verse_evolution.py` (generation-conditional thresholds/weights).

10. **Archive insertion criterion upgrade: quality + novelty delta, not quality-only replacement.**
    - **Why:** Reduces local quality greed that can suppress stylistic originality.
    - **Where:** `evo_rhyme/archive.py` (`add` replacement rule).

11. **Add explicit narrative-arc objective for 4/16 bar structure.**
    - **Why:** Current coherence is mostly semantic similarity; arc progression is under-modeled.
    - **Where:** `evo_rhyme/fitness.py` (new component), `evo_rhyme/block_archive.py`, `evo_rhyme/verse_builder.py`.

## P2 — Medium-high (policy learning + experimental rigor)

12. **Upgrade learned-policy updater to uncertainty-aware selection (e.g., bootstrap CI/UCB).**
    - **Why:** Avoids overcommitting to noisy top runs.
    - **Where:** `scripts/update_learned_policy.py`, `evo_rhyme/experiment_metrics.py`.

13. **Stratify policy by theme cluster and scheme.**
    - **Why:** A single global best config is unlikely to transfer across creative domains.
    - **Where:** `scripts/update_learned_policy.py`, `evo_rhyme/experiment_controls.py`, `scripts/run_continuous.py`.

14. **Automate regression dashboard for failure modes (collapse, coherence drift, repetition).**
    - **Why:** Turn qualitative failures into tracked KPIs.
    - **Where:** `webapp/services/analysis_service.py`, `webapp/templates/analysis.html`, `scripts/analyze_convergence.py`.

15. **Add deterministic “offline benchmark mode” (no external LM calls) for CI reproducibility.**
    - **Why:** Enables stable ablation comparisons and faster debugging.
    - **Where:** `evo_rhyme/lm_rewriter.py`, `evo_rhyme/lm_proposer.py`, `scripts/run_verse_qd.py` flags, `config/evolution.yaml`.

## P3 — Medium (engineering hygiene)

16. **Enforce schema migrations for DB columns/indexes with versioned migration scripts.**
    - **Why:** Current ad-hoc `ALTER TABLE` checks are practical but fragile over long-term evolution.
    - **Where:** `scripts/init_db.py`, `evo_rhyme/db.py`.

17. **Persist operator-level telemetry (which mutation/crossover produced accepted elites).**
    - **Why:** Essential for understanding search dynamics and tuning operator portfolios.
    - **Where:** `evo_rhyme/mutation.py`, `evo_rhyme/evolution.py`, `evo_rhyme/verse_evolution.py`, DB `generations.extra_json`.

18. **Add lightweight human-eval sampling loop for top elites every N generations.**
    - **Why:** Computational creativity requires human-grounded quality checks beyond proxies.
    - **Where:** `scripts/analyze_control_impact.py` (or new script), `webapp/templates/run_detail.html`.

---

## Suggested execution order (practical)

1. P0-1 + P0-5 (reproducibility + telemetry hardening).
2. P0-2 + P0-4 (coherence gates + semantic-safe crossover).
3. P1-6 + P1-7 (better QD diversity measurement and semantics).
4. P0-3 + P1-8 (multi-objective + adaptive operators).
5. Policy and dashboard upgrades (P2 group).

This sequence maximizes research validity first, then algorithmic quality/diversity gains.

