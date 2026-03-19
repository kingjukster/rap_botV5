#!/usr/bin/env python3
"""Analyze run data from DB: status, generations, fitness, failure patterns."""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from evo_rhyme.db import (
    db_enabled,
    list_runs,
    count_runs,
    count_runs_by_status,
    list_generations,
)


def main():
    if not db_enabled():
        print("DB not enabled. Cannot analyze.")
        return
    runs = list_runs(limit=107, offset=0)
    total = count_runs()
    by_status = count_runs_by_status()
    print("=" * 70)
    print("RUN ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nTotal runs: {total}")
    print(f"  completed: {by_status['by_status']['completed']}")
    print(f"  failed:    {by_status['by_status']['failed']}")
    print(f"  running:   {by_status['by_status']['running']}")

    # Per-run stats: gen count, final best_fitness, config keys
    completed = [r for r in runs if r.get("status") == "completed"]
    failed = [r for r in runs if r.get("status") == "failed"]
    running = [r for r in runs if r.get("status") == "running"]

    # Generation stats
    gen_counts = []
    final_fitness = []
    failed_gen_counts = []
    failed_final_fitness = []
    configs_completed = []
    configs_failed = []

    for r in completed:
        gens = list_generations(r["run_id"])
        gen_counts.append(len(gens))
        if gens:
            final_fitness.append(gens[-1].get("best_fitness"))
        configs_completed.append(r.get("config_json") or {})

    for r in failed:
        gens = list_generations(r["run_id"])
        failed_gen_counts.append(len(gens))
        if gens:
            failed_final_fitness.append(gens[-1].get("best_fitness"))
        configs_failed.append(r.get("config_json") or {})

    for r in running:
        gens = list_generations(r["run_id"])

    # Report
    print("\n--- COMPLETED RUNS ---")
    if gen_counts:
        print(f"  Generations: min={min(gen_counts)}, max={max(gen_counts)}, avg={sum(gen_counts)/len(gen_counts):.1f}")
    if final_fitness:
        print(f"  Final best_fitness: min={min(final_fitness):.4f}, max={max(final_fitness):.4f}, avg={sum(final_fitness)/len(final_fitness):.4f}")

    print("\n--- FAILED RUNS ---")
    if failed_gen_counts:
        zero_gen = sum(1 for g in failed_gen_counts if g == 0)
        print(f"  Runs with 0 generations (early crash): {zero_gen} / {len(failed)}")
        nonzero = [g for g in failed_gen_counts if g > 0]
        if nonzero:
            print(f"  Generations (when >0): min={min(nonzero)}, max={max(nonzero)}, avg={sum(nonzero)/len(nonzero):.1f}")
        if failed_final_fitness:
            print(f"  Final best_fitness (when had gens): min={min(failed_final_fitness):.4f}, max={max(failed_final_fitness):.4f}")

    # Config comparison
    print("\n--- CONFIG COMPARISON ---")
    def config_key(c):
        cfg = c or {}
        return (
            cfg.get("generations"),
            cfg.get("population_size"),
            tuple(sorted((k, v) for k, v in (cfg.get("control_snapshot") or {}).items())),
        )
    completed_keys = [config_key(c) for c in configs_completed]
    failed_keys = [config_key(c) for c in configs_failed]
    # Unique configs
    from collections import Counter
    cc = Counter(completed_keys)
    fc = Counter(failed_keys)
    print(f"  Unique configs (completed): {len(cc)}")
    print(f"  Unique configs (failed):    {len(fc)}")
    # Common config in completed
    if cc:
        top = cc.most_common(1)[0]
        print(f"  Most common completed config: gens={top[0][0]}, pop={top[0][1]}, count={top[1]}")

    # Running runs - how many gens?
    print("\n--- RUNNING RUNS (possibly stale) ---")
    for r in running[:15]:
        gens = list_generations(r["run_id"])
        created = r.get("created_at")
        print(f"  run_id={r['run_id']}: {len(gens)} gens, created={created}")

    # Failure rate by hour (if we have enough data)
    print("\n--- RECENT FAILURE RATE (last 40 runs) ---")
    recent = runs[:40]
    r_completed = sum(1 for r in recent if r.get("status") == "completed")
    r_failed = sum(1 for r in recent if r.get("status") == "failed")
    r_running = sum(1 for r in recent if r.get("status") == "running")
    total_recent = len(recent)
    print(f"  completed: {r_completed} ({100*r_completed/total_recent:.0f}%)")
    print(f"  failed:    {r_failed} ({100*r_failed/total_recent:.0f}%)")
    print(f"  running:   {r_running} ({100*r_running/total_recent:.0f}%)")

    # Init-type vs failure
    print("\n--- FAILURE BY INIT TYPE ---")
    init_failed = defaultdict(int)
    init_completed = defaultdict(int)
    for r in runs:
        cfg = r.get("config_json") or {}
        init = cfg.get("init", "?")
        if r.get("status") == "failed":
            init_failed[init] += 1
        elif r.get("status") == "completed":
            init_completed[init] += 1
    for init in sorted(set(init_failed) | set(init_completed)):
        c, f = init_completed[init], init_failed[init]
        total = c + f
        fail_pct = 100 * f / total if total else 0
        print(f"  init={init}: completed={c}, failed={f} (fail rate={fail_pct:.0f}%)")

    # Population size vs failure
    print("\n--- FAILURE BY POPULATION SIZE ---")
    pop_failed = defaultdict(int)
    pop_completed = defaultdict(int)
    for r in runs:
        cfg = r.get("config_json") or {}
        pop = cfg.get("population", "?")
        if r.get("status") == "failed":
            pop_failed[pop] += 1
        elif r.get("status") == "completed":
            pop_completed[pop] += 1
    for pop in sorted(set(pop_failed) | set(pop_completed), key=lambda x: (x == "?", x if isinstance(x, int) else 0)):
        c, f = pop_completed[pop], pop_failed[pop]
        total = c + f
        fail_pct = 100 * f / total if total else 0
        print(f"  pop={pop}: completed={c}, failed={f} (fail rate={fail_pct:.0f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
