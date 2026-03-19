# Control effect map

Runs: 3

## Policy Progression

Best policy so far: **v20260319133221** (avg_fitness=0.694, runs=1)

| policy_version | runs | avg_fitness | rhyme | flow | semantic | novelty | punchline | delta_vs_prev | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v20260319133221 | 1 | 0.694 | 0.488 | 0.729 | 0.599 | 1.000 | 0.371 | — | baseline |

Note: policy comparisons with fewer than 5 runs are low-confidence.

## elites

| Value | N |
|-------|---|
| 5 | 3 |

## embedding_weight

| Value | N |
|-------|---|
| 0.5 | 3 |

## epsilon

| Value | N |
|-------|---|
| 0.1 | 3 |

## exploration_applied

| Value | N |
|-------|---|
| False | 3 |

## generations

| Value | N |
|-------|---|
| 80 | 1 |
| 8 | 2 |

### Mean difference (fitness)
- **80 vs 8**: Δ = 0.0238, Cohen's d = 1.844

### Bootstrap 95% CI (fitness)
- **80**: mean = 0.6935, CI = [0.6935, 0.6935]
- **8**: mean = 0.6697, CI = [0.6568, 0.6826]

## immigrants

| Value | N |
|-------|---|
| 6 | 3 |

## init

| Value | N |
|-------|---|
| mixed | 3 |

## lm_fluency_weight

| Value | N |
|-------|---|
| 0.5 | 3 |

## min_fluency_accept

| Value | N |
|-------|---|
| 0.0 | 3 |

## min_lexical_accept

| Value | N |
|-------|---|
| 0.0 | 3 |

## min_ngram_fluency_accept

| Value | N |
|-------|---|
| 0.0 | 3 |

## min_semantic_accept

| Value | N |
|-------|---|
| 0.0 | 3 |

## multiobjective

| Value | N |
|-------|---|
| False | 3 |

## policy_hash

| Value | N |
|-------|---|
| c589e98bbf5819c3 | 1 |

## policy_mode

| Value | N |
|-------|---|
| learned | 1 |
| static | 2 |

### Mean difference (fitness)
- **learned vs static**: Δ = 0.0238, Cohen's d = 1.844

### Bootstrap 95% CI (fitness)
- **learned**: mean = 0.6935, CI = [0.6935, 0.6935]
- **static**: mean = 0.6697, CI = [0.6568, 0.6826]

## policy_source

| Value | N |
|-------|---|
| C:\Users\horne\rap_botV5\artifacts\learned_policy.json | 1 |
| defaults | 2 |

### Mean difference (fitness)
- **C:\Users\horne\rap_botV5\artifacts\learned_policy.json vs defaults**: Δ = 0.0238, Cohen's d = 1.844

### Bootstrap 95% CI (fitness)
- **C:\Users\horne\rap_botV5\artifacts\learned_policy.json**: mean = 0.6935, CI = [0.6935, 0.6935]
- **defaults**: mean = 0.6697, CI = [0.6568, 0.6826]

## policy_version

| Value | N |
|-------|---|
| v20260319133221 | 1 |

## population

| Value | N |
|-------|---|
| 100 | 1 |
| 40 | 2 |

### Mean difference (fitness)
- **100 vs 40**: Δ = 0.0238, Cohen's d = 1.844

### Bootstrap 95% CI (fitness)
- **100**: mean = 0.6935, CI = [0.6935, 0.6935]
- **40**: mean = 0.6697, CI = [0.6568, 0.6826]

## require_theme_presence

| Value | N |
|-------|---|
| False | 3 |

## runner

| Value | N |
|-------|---|
| couplet | 3 |

## sampled_policy_rank

| Value | N |
|-------|---|
| 1 | 1 |

## seed_info

| Value | N |
|-------|---|
| {"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1078052143, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}} | 1 |
| {"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540724762, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}} | 1 |
| {"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540723753, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}} | 1 |

### Mean difference (fitness)
- **{"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1078052143, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}} vs {"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540724762, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}}**: Δ = 0.0109, Cohen's d = 0.000
- **{"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1078052143, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}} vs {"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540723753, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}}**: Δ = 0.0368, Cohen's d = 0.000
- **{"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540724762, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}} vs {"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540723753, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}}**: Δ = 0.0259, Cohen's d = 0.000

### Bootstrap 95% CI (fitness)
- **{"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1078052143, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}}**: mean = 0.6935, CI = [0.6935, 0.6935]
- **{"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540724762, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}}**: mean = 0.6826, CI = [0.6826, 0.6826]
- **{"numpy_seeded": true, "python_hash_seed": null, "random_seeded": true, "seed": 1540723753, "torch_cudnn_benchmark": false, "torch_cudnn_deterministic": true, "torch_deterministic": true, "torch_seeded": true, "versions": {"numpy": "1.26.2", "python": "3.12.0", "torch": "2.10.0+cpu"}}**: mean = 0.6568, CI = [0.6568, 0.6568]

## style_weight

| Value | N |
|-------|---|
| 0.1 | 3 |

## theme

| Value | N |
|-------|---|
| pressure,mask,survival | 3 |

## tournament_k

| Value | N |
|-------|---|
| 3 | 3 |

## use_embeddings

| Value | N |
|-------|---|
| False | 3 |

## use_lm_fluency

| Value | N |
|-------|---|
| False | 3 |

## use_niching

| Value | N |
|-------|---|
| False | 3 |

## Correlations (numeric controls)

### elites
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### embedding_weight
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### epsilon
- fitness: r = -0.0000
- rhyme: r = -0.0000
- flow: r = 0.0000
- semantic: r = -0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### exploration_applied
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### generations
- fitness: r = 0.7288
- rhyme: r = 0.9165
- flow: r = -0.9316
- semantic: r = -0.9984
- novelty: r = 0.0000
- punchline: r = -0.5572

### immigrants
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### lm_fluency_weight
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### min_fluency_accept
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### min_lexical_accept
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### min_ngram_fluency_accept
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### min_semantic_accept
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### multiobjective
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### population
- fitness: r = 0.7288
- rhyme: r = 0.9165
- flow: r = -0.9316
- semantic: r = -0.9984
- novelty: r = 0.0000
- punchline: r = -0.5572

### require_theme_presence
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### style_weight
- fitness: r = -0.0000
- rhyme: r = -0.0000
- flow: r = 0.0000
- semantic: r = -0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### tournament_k
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### use_embeddings
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### use_lm_fluency
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000

### use_niching
- fitness: r = 0.0000
- rhyme: r = 0.0000
- flow: r = 0.0000
- semantic: r = 0.0000
- novelty: r = 0.0000
- punchline: r = 0.0000
