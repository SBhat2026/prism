# PRISM scaling — ablation + diagnostics report

Generated from `results/ablation_scaling/metrics.json`. Device `mps`, node_dim 658, edge_dim 6.
Split: **658 train / 165 test_small (n≤20) / 68 test_large (n≥21, frozen out of training)**.
Every condition: identical architecture, parameter count, seed, epochs (16) and split — only the feature mask differs.

## Interpolation — held-out small complexes (n ≤ 20)

| condition | top-1 | P(GT) | τ | τ_sym | τ_ring | what it is |
|---|---|---|---|---|---|---|
| `random` | 0.306 | 0.282 | 0.006 | 0.093 | 0.097 | untrained reference |
| `bsa_teacher` | 0.486 | 0.470 | -0.029 | 0.060 | 0.064 | untrained reference |
| `interface_only` | 0.500 | 0.350 | -0.082 | 0.014 | 0.016 | buried-surface-area channels only |
| `esm_only` | 0.602 | 0.525 | 0.186 | 0.306 | 0.306 | sequence only — no interface, no geometry |
| `esm_shuffled` | 0.466 | 0.394 | -0.038 | 0.074 | 0.078 | control — ESM rows permuted across chains within each complex |
| `no_esm` | 0.495 | 0.388 | -0.038 | 0.074 | 0.074 | structure/interface only — ESM node block + esm_cos + go_cos removed |
| `v4_baseline` | 0.610 | 0.525 | 0.159 | 0.273 | 0.276 | current PRISM v4 features — the reference row |
| `v4_plus_chiral` | 0.589 | 0.517 | 0.142 | 0.264 | 0.265 | Lever 1 interim — adds pseudoscalar chirality features |
| `all_features` | 0.597 | 0.522 | 0.166 | 0.284 | 0.286 | all channels including chirality |

## Extrapolation — held-out large complexes (n ≥ 21, never trained on)

| condition | top-1 | P(GT) | τ | τ_sym | τ_ring | what it is |
|---|---|---|---|---|---|---|
| `random` | 0.035 | 0.041 | 0.052 | 0.319 | 0.373 | untrained reference |
| `bsa_teacher` | 0.266 | 0.246 | 0.069 | 0.360 | 0.430 | untrained reference |
| `interface_only` | 0.254 | 0.102 | 0.117 | 0.386 | 0.428 | buried-surface-area channels only |
| `esm_only` | 0.194 | 0.140 | 0.132 | 0.442 | 0.485 | sequence only — no interface, no geometry |
| `esm_shuffled` | 0.194 | 0.106 | 0.032 | 0.372 | 0.404 | control — ESM rows permuted across chains within each complex |
| `no_esm` | 0.188 | 0.100 | 0.149 | 0.371 | 0.414 | structure/interface only — ESM node block + esm_cos + go_cos removed |
| `v4_baseline` | 0.196 | 0.143 | 0.080 | 0.405 | 0.444 | current PRISM v4 features — the reference row |
| `v4_plus_chiral` | 0.194 | 0.138 | 0.127 | 0.454 | 0.487 | Lever 1 interim — adds pseudoscalar chirality features |
| `all_features` | 0.214 | 0.138 | 0.113 | 0.435 | 0.463 | all channels including chirality |

## Degradation with complex size (top-1 next-subunit accuracy)

| condition | n=2-4 | n=5-10 | n=11-20 | n=21-40 | n=41+ |
|---|---|---|---|---|---|
| `random` | 0.385 | 0.319 | 0.151 | 0.040 | 0.010 |
| `bsa_teacher` | 0.481 | 0.521 | 0.318 | 0.294 | 0.107 |
| `interface_only` | 0.519 | 0.520 | 0.374 | 0.279 | 0.107 |
| `esm_only` | 0.769 | 0.610 | 0.375 | 0.187 | 0.234 |
| `esm_shuffled` | 0.538 | 0.485 | 0.291 | 0.210 | 0.102 |
| `no_esm` | 0.538 | 0.527 | 0.289 | 0.203 | 0.100 |
| `v4_baseline` | 0.788 | 0.613 | 0.395 | 0.210 | 0.114 |
| `v4_plus_chiral` | 0.731 | 0.594 | 0.400 | 0.200 | 0.159 |
| `all_features` | 0.808 | 0.596 | 0.360 | 0.223 | 0.159 |

_Counts per bin:_ 2-4=26, 5-10=116, 11-20=23, 21-40=58, 41+=10

## How much of the large-n error is symmetry bookkeeping?

τ → τ_sym → τ_ring progressively quotients out (a) within-orbit chain relabelling and (b) ring rotation/reflection. The gap is error that exact-order scoring charges the model for but which carries no information.

| condition | τ | τ_sym | τ_sym (non-degenerate only) | τ_ring | τ_ring − τ |
|---|---|---|---|---|---|
| `random` | 0.052 | 0.319 | 0.253 | 0.373 | 0.321 |
| `bsa_teacher` | 0.069 | 0.360 | 0.299 | 0.430 | 0.361 |
| `interface_only` | 0.117 | 0.386 | 0.327 | 0.428 | 0.311 |
| `esm_only` | 0.132 | 0.442 | 0.388 | 0.485 | 0.353 |
| `esm_shuffled` | 0.032 | 0.372 | 0.311 | 0.404 | 0.372 |
| `no_esm` | 0.149 | 0.371 | 0.310 | 0.414 | 0.266 |
| `v4_baseline` | 0.080 | 0.405 | 0.348 | 0.444 | 0.364 |
| `v4_plus_chiral` | 0.127 | 0.454 | 0.401 | 0.487 | 0.360 |
| `all_features` | 0.113 | 0.435 | 0.380 | 0.463 | 0.350 |

## Step-1 diagnostics (`diagnose_scaling.py`)

**D1 — symmetry census.** 891 usable complexes. 11.1% have a detected rotational point group overall, but **47.1%** of the n≥21 split does. Asymmetric-unit reduction collapses mean n **31.3 → 12.0**. 8.8% of the large split has an annotated order confined to a single orbit, where exact-order τ is not even well defined.

**D2 — chirality.** invariant channels differ by at most 0.00e+00 Å under reflection (= identical ⇒ architectural blind spot confirmed); chiral channels sum to at most 0.00e+00 with their mirror (= exact sign flip ⇒ handedness is now representable).

**D3 — BSA teacher (Lever 4).** Free-running order agreement with the annotated labels is τ=0.002, top-1=0.195. By size:

| n | complexes | τ | top-1 | steps the rule can't break the tie |
|---|---|---|---|---|
| 2-4 | 135 | 0.000 | 0.257 | 1.2% |
| 5-10 | 533 | -0.038 | 0.184 | 2.8% |
| 11-20 | 155 | 0.113 | 0.209 | 5.6% |
| 21-40 | 58 | 0.064 | 0.111 | 8.5% |
| 41+ | 10 | 0.098 | 0.186 | 13.6% |

**D4 — hierarchical reduction (Lever 2).** Interface-graph community detection takes the n≥21 split from mean n=31.3 to effective n=10.4 across 4.2 modules.

| n | complexes | effective n | fraction of n | GT steps staying inside a module |
|---|---|---|---|---|
| 2-4 | 135 | 4.0 | 100% | 100.0% |
| 5-10 | 533 | 6.1 | 95% | 95.4% |
| 11-20 | 155 | 7.1 | 50% | 65.3% |
| 21-40 | 58 | 9.9 | 36% | 58.9% |
| 41+ | 10 | 13.4 | 27% | 50.4% |

## Caveats

- 97% of the labelled corpus is tier-2 (heuristic/curated, not experimental assembly order) and 64% carries only a prefix of the order; every number here is agreement with those labels, not with in-vivo pathways.
- τ is computed on the annotated subset of the order, with the prediction projected onto that subset preserving predicted relative order.
- Greedy assembly is seeded by the largest-total-buried-area subunit (identical across conditions, no GT leak). `tau_gtseed` in `metrics.csv` reports the historical GT-seeded convention for comparison.
- top-1 and P(GT) are teacher-forced along the annotated order and are therefore seed-independent — they are the cleanest ablation signal.
