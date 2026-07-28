# PRISM scaling roadmap — implementation status

Maps every lever in `PRISM_scaling_roadmap.md` to the code that now implements it,
and records what has actually been measured versus what is scaffolded.

Run order:

```bash
python diagnose_scaling.py                     # Step 1 — diagnose (seconds, no training)
python run_ablation.py --epochs 35              # the feature ablation (§5 Q2)
python eval_hierarchical.py                     # Lever 2 payoff test
python make_ablation_report.py                  # → results/ablation_scaling/REPORT.md
```

---

## Lever 1 — symmetry- and chirality-aware architecture

| Sub-item | Status | Code |
|---|---|---|
| Explicit chiral descriptors (interim fix) | **done + measured** | `src/features/chiral_descriptors.py`; wired into `train_gnn.precompute(include_chiral=True)` |
| Symmetry detection | **done** | `src/evaluation/symmetry.py::detect_symmetry` |
| Predict on the asymmetric unit, tile by symmetry | **done (inference path)** | `symmetry.py::asymmetric_unit_indices`, `tile_order` |
| E(3)-equivariant backbone (full fix) | not started | — |

Three node descriptors (`tetra_chirality`, `helical_sense`, `frame_handedness`)
and one edge descriptor (`det[û_ij, v_i, v_j]`) are pseudoscalars: invariant
under rotation and translation, sign-flipping under reflection.

`diagnose_scaling.py` D2 verifies this empirically rather than by assertion —
it mirrors the coordinates and recomputes both feature families. The current v4
distance-based channels come back **bit-identical** (max |Δ| = 0.00e+00), which
is a direct proof of the roadmap's claim that the blind spot is representational,
not data-limited. The new channels sum to 0.00e+00 with their mirror image, i.e.
an exact sign flip.

Symmetry detection is sequence/UniProt orbits first (always available), plus a
coordinate pass that fits each orbit's chain centroids to a circle and accepts a
`C_k` ring when angular spacing is regular and the radius spread is small. Two
coaxial rings of the same fold are reported as `D_k`.

## Lever 2 — hierarchical / coarse-grained assembly

**Status: done, payoff measured on the frozen large split.**
`src/models/hierarchical.py` — `detect_modules` (weighted greedy modularity on
the buried-surface-area interface graph, resolution-swept to hit a target module
size, singletons absorbed into their strongest neighbour; spectral bisection
fallback when networkx is unavailable), `coarsen` (module-level pseudo-complex),
`induce` (per-module subgraph), `hierarchical_order` (module order ∘ within-module
order). The scorer is shape-agnostic, so the same weights run at both levels —
no retraining, no extra parameters. `eval_hierarchical.py` compares flat vs
hierarchical inference with identical weights.

## Lever 3 — curriculum learning on size

Not implemented. The size-stratified splits it needs (`src/evaluation/splits.py`)
are in place, so this is a training-loop change over an existing ordering.

## Lever 4 — weak / silver labels

| Sub-item | Status | Code |
|---|---|---|
| BSA-heuristic teacher (Marsh/Levy) | **done + quality-measured** | `src/data_prep/bsa_teacher.py` |
| Soft distillation targets | **done (unused by training yet)** | `bsa_teacher.teacher_step_targets` |
| Confidence / tie detection | **done** | `bsa_teacher.bsa_order_confidence` |
| AF3 / Boltz-2 silver structures | not started | — |

The teacher exposes soft per-step targets rather than a hard argmax, and a
per-step margin so genuinely tied steps can be down-weighted instead of
contributing invented labels. `diagnose_scaling.py` D3 measures the teacher's
agreement with the existing labels by size bin — read that before setting a
distillation weight.

## Lever 5 — mine the data that exists at scale

Not started. The corpus is already RCSB + CORUM + Complex Portal; EMDB and
3D-Complex ingestion are unbuilt.

## Lever 6 — evaluation that respects symmetry

**Status: done — this is what the ablation is scored with.**

- `tau_mod_symmetry` — Kendall τ after optimally relabelling chains within each
  orbit. The optimal relabelling is the sorted-to-sorted match, so this is exact
  and O(n log n) rather than a search over the symmetry group.
- `tau_mod_ring` — additionally quotients out cyclic rotation and reflection of
  each detected ring.
- `top1_accuracy_mod_symmetry` — a symmetry-equivalent pick counts as correct.
- `n_gt_orbits` is recorded per complex so complexes where τ_sym is *vacuous*
  (the annotated order lives inside one orbit) can be excluded — reported as
  `tau_sym_nondeg`.
- `src/evaluation/splits.py` — hash-stable size-stratified splits with every
  n ≥ 21 complex frozen out of training, so large-n numbers are extrapolation.

---

## Open questions from §5

1. **Is the ring failure a symmetry-breaking artifact?** — Partially answered.
   The τ → τ_sym → τ_ring ladder in `REPORT.md` quantifies how much of the
   large-n error is symmetry bookkeeping rather than genuine mis-ordering.
2. **Do ESM-2 embeddings carry assembly-order signal?** — This is what
   `run_ablation.py` is built to answer; see `REPORT.md`.
3. **Single in-vivo pathway vs plausible pathway?** — Unresolved, and it is the
   binding constraint on everything above: 97% of the corpus is tier-2 labels
   and 64% carries only a prefix of the order. Every number in `REPORT.md` is
   agreement with those labels.

## Backwards compatibility

`precompute(include_chiral=...)` defaults to `False`, so v4/v5/v6 checkpoints
(node_dim 655, edge_dim 5) load and run unchanged. The ablation opts in and
builds its own 658/6-dim models from scratch.
