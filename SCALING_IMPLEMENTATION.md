# PRISM scaling roadmap — implementation status

Maps every lever in `PRISM_scaling_roadmap.md` to the code that now implements it,
and records what has actually been measured versus what is scaffolded.

Run order:

```bash
python diagnose_scaling.py                     # Step 1 — diagnose (seconds, no training)
python diagnose_quaternary.py                   # Lever 7 diagnose (minutes, no training)
python run_ablation.py --epochs 35              # the feature ablation (§5 Q2)
python eval_hierarchical.py                     # Lever 2 payoff test
python make_ablation_report.py                  # → results/ablation_scaling/REPORT.md
```

> **Feature-dimension change.** `run_ablation.py` now builds 662-d node / 11-d
> edge tensors (was 658/6) because the Lever 7 block is appended. Every condition
> inside one run still has identical capacity, which is what the ablation
> controls for — but numbers from a new run are **not** directly comparable to
> the 658/6 figures in the existing `results/ablation_scaling/REPORT.md`. Re-run
> the baselines alongside anything new.

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

**Status: done, payoff measured on the frozen large split — and it is negative.**

`results/ablation_scaling/hierarchical.json`, `v4_plus_chiral` weights, 68 large
complexes: flat τ_sym **0.460** vs hierarchical **0.428**, τ_ring 0.491 vs 0.461,
and 11.8 s vs 9.6 s. The n-reduction is real (31.3 → 10.4 effective across 4.2
modules) but reusing chain-level weights at the module level loses accuracy and
costs time. The diagnostic explains it: GT assembly stays inside one module only
59% of steps at n=21–40 and 50% at n=41+, so the decomposition contradicts the
labels about half the time.

Lever 2 therefore survives only as a *training-time* proposal — a model actually
trained at the module level, over modules that are assembly-aligned rather than
modularity-optimal. It is no longer the free win §0 of the roadmap took it for.
The machinery below stays in place for that experiment.
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

## Lever 7 — chirality and binding patterns in nonstandard quaternary structure

**Status: implemented and diagnosed; the training payoff is not yet measured.**

| Sub-item | Status | Code |
|---|---|---|
| Screw-operator handedness (θ, rise, sign) | **done + mirror-verified** | `src/features/quaternary.py::screw_parameters`, `pair_screw` |
| Isologous / heterologous interface typing | **done + censused** | `quaternary.py::isologous_score`, `classify_interface` |
| Ring adjacency + ring-frame node features | **done** | `quaternary.py::compute_quaternary_features` |
| Orbit-marginal (symmetry-quotiented) loss | **done, wired into the ablation** | `src/training/orbit_loss.py` |
| Contact-restricted target set | **done** | `orbit_loss.py::orbit_target_positions` |
| Frontier pooling / cross-attention | **done, low prior** | `scorer_gnn.py::AssemblyScorer(anchor_mode="frontier")` |
| Trained comparison vs `v4_baseline` | not run | — |

### Why the screw operator and not the old pseudoscalars

`chiral_descriptors.py` builds handedness from whole-chain PCA axes. That flips
sign under reflection, which is necessary but not sufficient — it is a
pseudoscalar of the *chains*, and for a homomer it mostly measures how the two
copies' inertia tensors happen to be oriented. The physically meaningful
handedness is a property of the operator mapping one copy onto the next.
Decomposing that rigid motion gives rotation θ about an axis plus a rise d along
it, and sign(d) is the handedness of the ring or helix.

Three properties make it the right feature, all verified in
`diagnose_quaternary.py` Q1 over 1,826 same-orbit pairs:

- **Exact under reflection.** Rise sign-flip residual 0.00e+00, while θ and the
  isologous overlap are exactly invariant. The old descriptor flipped everything
  including quantities that must not flip.
- **Swap-symmetric.** The inverse operator has axis −n̂ and translation −Rᵀt,
  whose dot product is the same d, so the feature belongs to the pair.
- **Zero for a flat ring.** A planar C_k ring has no handedness, and the
  descriptor reports none. 97/120 complexes still have a nonzero rise
  (mean |d| = 13.8 Å), so this is selectivity, not silence.

### Why the loss was the bigger problem

`diagnose_quaternary.py` Q3: above n=21, **73.6%** of supervised steps ask the
model to pick one of a mean 4.74 interchangeable copies. Exact-index
cross-entropy against an unidentifiable label has a floor of log k nats —
**1.146 at n=21–40, 1.487 at n=41+** — that a perfect model still pays and whose
gradient is noise. PRISM has *scored* modulo symmetry since Lever 6;
`train_gnn._get_orders` only quotients the objective for whole-complex homomers,
which is the rare case.

`orbit_marginal_ce` replaces the point label with the marginal likelihood over
the orbit. It is the exact MLE under a within-orbit-censored label, reduces
identically to smoothed cross-entropy for singleton orbits (unit-tested), and
costs one logsumexp. Contact restriction narrows the target set further but only
buys ~11% more (1.146 → 1.017 nats), so the marginal is doing the work.

### Recorded negative

The `frontier` anchor mode was built on the hypothesis that mean-pooling over the
assembled core dilutes the decisive interface. Q4 refutes it: mean and sum give
identical rankings (the divisor is constant across candidates at a fixed step)
and max is slightly worse in every size bin. It is retained only as a strict
capacity superset and ablated separately so it is judged rather than assumed.
The useful part of Q4 is the level: mean-pooled buried area alone reaches top-1
0.294 at n=21–40, above every trained condition in `REPORT.md`.

### Cost and compatibility

`precompute(..., include_quaternary=True)` appends node +4 / edge +5 after the
chiral block, so node_dim 655 → 658 → 662 and edge_dim 5 → 6 → 11. Both blocks
default to off, so v4/v5/v6 checkpoints load unchanged. Features are cached to
`data/quaternary_cache/{pdb}_quat.npz`; the O(N²) screw and interface work is
restricted to same-orbit pairs that actually touch, and capped at `max_pairs`.
`anchor_mode` defaults to `"pooled"`, so the frontier parameters are never
constructed unless asked for.

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
