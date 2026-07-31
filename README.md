---
title: PRISM
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# PRISM — Protein Ranking by Interface and Sequence Modeling

**Predicting the *order* in which a protein complex assembles.**

A structure tells you what a complex looks like when it is finished. It does not
tell you how it got there — which subunit binds first, which pair forms an
obligate intermediate, which chain can only dock once a scaffold already exists.
PRISM takes a multi-subunit complex and predicts that pathway: at each step it
scores every unplaced subunit for how well it extends the partial assembly, and
builds the order greedily.

- **Live demo** — [prism-assembly.prismlab.workers.dev](https://prism-assembly.prismlab.workers.dev)
- **Backend Space** — [huggingface.co/spaces/Sbhat2026/PRISM](https://huggingface.co/spaces/Sbhat2026/PRISM)

---

## Contents

- [What PRISM does](#what-prism-does)
- [Results, honestly](#results-honestly)
- [Quickstart](#quickstart)
- [Method](#method)
- [The scaling program](#the-scaling-program)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Known limitations](#known-limitations)

---

## What PRISM does

Given a complex — chains, sequences, and (where available) coordinates — PRISM:

1. builds an **interface graph** over the subunits, with buried surface area,
   Voronoi contact density, STRING interaction confidence, ESM-2 embedding
   similarity, and GO-function coherence on the edges;
2. embeds it with a **3-layer GATv2** network (4 heads, 128 hidden);
3. at each assembly step, scores every remaining subunit against the
   representation of what has already been placed;
4. emits the greedy order, plus per-step probabilities and attention
   attributions you can inspect in the viewer.

It also supports point mutations (BLOSUM62-scored interface perturbation) and
side-by-side comparison of predicted vs annotated pathways.

## Results, honestly

The promoted checkpoint is **v4** (`results/v4_metrics.json`, promoted
2026-05-15, node_dim 655 / edge_dim 5).

| Set | n | τ | Top-1 |
|---|---|---|---|
| Exp GT v2 — **held out** | 14 | **0.976** | **0.964** |
| Marsh 2013 — in sample | 10 | 0.933 | 0.767 |

Three things that number does not mean, all of which matter more than it does:

- **It is 14 complexes, all small.** Everything in the held-out set is in the
  regime where the problem is comparatively easy.
- **It does not survive scale.** On a frozen split of 68 complexes with ≥21
  chains — never trained on — **no trained configuration beats an untrained
  buried-area heuristic** (BSA teacher top-1 0.266, v4 0.196). Top-1 decays
  0.788 → 0.114 from n=2–4 to n=41+. See
  [`results/ablation_scaling/REPORT.md`](results/ablation_scaling/REPORT.md).
- **The deployed Worker is not the GNN.** The live site's scorer is a
  SASA-greedy heuristic (τ ≈ 0.70 on its 10-complex list). The table above is
  model performance, not deployment performance; `roadmap.html` shows the
  deployed number.

The gap between those two facts — strong in-distribution, near-random above
n=21 — is the entire current research program.

## Quickstart

```bash
git clone https://github.com/SBhat2026/prism.git
cd prism
pip install -r requirements.txt
```

Run the viewer locally:

```bash
python assembly_viewer/server.py     # FastAPI + static frontend
```

Reproduce the scaling analysis (no training, minutes):

```bash
python diagnose_scaling.py        # symmetry census, chirality probe, BSA-teacher
                                  # quality, hierarchical reduction
python diagnose_quaternary.py     # screw handedness, interface typing, orbit
                                  # degeneracy, pooling test
```

Run the feature ablation (trains 12 conditions under one fixed split and seed):

```bash
python run_ablation.py --epochs 35
python make_ablation_report.py    # → results/ablation_scaling/REPORT.md
```

Train:

```bash
python train_gnn.py --use_labeled_15k --epochs 600 --patience 120 \
                    --anchor_mode pooled --no_go
```

## Method

### Node features (655 d, or 662 d with both optional blocks)

| Block | Dim | Source |
|---|---|---|
| ESM-2 embedding | 640 | ESM-2 150M, per chain |
| pLDDT statistics | 5 | AlphaFold confidence: mean, frac>90, frac<50, q10, q90 |
| Buried SASA (self) | 1 | FreeSASA, normalised |
| Monomer geometry | 9 | radius of gyration, asphericity, CA-trace descriptors |
| Chirality *(opt-in)* | 3 | pseudoscalars: tetrahedral volume, helical sense, frame handedness |
| Quaternary *(opt-in)* | 4 | ring membership, fold, ring radius, axial offset |

### Edge features (5 d, up to 11 d)

| Channel | Source |
|---|---|
| `sasa` | buried surface area of the pair |
| `contact` | Voronoi contact density |
| `ppi` | STRING interaction confidence |
| `esm_cos` | ESM-2 embedding cosine |
| `go_cos` | GO molecular-function coherence |
| `chiral` *(opt-in)* | interface handedness det[û_ij, v_i, v_j] |
| `isologous`, `screw_angle`, `screw_rise`, `screw_handed`, `ring_adj` *(opt-in)* | quaternary-structure descriptors — see below |

### Model

GATv2 with residual connections and layer norm; a copy-index embedding for
symmetry breaking; an optional sparse binding-probability prefilter that prunes
to top-k neighbours before message passing. The score head consumes the
candidate representation, a pooled representation of the assembled core, and the
candidate's edges to that core.

### Evaluation

Exact-order Kendall τ is the wrong metric above a handful of chains: swapping two
chains related by a symmetry operator is a zero-information "error". PRISM
therefore reports a ladder (`src/evaluation/symmetry.py`):

- `tau` — exact order, greedy from a BSA-chosen seed (no ground-truth leak)
- `tau_sym` — after optimal within-orbit relabelling (sorted-to-sorted matching,
  which is provably optimal for τ and O(n log n) rather than a search over |G|)
- `tau_ring` — additionally quotienting out cyclic rotation and reflection of
  each detected ring
- `top1` / `top1_sym` — teacher-forced next-subunit accuracy, seed-independent
- `n_gt_orbits` per complex, so complexes where τ_sym is vacuous can be excluded

Splits are hash-stable and size-stratified, with **every** n ≥ 21 complex frozen
out of training, so large-n numbers measure extrapolation rather than
interpolation.

## The scaling program

The target is complexes like the nuclear pore (~552 chains from ~30 nucleoporins
in 8-fold symmetry). 92.8% of protein-only PDB entries have ≤4 chains and only
1.3% have >40, so the data to train that directly does not exist. The plan for
getting there, and its measured status, lives in
[`PRISM_scaling_roadmap.md`](PRISM_scaling_roadmap.md) and
[`SCALING_IMPLEMENTATION.md`](SCALING_IMPLEMENTATION.md).

| Lever | Status |
|---|---|
| 1 — chirality descriptors, symmetry-detect-then-tile | done; interim descriptors measured as a wash |
| 1 — full E(3)-equivariant backbone | on hold — a cheaper explanation for the wash is now untested |
| 2 — hierarchical / modular assembly | done; **payoff test negative** with shared weights (τ_sym 0.428 vs 0.460 flat) |
| 3 — curriculum learning on size | not started; splits are in place |
| 4 — BSA teacher weak labels | done; teacher agrees with labels at τ≈0.00 step-wise, so it is a local preference, not a pathway |
| 5 — EMDB / 3D-Complex mining | not started |
| 6 — symmetry-aware evaluation | done; this is what everything is scored with |
| **7 — nonstandard quaternary structure** | **implemented, diagnosed; training payoff not yet measured** |

Things the program has ruled *out*, which is most of its value so far:

- **ESM scale-up buys nothing where this is headed.** At small n, ESM does
  almost all the work (`esm_only` 0.602 ≈ `v4_baseline` 0.610, and an
  ESM-shuffled control drops below no-ESM, so it is chain-specific embedding
  *identity*, not dimensionality). But that advantage is entirely
  in-distribution: ESM top-1 decays 0.769 → 0.187 across size bins.
- **Symmetry-aware scoring does not by itself rescue the model.** Ring-aware
  scoring lifts v4's τ from 0.080 to 0.444 — but lifts *random* by +0.32 too, so
  the honest margin over random is +0.07. The metric was necessary to measure
  anything; it was not a fix.
- **Mean-pooling over the assembled core is not the size bug.** Mean and sum
  induce identical rankings and max is slightly worse. Recorded because the same
  test showed buried area *alone* scoring top-1 0.294 at n=21–40 — above every
  trained condition.

### Lever 7 — rings, helices, and general homomers

Above n=21 symmetric homomeric sub-assemblies are not an edge case, they are the
corpus. Three parts (`src/features/quaternary.py`, `src/training/orbit_loss.py`):

**Handedness lives in the operator, not the chains.** Superposing one copy of a
subunit onto the next gives a rigid motion, which decomposes into a rotation θ
about an axis plus a rise *d* along it. `sign(d)` is the handedness of the ring
or helix. It flips exactly under reflection (measured residual 0.00e+00 over
1,826 pairs), is invariant to swapping the two copies, and is **exactly zero for
a flat ring** — which is correct, since a planar C_k ring has no handedness. The
earlier whole-chain pseudoscalars flipped sign under reflection too, but were
manufacturing handedness out of PCA noise on the achiral majority.

**Isologous vs heterologous interfaces (Monod; Levy & Teichmann).** The Jaccard
overlap of the two interface residue sets says whether a homomeric contact is
self-complementary — a 2-fold that terminates growth at a dimer — or head-to-tail,
which propagates into a ring or an unbounded helix. The census says the training
distribution teaches the wrong one:

| n | isologous | heterologous | helical |
|---|---|---|---|
| 5–10 | 0.334 | 0.291 | 0.327 |
| 21–40 | 0.052 | 0.509 | 0.390 |
| 41+ | 0.076 | 0.236 | 0.673 |

**The loss was fitting noise.** PRISM has *scored* modulo symmetry since Lever 6,
but the objective still put all its mass on one chain index per step. Above n=21,
73.6% of supervised steps ask the model to distinguish a mean 4.74
interchangeable copies — an irreducible cross-entropy floor of 1.15 nats
(1.49 at n=41+) that a perfect model still pays and whose gradient is noise.
`orbit_marginal_ce` replaces the point label with the marginal likelihood over
the orbit, `−log Σ_{j∈orbit} p_j`: the exact MLE under a within-orbit-censored
label, identical to plain cross-entropy when the orbit is a singleton, one
logsumexp of overhead.

## Repository layout

```
prism/
├── train_gnn.py                  main training loop, dataset loaders, precompute
├── run_ablation.py               12-condition feature/loss/architecture ablation
├── diagnose_scaling.py           symmetry, chirality, BSA-teacher, hierarchical
├── diagnose_quaternary.py        Lever 7 diagnostics
├── eval_hierarchical.py          flat vs hierarchical inference
├── make_ablation_report.py       → results/ablation_scaling/REPORT.md
├── app.py / Dockerfile           HF Space entry point
├── src/
│   ├── models/       scorer_gnn.py (GATv2 + score head), hierarchical.py, attribution.py
│   ├── features/     chiral_descriptors.py, quaternary.py, channel_health.py
│   ├── evaluation/   symmetry.py, metrics.py, splits.py, significance_test.py
│   ├── training/     orbit_loss.py, losses.py, gnn_trainer.py, dataset.py
│   ├── data_prep/    bsa_teacher.py, fast_sasa.py, voronoi_interface.py, string_api.py
│   ├── data_pipeline/ RCSB / CORUM / Complex Portal ingestion, GT labelling
│   └── preprocessing/ quotient_graph.py (orbit compression)
├── assembly_viewer/              local FastAPI server + 3Dmol.js viewer
├── data/                         complexes, caches, splits (mostly gitignored)
└── results/                      metrics, diagnostics, ablation reports
```

## Data

| Source | Count | Use |
|---|---|---|
| Labelled training corpus (`data/labeled_15k`) | 6,919 | training + ablation |
| Browsable corpus (RCSB 6,353 + CORUM 979 + Complex Portal 608 + Exp GT 23) | 7,963 | site browse tab |
| Marsh 2013 | 10 | in-sample benchmark |
| Experimental GT v2 | 14 | held-out benchmark |
| Path-LZerD test set | — | external comparison |

Ground-truth assembly order is far scarcer than structure. **97% of the labelled
corpus is tier-2** (heuristic or curated, not experimentally determined order)
and **64% carries only a prefix** of the order. Every metric in this repository
is agreement with those labels, not with in-vivo pathways.

## Known limitations

- Assembly-order labels are mostly weak (see above). Whether the target should be
  "the single true in-vivo pathway" or "a plausible, energetically monotonic
  pathway" is unresolved, and the second is far better posed.
- Orbit detection is UniProt-first. Every sequence field in the current corpus is
  empty, so chains without a UniProt accession (3–8% depending on size bin) fall
  back to singleton orbits — which makes the reported degeneracy a lower bound.
- The GO coherence channel is identically zero in the current corpus and cannot
  contribute; `src/features/channel_health.py` gates on this so a dead channel is
  never mistaken for a modelling result.
- The deployed Cloudflare Worker scores with a SASA-greedy heuristic, not the GNN.
- `src/models/scorer_gnn.py` has diverged from the v4/v5 checkpoints in the past;
  `precompute()`'s optional feature blocks default to off specifically so older
  checkpoints keep loading.

## License

See repository settings. Built on PDB, UniProt, STRING, CORUM, Complex Portal,
AlphaFold DB, and ESM-2.
