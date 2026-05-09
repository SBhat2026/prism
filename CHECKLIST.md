# Protein Assembly Sim — Project Checklist
*Updated: 2026-04-27 v2 | Goal: GNN-guided assembly order prediction, Kendall's τ on Marsh 2013*

---

## Phase 0 — Environment & Setup

- ✅ GROMACS 2026.1 installed (`/opt/homebrew/bin/gmx`)
- ✅ PyTorch 2.10.0 + MPS available (M4 Mac Air)
- ✅ ESM 2.0.0 available (base conda env)
- ✅ Project scaffold created (`~/protein-assembly-sim/`)
- ✅ Toy 4-subunit demo runs (greedy τ=0.667, ablation table prints)
- ⬜ Create `prot-assembly` conda env: `conda env create -f configs/environment.yml`
- ⬜ Verify freesasa installs cleanly in new env (needed for SASA fallback)

---

## Phase 1 — Data Preparation

- ✅ **1a.** Marsh 2013 benchmark PDBs downloaded (10 complexes in `data/complexes/marsh2013/`)
- ✅ **1b.** NPC structure 3F3F downloaded (8-chain, `data/complexes/npc/`)
- ✅ **1c.** ESM-2 8M (320d) embeddings computed for all 71 Marsh2013 chain sequences
- ✅ **1c+.** ESM-2 150M (640d) embeddings computed — critical finding: τ=1.000 for homomers vs 0.143 for 8M
- ✅ **1d.** GROMACS SASA pipeline working; SASA cached in `data/sasa_cache/`
- ✅ **1e.** ProtFuncBridge loading `unified_v1.pth` (330MB) correctly with `weights_only=False`

---

## Phase 2 — Baseline Validation (Marsh 2013 Heuristic)

- ✅ **2a.** GROMACS SASA computed for all 10 complexes
- ✅ **2b.** Pairwise SASA matrices built (ring topology confirmed for 1AON, 1GRU)
- ✅ **2c.** Greedy assembly runs for all conditions
- ✅ **2d.** Kendall's τ computed: SASA-only mean τ=0.900 across 10 complexes
- ✅ **2e.** Baseline reproduced; note: 1GRU has SASA ambiguity (G closer to A than B → contradicts GT)

---

## Phase 3 — ProtFunc GO Coherence Term

- ✅ **3a.** ProtFuncBridge loads cleanly; switched to `get_logits()` (pre-sigmoid) for OOD robustness
- ✅ **3b.** GO coherence matrix computed (cosine on logits; range 0.976–1.000 for human proteins)
- ✅ **3c.** GO-only ablation: τ=0.917 heteromers vs 0.833 ESM-only — supports hypothesis
- ✅ **3d.** GO coherence edges ESM for heteromers; OOD limitation noted (ProtFunc trained on insects/mammals)

---

## Phase 4 — ESM Structural Compatibility Term

- ✅ **4a.** ESM cosine matrices loaded for both 8M and 150M models
- ✅ **4b.** Ablation run: ESM-only τ=0.833 homomers (8M), τ=1.000 (150M)
- ✅ **4c.** ESM 150M massively outperforms 8M on homomers; now used as GNN node features

---

## Phase 5 — STRING PPI Term

- ✅ **5a.** `src/data_prep/string_api.py` implemented (UniProt→STRING mapping + network query)
- ⬜ **5b.** Run STRING queries for Marsh 2013 proteins (mixed species — needs per-taxon queries)
- ⬜ **5c.** Run ablation: `greedy_assembly(w_ppi=1.0)`
- ⬜ **5d.** PPI expected to help most for larger complexes; currently zeros in GNN edges

---

## Phase 6 — Full Ablation Table (Table 1 of paper)

- ✅ **6a-b.** All 7 conditions × 10 complexes run; results in `results/alpha/alpha_results.json`
- ✅ **6c.** GO coherence Δτ: +0.084 for heteromers vs +0.000 for homomers — hypothesis supported
- ✅ **6d.** `results/alpha/FINAL_ALPHA_REPORT.png` generated (6-panel comprehensive figure)

---

## Phase 7 — GNN Scorer Training

- ✅ **7a.** Node features: ESM-150M (640d) + pLDDT (5d, AF-fetched per chain) + SASA diag (1d) = 646d
- ✅ **7b.** Edge features: [PPI, ESM_cos, GO_cos, SASA_norm] = 4d; adjacency = dense NxN
- ✅ **7c.** WeightLearner trained 400ep; GAT (3-layer, 167K params) v1: 500ep on MPS
- ⚠️ **7d.** v1 GNN τ=0.933 (plateau); gt_prob 0.62→0.74; homo_tau stuck at 0.667
         ROOT CAUSES: (a) wrong eval metric for symmetric ring homomers, (b) only 2 complexes with training signal, (c) 2/4 edge features zeros (PPI, GO)
- ✅ **7e.** v2 FIXES:
         - Bidirectional ring augmentation: 2N cyclic orders (CW+CCW) per homomer
         - Circular τ eval: max over all 2N ring orderings (correct scientific metric)
         - pLDDT node features replace 4d GO zeros (real AlphaFold confidence data)
         - Expanded training set: 100 PDB hetero-oligomers (4-14 chains, ≤2.0Å, RCSB query)
         - fast_sasa.py (freesasa) replaces GROMACS for new complexes
         - FINDING: old model with v1 eval achieved τ=1.0 on all complexes with circular τ — model was correct, metric was broken
- ✅ **7f.** v2 GNN training completed (killed); v3 launched with all fixes below
- 🔄 **7f+.** v3 GNN training running (PID 82981): 110 complexes, 646d node features
         FIXES vs v2: (a) BENCHMARK truncation fixed — 1BRS→6 chains, 1A2K→5, 1SBB→4
         (b) RingAwareLoss: min over 2N ring orderings (was mean), gradient to best path only
         (c) Multi-metric eval: τ + top1_accuracy + mean_confidence + early_step_tau
         (d) Checkpoint now saved on best gt_prob (not τ, which saturates at 1.0)
         (e) freesasa SASA fallback when GROMACS cache shape mismatch detected
- ✅ **7g.** Live dashboard: `python serve_dashboard.py` → http://localhost:8765/index.html

---

## Phase 8 — NPC Assembly Pathway Prediction

- ✅ **8a.** 3F3F (8-chain Y-complex) loaded; scoring matrices computed
- ✅ **8b.** 500-path ensemble run: 473/500 unique pathways (near-max diversity)
- ✅ **8c.** Step entropy computed; bottleneck at step 3 identified
- ⬜ **8d.** Comparison to EMBL 2022 mitosis/interphase modes not yet done
- ✅ **8e.** `results/alpha/npc_pathway_ensemble.png` + `npc_step_entropy.png` generated

---

## Phase 9 — GROMACS Interface Energy (Optional Enhancement)

- ⬜ **9a.** Run full GROMACS pipeline on 1AY7 (barnase-barstar, already prepped):
  ```bash
  python src/gromacs/run_md.py --pdb ~/1AY7.pdb
  ```
- ⬜ **9b.** Extract ΔG_interface from edr as additional scoring term
- ⬜ **9c.** Compare ΔG_interface vs ΔSASA correlation for known interfaces
- ⬜ **9d.** Only pursue if SASA alone leaves unexplained τ variance

---

## Integration Architecture Summary

```
Sequence → ESM-2 (via ProtFuncBridge) → GO probs + 320d embedding
PDB → GROMACS gmx sasa → ΔSASA per pair
UniProt IDs → STRING API → PPI scores
↓
Assembly simulation (greedy + stochastic)
↓
Kendall's τ vs Marsh 2013 ground truth
↓
Table 1: ablation | Figure 1: NPC pathway diversity
```

---

## Open Questions (from literature)
- Does ESM cosine proxy interface compatibility? → Test in Phase 4c
- Does GO coherence lift τ for heteromers? → Test in Phase 3d (central bet)
- Are assembly/disassembly symmetric? → Stochastic ensemble explores this
- NPC cell-cycle mode → scope to single pathway per run (no cell-cycle state)
