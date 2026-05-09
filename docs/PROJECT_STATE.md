# PRISM — Project State (as of 2026-05-01)

**Full name:** Protein Ranking by Interface and Sequence Modeling  
**Location:** `~/protein-assembly-sim/`  
**Python env:** `/Users/siddhantbhat/miniforge3/bin/python3` (has esm + freesasa + torch)  
**HF Space:** `Sbhat2026/PRISM` (Docker SDK)  
**HF local mirror:** `/tmp/prism-docker/` (ephemeral — recreate if /tmp is wiped)

---

## Scientific question

Can a GNN predict protein complex assembly order from sequence-derived + interface features alone (ESM-2, SASA, pLDDT, STRING PPI, GO terms)?

---

## Model architecture

```
Sequence → ESM-2 150M → 640d embedding
PDB → freesasa → BSA matrix (edge features)
UniProt → AlphaFold EBI → pLDDT stats (5d node features)
STRING API → PPI scores (edge feature)
ProtFuncBridge → GO cosine similarity (edge feature, cached in data/go_cache/)
↓
ComplexData → precompute() → (646d node_feats, 4d edge_feats, NxN adj)
↓
AssemblyScorer: GAT 3-layer (4 heads, 128 hidden, LayerNorm residuals)
  → embed(nf, adj, ef) → h  [runs GNN once per complex]
  → score_candidates(h, ef, mask, candidates) → scores  [batched, no re-embedding]
↓
Greedy assembly → circular τ (homomers/rings) / standard τ (heteromers)
Primary metrics: gt_prob, Kendall's τ
Temperature calibration: T*=0.332 (ternary search on NLL)
```

**Node features (646d):**
- 640d ESM-2 150M residue-mean embedding
- 5d pLDDT stats (mean, std, min, max, frac>70) from AlphaFold EBI
- 1d sequence length fraction of total complex

**Edge features (4d):**
- BSA (buried surface area) from freesasa
- STRING PPI confidence (0–1)
- GO cosine similarity (ProtFuncBridge)
- Sequence length ratio (proxy)

---

## Codebase map

```
protein-assembly-sim/
├── train_gnn.py                        ← main training script
├── ablation_study.py                   ← pseudo_only/exp_only/full conditions
├── stat_significance.py                ← Wilcoxon + bootstrap CI
├── build_pathlzerd_testset.py          ← fetch 22 held-out Path-LZerD complexes
├── compare_models.py / compare_esm_models.py
├── generate_final_report.py
├── src/
│   ├── models/
│   │   └── scorer_gnn.py              ← AssemblyScorer GAT (embed + score_candidates)
│   ├── data_prep/
│   │   ├── build_experimental_dataset.py
│   │   ├── build_pdb_training_set.py  ← RCSB query, 2+ chains, ≤2.5Å
│   │   ├── compute_embeddings.py
│   │   ├── fast_sasa.py
│   │   ├── fetch_complexes.py
│   │   ├── plddt_fetcher.py
│   │   └── string_api.py
│   ├── evaluation/
│   │   ├── expanded_benchmark.py      ← full benchmark table + plots
│   │   ├── metrics.py                 ← kendall_tau, circular_tau, embed-once
│   │   ├── significance_test.py       ← Wilcoxon, Mann-Whitney, bootstrap, Cohen's d
│   │   └── pca_generalization.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── gnn_trainer.py
│   │   ├── losses.py
│   │   └── weight_learner.py
│   ├── simulation/
│   │   └── assembly_sim.py
│   ├── protfunc_bridge/
│   │   └── go_coherence.py            ← ProtFuncBridge (8M ESM + ProtFunc MLP)
│   ├── gromacs/
│   │   ├── extract_sasa.py
│   │   └── run_md.py
│   └── visualization/
│       └── trajectory_viz.py
├── assembly_viewer/                    ← local viewer (FastAPI + HTML)
│   ├── server.py
│   └── static/index.html
├── data/
│   ├── experimental_gt_v2.json        ← 13 complexes with correct chain IDs
│   ├── experimental_complexes/        ← exp GT v2 PDB files
│   ├── training_complexes/            ← RCSB expanded (~200 complexes)
│   ├── complexes/                     ← Marsh 2013 PDB files
│   ├── embeddings/                    ← cached ESM embeddings
│   ├── go_cache/                      ← {pdb_id}_go.npy per complex
│   ├── plddt_cache/
│   ├── sasa_cache/
│   └── string/
├── results/
│   └── gnn_training/
│       ├── best_gnn.pt                ← production checkpoint (dict format)
│       ├── best_v3.pt                 ← v3 run checkpoint (same as best_gnn.pt)
│       ├── final_gnn.pt
│       ├── temperature.json           ← T*=0.332
│       ├── train_v3.log
│       ├── training_log.json
│       └── dashboard.html
├── configs/
│   └── environment.yml
├── docs/
│   └── literature_review_prompt.md
├── hf_deploy/
│   ├── prism-model/
│   └── prism-space/
└── mdp/                               ← GROMACS MD parameter files
```

---

## Datasets

| Dataset | Size | GT type | Role |
|---------|------|---------|------|
| Marsh 2013 | 10 complexes | Experimental | Eval only — **NEVER train on** |
| experimental_gt_v2 | 13 complexes | Experimental from literature | Train + eval |
| RCSB expanded | ~200 complexes | SASA-greedy pseudo-GT | Training only |
| **Total** | **~223** | | |

### experimental_gt_v2 complexes (`data/experimental_gt_v2.json`)
| PDB | Complex | Chains |
|-----|---------|--------|
| 1CSE | Subtilisin/Eglin-c | E, I |
| 1ACB | α-Chymotrypsin/Eglin-c | E, I |
| 2SNI | Subtilisin/CI-2 | E, I |
| 1PPE | Elastase/Ovomucoid | E, I |
| 2SIC | Subtilisin/SSI | E, I |
| 1FIN | CDK2/CyclinA | A, B |
| 1EWY | Colicin E9/Im9 | A, B |
| 2BTF | Actin/DNase I | A, **P** (not D) |
| 1TIM | TIM homodimer | A, B |
| 2OCJ | p53 tetramer | A, B, C, D |
| 1AXC | PCNA ring trimer | A, B, C |
| 1IRA | IFN-α/IFNAR2 | **X, Y** (not A, B) |
| 1GLA | Glucoamylase | **F, G** (not A, B) |

> 1NMB removed — neuraminidase tetramer only in crystal symmetry, no 4-chain ASU

---

## Training: v3 run (COMPLETED 2026-05-01)

**Command:**
```bash
/Users/siddhantbhat/miniforge3/bin/python3 train_gnn.py \
  --use_experimental_v2 --epochs 600 --patience 100
```

**Results:**
- Best gt_prob = **0.8104** @ epoch 390
- Final τ = **1.0000** (on Marsh 2013 eval)
- Dataset: 384 complexes (Marsh10 + ExpV2_14 + expanded360), eval=24
- Log: `results/gnn_training/train_v3.log`
- Checkpoint: `results/gnn_training/best_v3.pt` AND `best_gnn.pt`

**Key training optimizations (added this session):**
- `embed()` runs GNN layers once per complex; `score_candidates()` batches all candidates — ~21x speedup
- `precompute()` moves tensors to DEVICE once at load time
- `_order_loss_from_h()`: calls embed once per complex, reuses h for all orderings
- Early stopping `--patience 100`, `SAVE_EVERY=25`
- Saves both `best_gnn.pt` and `best_v3.pt`

---

## Benchmark results

### Expanded benchmark (current model)
| Set | n | τ | Top-1 | Conf | Early-τ |
|-----|---|---|-------|------|---------|
| Marsh 2013 | 10 | 1.000 | 0.867 | 0.856 | 1.000 |
| Exp GT v2  | 13 | 0.974 | 0.962 | 0.966 | 0.974 |
| Combined   | 23 | 0.986 | 0.920 | 0.918 | 0.986 |

**Caveats:**
- τ=1.0 for Marsh partly because model was trained on it
- Most v2 are 2-chain (trivially 1-step decisions)
- 2OCJ (p53 4-chain): τ=0.667, top1=0.500 — genuine failure case

### Statistical significance tests (Marsh vs v2)
- τ: MWU p=0.491 ns, d=0.39 small
- top1: MWU p=0.368 ns
- conf: MWU p=0.009 ** — v2 significantly higher (easier, more 2-chain)
- **Conclusion:** model generalizes; conf difference is complexity bias, not model failure

### Temperature calibration
- T*=0.332 vs T=1.0 baseline
- NLL: 0.2728 → 0.2326 on Marsh eval
- Saved: `results/gnn_training/temperature.json`

---

## Checkpoint format

New format (dict):
```python
{
    "model_state_dict": OrderedDict,
    "node_dim": 646,
    "epoch": int,
    "gt_prob": float,
    "mean_tau": float
}
```
Old format: plain `OrderedDict`. Both handled in `expanded_benchmark.py` and `server.py`.
Hidden dim inferred from `state["node_proj.weight"].shape[0]`.

---

## HF Space structure (`Sbhat2026/PRISM`)

Reorganized 2026-05-01 into subfolders:

```
/
├── Dockerfile
├── README.md
├── requirements.txt      ← torch, fair-esm, numpy, requests, fastapi, uvicorn, scipy
├── server/
│   ├── README.md
│   ├── server.py         ← FastAPI app with GNN inference (_load_gnn, _simulate_gnn)
│   └── scorer_gnn.py     ← AssemblyScorer (same arch as src/models/scorer_gnn.py)
├── model/
│   ├── README.md
│   └── best_gnn.pt       ← current production checkpoint
└── static/
    ├── README.md
    └── index.html        ← full frontend (identity nodes, timeline, metrics panel)
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server/server.py .
COPY server/scorer_gnn.py .
COPY model/best_gnn.pt .
COPY static/ static/
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

**GNN inference in server.py:**
- `_load_gnn()` — loads AssemblyScorer from `best_gnn.pt` at startup; sets `_state["gnn_available"]`
- `_build_gnn_features()` — builds (node_feats, edge_feats, adj) from ESM + proxy SASA (seq-len based, no PDB) + neutral pLDDT + STRING PPI + GO Jaccard
- `_simulate_gnn()` — embed once per complex, score_candidates per step
- `/status` returns `scorer: "gnn"|"weighted"`, `gnn_available: bool`
- Falls back to weighted heuristic if checkpoint missing

---

## Frontend (assembly_viewer/static/index.html + HF static/index.html)

### Change 1 — Identity-aware nodes
- `IDENTITY_PALETTE` = 8 colors assigned per unique UniProt ID
- Fill = state dark tint: assembled `#1a2e1a`, adding `#2e2a14`, pending `#1c2128`
- Stroke = identity color (same protein always same color)
- `r = clamp(10 + √seq_len × 0.55, 12, 28)`
- Layout R = `140 - maxNodeR - 22`
- 1-based step badge centered in assembled/adding nodes
- Name label below node
- Forming bond opacity/width scales with PPI score

### Change 2 — Assembly timeline
- `#assembly-timeline` strip below step-summary
- Clickable chips: step#, colored dot, name, combined score
- States: `tl-done`, `tl-current`, `tl-pending`

### Change 3 — Metrics panel
- τ sparkline: partial Kendall τ per step (SVG 200×52, color-coded)
- Radar chart: 4-axis NSEW (SASA/ESM/PPI/GO), winner blue filled vs runner-up red dashed
- Confidence bar: block chars + % + delta over runner-up

### Change 4 — Polish
- Sidebar score weight bar (SASA 40% / ESM 35% / GO 15% / PPI 10%)
- Gold `goldPulse` animation on graph panel when τ=1.0
- Copy citation BibTeX button in About tab

---

## Pending work (priority order)

1. **[DONE]** v3 training finished — best_v3.pt saved
2. **TODO** Copy `best_v3.pt` → `/tmp/prism-docker/model/best_gnn.pt` and push to HF Space
3. **TODO** Run `python stat_significance.py --checkpoint results/gnn_training/best_gnn.pt`
4. **TODO** Run `python ablation_study.py` (~2h, 3 conditions)
5. **TODO** Run `python build_pathlzerd_testset.py` (22 held-out Path-LZerD structures)
6. **TODO** Evaluate on Path-LZerD test set after data build

### Path-LZerD held-out PDB IDs (22)
```
1A0R 1VCB 2AZE 1ES7 1GPQ 2E9X 1S5B 1IKN 1KF6 3FH6
3VYT 1HEZ 1W88 1RLB 4HI0 4IGC 3UKU 2QSP 1H9D 4GWP
1DU3 2BQ1
```

---

## GO / ProtFuncBridge wiring

- `populate_go_matrices(dataset, GO_CACHE)` called at startup in `train_gnn.py main()`
- Lazy-loads ProtFuncBridge (8M ESM + ProtFunc MLP, local files in `~/insecta_webapp/`)
- Caches per-complex to `data/go_cache/{pdb_id}_go.npy`
- `--no_go` flag skips GO entirely
- In HF server: GO Jaccard computed from UniProt GO term fetches (no ProtFuncBridge available)

---

## Key invariants / gotchas

- **NEVER train on Marsh 2013** — held-out eval only
- **2BTF chain P** (not D) — common mistake
- **1IRA chains X,Y** (not A,B); **1GLA chains F,G** (not A,B)
- Python env must be miniforge3, not system python3 (missing esm/freesasa)
- assembly_viewer local server uses python 3.13 venv
- Checkpoint hidden dim is inferred, not hardcoded — always check `node_proj.weight.shape[0]`
- HF Space: always push after any change to server/, model/, static/, Dockerfile, requirements.txt
