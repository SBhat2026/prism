# Claude catch-up prompt

Copy-paste this at the start of a new Claude session to get it fully up to speed on PRISM.

---

## Paste this:

```
I'm continuing work on PRISM (Protein Ranking by Interface and Sequence Modeling), a research project at ~/protein-assembly-sim/. Here's the full context:

## What PRISM does
Predicts protein complex assembly order using a GNN. Given an N-subunit complex, at each step it scores all candidate subunits and greedily picks the next one to join. Evaluated with Kendall's τ (circular variant for ring complexes).

## Model architecture
- Input: 646d node features (ESM-2 150M 640d + 5d pLDDT + 1d seq-len fraction)
- Edge: 4d (BSA/freesasa, STRING PPI, GO cosine, seq-len ratio)
- GNN: 3-layer GAT, 4 heads, 128 hidden, LayerNorm residuals
- Key optimization: embed() runs GNN once per complex; score_candidates() batches all candidates in one pass (~21x speedup over naive per-candidate forward)
- Temperature calibration: T*=0.332 (saves to results/gnn_training/temperature.json)
- Checkpoint format: dict with keys model_state_dict, node_dim, epoch, gt_prob, mean_tau

## Key files
- train_gnn.py — training (--use_experimental_v2, --epochs N, --patience N, --no_go)
- src/models/scorer_gnn.py — AssemblyScorer with embed() and score_candidates()
- src/evaluation/expanded_benchmark.py — benchmark table + plots
- src/evaluation/metrics.py — kendall_tau, circular_tau, greedy (embed-once)
- src/evaluation/significance_test.py — Wilcoxon, Mann-Whitney, bootstrap CI
- src/protfunc_bridge/go_coherence.py — ProtFuncBridge GO scoring
- ablation_study.py — pseudo_only / exp_only / full training conditions
- stat_significance.py — Wilcoxon GNN vs SASA-greedy, GNN vs random, bootstrap CI
- build_pathlzerd_testset.py — fetches 22 held-out Path-LZerD PDB structures

## Datasets
- Marsh 2013 (10 complexes): experimental GT — EVAL ONLY, NEVER TRAIN ON
- experimental_gt_v2 (13 complexes): experimental GT, train+eval (data/experimental_gt_v2.json)
- RCSB expanded (~200 complexes): SASA-greedy pseudo-GT, training only
- Total: ~223 complexes

## experimental_gt_v2 chain IDs (common mistakes here)
- 1CSE,1ACB,2SNI,1PPE,2SIC: chains E,I (protease-inhibitor)
- 1FIN,1EWY: chains A,B
- 2BTF: chains A,P (NOT D — Actin/DNase I)
- 1TIM: chains A,B (homodimer)
- 2OCJ: chains A,B,C,D (p53 tetramer)
- 1AXC: chains A,B,C (PCNA ring)
- 1IRA: chains X,Y (NOT A,B — IFN-α/IFNAR2)
- 1GLA: chains F,G (NOT A,B — Glucoamylase)

## Current results (v3 training, COMPLETED)
- Best gt_prob = 0.8104 @ epoch 390 (out of 600, early stopped at ~500)
- Marsh 2013: τ=1.000, top1=0.867
- Exp GT v2: τ=0.974, top1=0.962
- Combined: τ=0.986, top1=0.920
- Checkpoint: results/gnn_training/best_gnn.pt AND best_v3.pt

## HF Space: Sbhat2026/PRISM (Docker)
Local mirror: /tmp/prism-docker/ (may be wiped, recreate if needed)
Structure after reorganization:
  server/server.py + scorer_gnn.py + README.md
  model/best_gnn.pt + README.md
  static/index.html + README.md
  Dockerfile (COPYs from server/, model/, static/)
  requirements.txt

IMPORTANT: Always push to HF after any changes to HF Space files. Use huggingface_hub:
  from huggingface_hub import HfApi
  HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id='Sbhat2026/PRISM', repo_type='space')

## Frontend (static/index.html) — recently rewritten
4 changes made this session:
1. Identity-aware nodes: unique color per UniProt, size ∝ seq_len (r=clamp(10+√L×0.55,12,28)), step badge, name label, PPI-scaled bond opacity
2. Assembly timeline: clickable chips below graph (step#, dot, name, score)
3. Metrics panel: τ sparkline (partial Kendall τ per step), radar chart (4-axis SASA/ESM/PPI/GO, winner vs runner-up), confidence bar
4. Polish: sidebar score weight bar (40/35/15/10), gold pulse when τ=1.0, copy citation button

## Python environment
MUST use: /Users/siddhantbhat/miniforge3/bin/python3
NOT: system python3 (missing esm, freesasa)
assembly_viewer local server: python 3.13 venv

## Pending work (in priority order)
1. Copy best_v3.pt → /tmp/prism-docker/model/best_gnn.pt and push to HF
2. Run: python stat_significance.py --checkpoint results/gnn_training/best_gnn.pt
3. Run: python ablation_study.py (~2h)
4. Run: python build_pathlzerd_testset.py (22 Path-LZerD held-out structures)
   IDs: 1A0R 1VCB 2AZE 1ES7 1GPQ 2E9X 1S5B 1IKN 1KF6 3FH6 3VYT 1HEZ 1W88 1RLB 4HI0 4IGC 3UKU 2QSP 1H9D 4GWP 1DU3 2BQ1
5. Run evaluation on Path-LZerD test set

## Response style preferences
- Bullet points / checklists only, no prose paragraphs
- Status symbols: ✅ done / ⬜ todo / ⚠️ broken
- No filler words, no "Great!", "Sure!", etc.
- Terse, action-first
- When giving code: complete solutions, no placeholders
```
