# Protein Assembly Viewer

Interactive browser for GNN-guided protein complex assembly order simulations.

## Install

```bash
pip install fastapi uvicorn requests torch fair-esm numpy
```

## Run

```bash
cd protein-assembly-sim
python assembly_viewer/server.py
# Browser opens automatically at http://localhost:8765
```

## What it shows

- 10 pre-defined protein complexes (heteromers, homomers, rings)
- Greedy assembly simulation: SASA (40%) + ESM-2 (35%) + GO coherence (15%) + PPI (10%)
- Step-by-step SVG graph: assembled (green) / adding (yellow) / pending (grey)
- Per-step score breakdown bars with weighted contributions
- GO term pills colored by category (F=molecular function, P=process, C=component)
- Rejected candidates table with per-term mini-bars
- Kendall τ and circular τ (for ring complexes) displayed live
- Temperature slider → rescales softmax for stochastic exploration, Rerun button reruns sim
- Keyboard: ← → to step, Space to play/pause
