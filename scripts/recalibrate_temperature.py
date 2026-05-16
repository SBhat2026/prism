#!/usr/bin/env python3
"""
Temperature recalibration for PRISM assembly scorer.

Finds T* minimising NLL on the validation set via ternary search over [0.01, 2.0].
Loads best_v2.pt checkpoint and val split from data/labeled_15k/.

Usage: python scripts/recalibrate_temperature.py [--checkpoint results/v2/best_v2.pt]
Output: results/v2/temperature.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def compute_nll(model, bundles, T: float, device: str) -> float:
    """Compute mean negative log-likelihood of GT candidate at each step."""
    model.eval()
    nlls = []
    with torch.no_grad():
        for b in bundles:
            node_feats = b.node_feats.to(device)
            adj = b.adj.to(device)
            edge_feats = b.edge_feats.to(device)
            copy_idx = getattr(b, 'copy_indices', None)
            if copy_idx is not None:
                copy_idx = copy_idx.to(device)
            h = model.embed(node_feats, adj, edge_feats, copy_idx)

            remaining = list(range(b.N))
            assembled_mask = torch.zeros(b.N, dtype=torch.bool, device=device)

            for step, gt_idx in enumerate(b.gt_order):
                if gt_idx not in remaining:
                    continue
                scores = model.score_candidates(h, edge_feats, assembled_mask, remaining)
                # Temperature-scaled softmax NLL
                probs = F.softmax(scores / T, dim=0)
                gt_pos = remaining.index(gt_idx)
                nll = -torch.log(probs[gt_pos] + 1e-9).item()
                nlls.append(nll)

                assembled_mask[gt_idx] = True
                remaining.remove(gt_idx)

    return float(np.mean(nlls)) if nlls else float("inf")


def ternary_search(model, bundles, lo: float, hi: float, n_iters: int,
                   device: str) -> tuple[float, float]:
    """Ternary search for T* minimising NLL."""
    for _ in range(n_iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if compute_nll(model, bundles, m1, device) < compute_nll(model, bundles, m2, device):
            hi = m2
        else:
            lo = m1
    T_star = (lo + hi) / 2
    nll = compute_nll(model, bundles, T_star, device)
    return T_star, nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/v2/best_v2.pt")
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--lo", type=float, default=0.01)
    parser.add_argument("--hi", type=float, default=2.0)
    args = parser.parse_args()

    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    from src.models.scorer_gnn import AssemblyScorer
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = AssemblyScorer(
        node_dim=cfg.get("node_dim", 652),
        edge_dim=cfg.get("edge_dim", 5),
        hidden=cfg.get("hidden", 128),
        n_layers=cfg.get("n_layers", 3),
        heads=cfg.get("heads", 4),
        dropout=0.0,
        anchor_mode=cfg.get("anchor_mode", "pooled"),
        sparse_k=cfg.get("sparse_k", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Load validation split
    from train_v2 import load_split
    splits_path = ROOT / "data" / "labeled_15k" / "splits.json"
    if not splits_path.exists():
        print("splits.json not found — run train_v2.py first to generate splits")
        sys.exit(1)
    splits = json.loads(splits_path.read_text())
    val_keys = splits.get("val", [])
    print(f"Loading {len(val_keys)} validation complexes...")
    bundles = [b for k in val_keys if (b := load_split(k)) is not None]
    print(f"Loaded {len(bundles)} valid bundles")

    if not bundles:
        print("No validation data. Cannot calibrate.")
        sys.exit(1)

    nll_uncal = compute_nll(model, bundles, T=1.0, device=device)
    print(f"NLL at T=1.0 (uncalibrated): {nll_uncal:.4f}")

    print(f"Searching T* over [{args.lo}, {args.hi}] for {args.iters} iterations...")
    T_star, nll_cal = ternary_search(model, bundles, args.lo, args.hi, args.iters, device)
    print(f"T* = {T_star:.6f}  →  NLL {nll_uncal:.4f} → {nll_cal:.4f} "
          f"(Δ = {nll_uncal - nll_cal:.4f})")

    out_path = ROOT / "results" / "v2" / "temperature.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "T_star": T_star,
        "nll_uncalibrated": nll_uncal,
        "nll_calibrated": nll_cal,
        "n_val_complexes": len(bundles),
        "checkpoint": str(args.checkpoint),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
