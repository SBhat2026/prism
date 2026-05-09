"""
Ablation study: compare three training conditions.

  A) pseudo_only  — expanded SASA-greedy pseudo-GT only (360 complexes, no expGT)
  B) exp_only     — experimental GT only (10 Marsh + 14 v2 = 24 complexes)
  C) full         — all 384 (current default)

Each condition trains for --epochs (default 200) with early stopping (patience 80).
Reports mean Kendall τ and gt_prob on the 24-complex eval set.
Saves results to results/ablation/ablation_results.json.
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.scorer_gnn import AssemblyScorer
from src.training.dataset import ComplexData
from src.training.losses import kendall_tau_from_order
from src.evaluation.metrics import evaluate_dataset
from train_gnn import (
    load_dataset_from_alpha, load_expanded_dataset, load_experimental_v2_dataset,
    precompute, populate_go_matrices, eval_gt_prob, gnn_ranking_loss,
    DEVICE, GNN_DIR, GO_CACHE, PLDDT_NODE_DIM
)

ABLATION_DIR = ROOT / "results" / "ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def train_condition(
    name: str,
    train_dataset: list,
    eval_dataset: list,
    epochs: int = 200,
    patience: int = 80,
) -> dict:
    esm_dim  = train_dataset[0].esm_embeddings.shape[1] if train_dataset else 640
    node_dim = esm_dim + PLDDT_NODE_DIM + 1

    print(f"\n[ablation] === {name} ===")
    print(f"  train={len(train_dataset)}, eval={len(eval_dataset)}, epochs={epochs}")

    precomputed = {cd.pdb_id: precompute(cd, esm_dim) for cd in train_dataset}
    # eval complexes not in train precomputed
    for cd in eval_dataset:
        if cd.pdb_id not in precomputed:
            precomputed[cd.pdb_id] = precompute(cd, esm_dim)

    model = AssemblyScorer(
        node_dim=node_dim, edge_dim=4, hidden=128, n_layers=3, heads=4, dropout=0.1
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_gt_prob = -1.0
    best_tau     = 0.0
    patience_ctr = 0
    history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = gnn_ranking_loss(model, train_dataset, precomputed)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            eval_results = evaluate_dataset(model, eval_dataset, precomputed, device=DEVICE)
            summary = eval_results.pop("_summary")
            gtp = eval_gt_prob(model, eval_dataset, precomputed)
            tau = summary["mean_tau"]

            if gtp > best_gt_prob:
                best_gt_prob = gtp
                best_tau     = tau
                patience_ctr = 0
            else:
                patience_ctr += 5

            history.append({"epoch": epoch, "loss": float(loss.item()), "tau": tau, "gt_prob": gtp})
            print(f"  ep {epoch:4d} | loss={loss.item():.4f} | τ={tau:.4f} | gtp={gtp:.4f} | best_gtp={best_gt_prob:.4f}")

            if patience_ctr >= patience:
                print(f"  [ablation] early stop at ep {epoch}")
                break

    result = {
        "condition":   name,
        "n_train":     len(train_dataset),
        "n_eval":      len(eval_dataset),
        "best_gt_prob": best_gt_prob,
        "best_tau":    best_tau,
        "history":     history,
    }
    print(f"  RESULT: τ={best_tau:.4f}, gt_prob={best_gt_prob:.4f}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--no_go", action="store_true")
    args = parser.parse_args()

    marsh    = load_dataset_from_alpha()
    expanded = load_expanded_dataset()
    v2       = load_experimental_v2_dataset()

    all_train = marsh + expanded + v2
    eval_set  = marsh + v2

    if not args.no_go:
        print("[go] populating GO matrices...")
        populate_go_matrices(all_train, GO_CACHE)
        print("[go] done")

    conditions = [
        ("pseudo_only",  expanded,          eval_set),
        ("exp_only",     marsh + v2,        eval_set),
        ("full",         marsh + expanded + v2, eval_set),
    ]

    results = []
    for name, train, ev in conditions:
        r = train_condition(name, train, ev, args.epochs, args.patience)
        results.append(r)

    out = ABLATION_DIR / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ablation] saved → {out}")

    # Summary table
    print("\n=== ABLATION SUMMARY ===")
    print(f"{'Condition':<15} {'N_train':>8} {'τ':>8} {'gt_prob':>8}")
    for r in results:
        print(f"{r['condition']:<15} {r['n_train']:>8} {r['best_tau']:>8.4f} {r['best_gt_prob']:>8.4f}")


if __name__ == "__main__":
    main()
