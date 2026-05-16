"""
Statistical significance testing for assembly order prediction.

Tests:
  1. Wilcoxon signed-rank: GNN τ vs SASA-baseline τ on eval set
  2. Wilcoxon signed-rank: GNN τ vs random-ordering baseline
  3. Bootstrap 95% CI for mean τ

Run after training: python stat_significance.py --checkpoint results/gnn_training/best_gnn.pt
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import wilcoxon, bootstrap

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.scorer_gnn import AssemblyScorer
from src.evaluation.metrics import evaluate_dataset, kendall_tau, circular_tau
from train_gnn import (
    load_dataset_from_alpha, load_experimental_v2_dataset,
    precompute, populate_go_matrices, DEVICE, GO_CACHE, PLDDT_NODE_DIM
)


def sasa_greedy_order(cdata) -> list:
    """Greedy ordering by SASA — highest interfacial SASA first."""
    N = len(cdata.chain_ids)
    sasa = cdata.sasa_matrix
    remaining = list(range(N))
    assembled = []
    for _ in range(N):
        if not assembled:
            scores = [sasa[i, :].sum() for i in remaining]
        else:
            scores = [sasa[i, assembled].max() for i in remaining]
        best = remaining[int(np.argmax(scores))]
        assembled.append(best)
        remaining.remove(best)
    return assembled


def random_tau_expected(N: int, n_samples: int = 10000) -> float:
    """Expected τ under random ordering (should be ~0)."""
    gt = list(range(N))
    taus = [kendall_tau(list(np.random.permutation(N)), gt) for _ in range(n_samples)]
    return float(np.mean(taus))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/gnn_training/best_gnn.pt")
    parser.add_argument("--no_go", action="store_true")
    args = parser.parse_args()

    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"[sig] checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"[sig] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    node_dim = ckpt.get("node_dim", 646)
    print(f"[sig] epoch={ckpt.get('epoch')}, gt_prob={ckpt.get('gt_prob', 0):.4f}, "
          f"tau={ckpt.get('mean_tau', 0):.4f}")

    model = AssemblyScorer(
        node_dim=node_dim, edge_dim=4, hidden=128, n_layers=3, heads=4, dropout=0.1
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(DEVICE)

    marsh = load_dataset_from_alpha()
    v2    = load_experimental_v2_dataset()
    eval_set = marsh + v2

    if not args.no_go:
        print("[go] loading GO matrices...")
        populate_go_matrices(eval_set, GO_CACHE)

    esm_dim  = eval_set[0].esm_embeddings.shape[1]
    precomputed = {cd.pdb_id: precompute(cd, esm_dim) for cd in eval_set}

    # GNN τ per complex
    print("\n[sig] evaluating GNN...")
    eval_results = evaluate_dataset(model, eval_set, precomputed, device=DEVICE)
    eval_results.pop("_summary", None)

    gnn_taus  = []
    sasa_taus = []
    rand_taus = []

    print(f"\n{'PDB':>6} {'N':>3} {'GNN_τ':>7} {'SASA_τ':>7} {'RAND_τ':>7} {'type':>10}")
    print("-" * 50)

    for cdata in eval_set:
        pdb = cdata.pdb_id
        N   = len(cdata.chain_ids)

        gnn_tau  = eval_results.get(pdb, {}).get("tau", 0.0)
        sasa_ord = sasa_greedy_order(cdata)
        if cdata.complex_type == "homomer":
            sasa_tau = circular_tau(sasa_ord, cdata.gt_order)
        else:
            sasa_tau = kendall_tau(sasa_ord, cdata.gt_order)

        rand_tau = random_tau_expected(N, n_samples=5000)

        gnn_taus.append(gnn_tau)
        sasa_taus.append(sasa_tau)
        rand_taus.append(rand_tau)

        print(f"{pdb:>6} {N:>3} {gnn_tau:>7.3f} {sasa_tau:>7.3f} {rand_tau:>7.3f} {cdata.complex_type:>10}")

    gnn_arr  = np.array(gnn_taus)
    sasa_arr = np.array(sasa_taus)
    rand_arr = np.array(rand_taus)

    print(f"\n{'MEAN':>6} {'':>3} {gnn_arr.mean():>7.3f} {sasa_arr.mean():>7.3f} {rand_arr.mean():>7.3f}")

    # Wilcoxon signed-rank: GNN vs SASA
    print("\n=== Wilcoxon signed-rank: GNN vs SASA-greedy ===")
    diff_gs = gnn_arr - sasa_arr
    if np.any(diff_gs != 0):
        stat_gs, p_gs = wilcoxon(gnn_arr, sasa_arr, alternative="greater")
        print(f"  statistic={stat_gs:.3f}, p={p_gs:.4f} (one-sided, GNN > SASA)")
        stat_gs2, p_gs2 = wilcoxon(gnn_arr, sasa_arr, alternative="two-sided")
        print(f"  statistic={stat_gs2:.3f}, p={p_gs2:.4f} (two-sided)")
    else:
        print("  all differences are zero — cannot run Wilcoxon")

    # Wilcoxon: GNN vs random
    print("\n=== Wilcoxon signed-rank: GNN vs random ===")
    stat_gr, p_gr = wilcoxon(gnn_arr, rand_arr, alternative="greater")
    print(f"  statistic={stat_gr:.3f}, p={p_gr:.4f} (one-sided, GNN > random)")

    # Bootstrap 95% CI for mean GNN τ
    print("\n=== Bootstrap 95% CI for mean GNN τ ===")
    bs = bootstrap((gnn_arr,), np.mean, n_resamples=10000, confidence_level=0.95)
    ci = bs.confidence_interval
    print(f"  mean τ = {gnn_arr.mean():.4f}  95% CI: [{ci.low:.4f}, {ci.high:.4f}]")

    # Save results
    out_path = ROOT / "results" / "stat_significance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": str(ckpt_path),
        "epoch":      ckpt.get("epoch"),
        "n_complexes": len(eval_set),
        "mean_gnn_tau":  float(gnn_arr.mean()),
        "mean_sasa_tau": float(sasa_arr.mean()),
        "ci_95": [float(ci.low), float(ci.high)],
        "wilcoxon_gnn_vs_sasa": {"statistic": float(stat_gs), "p_value": float(p_gs)},
        "wilcoxon_gnn_vs_random": {"statistic": float(stat_gr), "p_value": float(p_gr)},
        "per_complex": [
            {"pdb_id": cd.pdb_id, "N": len(cd.chain_ids),
             "gnn_tau": float(gnn_taus[i]), "sasa_tau": float(sasa_taus[i])}
            for i, cd in enumerate(eval_set)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[sig] saved → {out_path}")


if __name__ == "__main__":
    main()
