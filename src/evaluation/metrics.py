"""
Multi-metric evaluation for assembly order prediction.

evaluate_complex() returns a dict with four non-saturating metrics:
  - tau           : Kendall's τ (circular for homomers)
  - top1_accuracy : fraction of steps where highest-scored candidate = GT next
  - mean_confidence: mean softmax probability assigned to GT next subunit
  - early_step_tau : τ computed on only the first ceil(N/2) steps (most informative)
"""

import numpy as np
import torch
from scipy.stats import kendalltau
from typing import Optional


def kendall_tau(pred: list[int], gt: list[int]) -> float:
    if len(pred) != len(gt) or len(pred) < 2:
        return 0.0
    stat, _ = kendalltau(
        [pred.index(x) for x in gt],
        list(range(len(gt))),
    )
    return float(stat) if not np.isnan(stat) else 0.0


def circular_tau(pred: list[int], gt: list[int]) -> float:
    """Max τ over all 2N cyclic rotations (N forward + N backward)."""
    N = len(gt)
    base = list(gt)
    rev = list(reversed(gt))
    best = -1.0
    for i in range(N):
        best = max(best, kendall_tau(pred, base[i:] + base[:i]))
        best = max(best, kendall_tau(pred, rev[i:] + rev[:i]))
    return best


def _score_candidates(model, node_feats, adj, edge_feats, assembled_mask, candidates, device, h=None):
    """Score candidates. Pass precomputed h to skip redundant embed calls."""
    if h is None:
        h = model.embed(node_feats.to(device), adj.to(device), edge_feats.to(device))
    am = assembled_mask.to(device)
    ef = edge_feats.to(device)
    return model.score_candidates(h, ef, am, candidates)


def evaluate_complex(
    model,
    cdata,
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    adj: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """
    Full metric suite for one complex.

    Returns dict with keys:
      tau, top1_accuracy, mean_confidence, early_step_tau
    """
    from src.training.dataset import ComplexData

    model.eval()
    N = len(cdata.chain_ids)
    gt = list(cdata.gt_order)
    is_homomer = cdata.complex_type == "homomer"

    # Greedy prediction seeded at GT[0] (heteromers) or best-seed (homomers)
    if is_homomer:
        pred_order, step_probs, step_top1 = _greedy_best_seed(
            model, N, node_feats, edge_feats, adj, gt, device
        )
    else:
        pred_order, step_probs, step_top1 = _greedy_seed(
            model, N, gt[0], node_feats, edge_feats, adj, gt, device
        )

    # τ
    if is_homomer:
        tau = circular_tau(pred_order, gt)
    else:
        tau = kendall_tau(pred_order, gt)

    # top-1 accuracy
    top1_acc = float(np.mean(step_top1)) if step_top1 else 0.0

    # mean step confidence (softmax prob of GT next)
    mean_conf = float(np.mean(step_probs)) if step_probs else 0.0

    # early-step τ: first ceil(N/2) steps
    early_n = max(2, int(np.ceil(N / 2)))
    early_pred = pred_order[:early_n]
    early_gt   = gt[:early_n]
    # re-index to 0..early_n-1
    idx_map = {v: i for i, v in enumerate(early_gt)}
    early_pred_r = [idx_map[x] for x in early_pred if x in idx_map]
    early_gt_r   = list(range(len(early_gt)))
    if len(early_pred_r) == len(early_gt_r) and len(early_gt_r) >= 2:
        early_tau = kendall_tau(early_pred_r, early_gt_r)
    else:
        early_tau = tau

    return {
        "tau":            tau,
        "top1_accuracy":  top1_acc,
        "mean_confidence": mean_conf,
        "early_step_tau": early_tau,
    }


def _greedy_seed(model, N, seed, node_feats, edge_feats, adj, gt, device, h=None):
    """Greedy assembly from fixed seed. Pass h to avoid redundant embed."""
    order = [seed]
    remaining = [i for i in range(N) if i != seed]
    step_probs, step_top1 = [], []

    with torch.no_grad():
        if h is None:
            h = model.embed(node_feats.to(device), adj.to(device), edge_feats.to(device))
        for step in range(N - 1):
            if not remaining:
                break
            assembled_mask = torch.zeros(N, dtype=torch.bool)
            assembled_mask[order] = True
            scores = _score_candidates(model, node_feats, adj, edge_feats,
                                       assembled_mask, remaining, device, h=h)
            probs = torch.softmax(scores, dim=0)

            best_idx = int(scores.argmax().item())
            best_cand = remaining[best_idx]
            order.append(best_cand)
            remaining.remove(best_cand)

            gt_next_step = step + 1
            if gt_next_step < len(gt):
                gt_next = gt[gt_next_step]
                if gt_next in remaining or gt_next == best_cand:
                    orig_remaining_full = list(set(list(range(N))) - set(order[:-1]))
                    am2 = torch.zeros(N, dtype=torch.bool)
                    am2[order[:-1]] = True
                    s2 = _score_candidates(model, node_feats, adj, edge_feats,
                                           am2, orig_remaining_full, device, h=h)
                    p2 = torch.softmax(s2, dim=0)
                    if gt_next in orig_remaining_full:
                        gi = orig_remaining_full.index(gt_next)
                        step_probs.append(float(p2[gi].item()))
                        step_top1.append(int(orig_remaining_full[int(s2.argmax())] == gt_next))

    return order, step_probs, step_top1


def _greedy_best_seed(model, N, node_feats, edge_feats, adj, gt, device):
    """For homomers: try all N seeds, return best-τ result. Embeds once for all seeds."""
    best_tau = -2.0
    best_result = None
    with torch.no_grad():
        h = model.embed(node_feats.to(device), adj.to(device), edge_feats.to(device))
    for seed in range(N):
        order, probs, top1 = _greedy_seed(model, N, seed, node_feats, edge_feats, adj, gt, device, h=h)
        tau = circular_tau(order, gt)
        if tau > best_tau:
            best_tau = tau
            best_result = (order, probs, top1)
    return best_result


def evaluate_dataset(model, dataset, precomputed, device="cpu") -> dict:
    """
    Evaluate all complexes. Returns dict of pdb_id → metric dict,
    plus aggregate summary under key "_summary".
    """
    results = {}
    for cdata in dataset:
        nf, ef, adj = precomputed[cdata.pdb_id]
        results[cdata.pdb_id] = evaluate_complex(model, cdata, nf, ef, adj, device)

    taus   = [v["tau"]            for v in results.values()]
    top1s  = [v["top1_accuracy"]  for v in results.values()]
    confs  = [v["mean_confidence"] for v in results.values()]
    etaus  = [v["early_step_tau"] for v in results.values()]

    results["_summary"] = {
        "mean_tau":        float(np.mean(taus)),
        "mean_top1":       float(np.mean(top1s)),
        "mean_confidence": float(np.mean(confs)),
        "mean_early_tau":  float(np.mean(etaus)),
    }
    return results
