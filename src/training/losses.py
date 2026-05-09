"""
Loss functions for assembly order ranking.

Primary: Softmax cross-entropy over candidates at each step.
  At step t, there are k remaining candidates; 1 is correct.
  Loss = -log(softmax(scores)[correct_idx])

Secondary: Kendall's τ (non-differentiable, used for evaluation only).
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau


def step_ce_loss(scores: torch.Tensor, correct_idx: int) -> torch.Tensor:
    """Softmax CE over candidate scores at one step. scores: (k,)"""
    return F.cross_entropy(scores.unsqueeze(0), torch.tensor([correct_idx]))


def batch_ranking_loss(
    score_fn,
    step_groups: list,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sum cross-entropy loss over all (step_group) in the batch.
    score_fn: callable(features: np.ndarray) -> torch.Tensor of shape (k,)
    """
    total_loss = torch.tensor(0.0, requires_grad=True)
    n_steps = 0
    for group in step_groups:
        feats = np.stack([ex.features for ex in group])  # (k, 8)
        feats_t = torch.tensor(feats, dtype=torch.float32)
        scores = score_fn(feats_t) / temperature          # (k,)
        correct_idx = next(i for i, ex in enumerate(group) if ex.is_correct)
        loss = step_ce_loss(scores, correct_idx)
        total_loss = total_loss + loss
        n_steps += 1
    return total_loss / max(n_steps, 1)


def kendall_tau_from_order(pred_order: list[int], gt_order: list[int]) -> float:
    if len(pred_order) <= 1:
        return 1.0
    gt_rank = {v: i for i, v in enumerate(gt_order)}
    gt_ranks = [gt_rank[v] for v in pred_order]
    tau, _ = kendalltau(list(range(len(pred_order))), gt_ranks)
    return float(tau)


def evaluate_all_tau(model, dataset, device="cpu") -> dict[str, float]:
    """
    Run greedy assembly on all complexes in dataset, return per-complex and mean τ.
    model: WeightLearner or GNNScorer
    """
    from src.simulation.assembly_sim import Complex, greedy_assembly, _ensure
    import sys

    results = {}
    for cdata in dataset:
        N     = len(cdata.chain_ids)
        sasa  = cdata.sasa_matrix / (cdata.sasa_matrix.max() + 1e-8)
        cplx  = Complex(
            name=cdata.pdb_id,
            sequences=cdata.sequences,
            chain_ids=cdata.chain_ids,
            sasa_matrix=sasa,
            ppi_matrix=cdata.ppi_matrix,
            esm_matrix=cdata.esm_matrix,
            go_matrix=cdata.go_matrix,
            ground_truth_order=cdata.gt_order,
        )
        # Use model weights for greedy assembly
        w = model.get_weights()
        r = greedy_assembly(cplx, w_sasa=w[0], w_ppi=w[3], w_esm=w[1], w_go=w[2])
        results[cdata.pdb_id] = r.kendall_tau or 0.0
    return results
