"""Point mutation utilities for PRISM assembly scorer."""

import numpy as np

_AA1 = frozenset("ACDEFGHIKLMNPQRSTVWY")


def apply_mutation(sequence: str, position: int, mutant_aa: str) -> str:
    """Replace residue at 1-indexed position with mutant_aa."""
    if not (1 <= position <= len(sequence)):
        raise ValueError(f"Position {position} out of range [1, {len(sequence)}]")
    if mutant_aa not in _AA1:
        raise ValueError(f"Invalid amino acid: {mutant_aa!r}")
    seq = list(sequence)
    seq[position - 1] = mutant_aa
    return "".join(seq)


def compute_kl_per_step(
    wt_probs: list[dict],
    mut_probs: list[dict],
    eps: float = 1e-9,
) -> list[float]:
    """KL(WT || MUT) per assembly step. Inputs are {candidate_idx: prob} dicts."""
    kls = []
    for wt_p, mut_p in zip(wt_probs, mut_probs):
        all_idx = sorted(set(wt_p) | set(mut_p))
        p = np.array([wt_p.get(i, eps) for i in all_idx], dtype=np.float64)
        q = np.array([mut_p.get(i, eps) for i in all_idx], dtype=np.float64)
        p = np.clip(p, eps, None)
        p /= p.sum()
        q = np.clip(q, eps, None)
        q /= q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        kls.append(round(kl, 4))
    return kls


def detect_pathway_redirect(wt_pred: list[int], mut_pred: list[int]) -> dict:
    """Detect first step where predicted order diverges between WT and MUT."""
    n = min(len(wt_pred), len(mut_pred))
    for i in range(n):
        if wt_pred[i] != mut_pred[i]:
            return {
                "redirected": True,
                "first_divergence_step": i,
                "n_steps_affected": n - i,
            }
    return {"redirected": False, "first_divergence_step": None, "n_steps_affected": 0}
