"""
bsa_teacher.py — Marsh/Levy buried-surface-area weak ("silver") assembly-order labels.

Lever 4 of PRISM_scaling_roadmap.md.

Assembly-*order* ground truth is far scarcer than structure: 23 of PRISM's 6,919
labelled complexes are tier-1 experimental, and 64% of the rest carry only a
prefix of the order. But the Marsh/Levy empirical rule — *the largest buried
interface forms first, and assembly proceeds by adding whichever subunit buries
the most surface against what is already there* — yields an order for **any**
structure with an interface matrix. That turns the whole PDB into training
signal for pre-training / distillation.

Provided here
-------------
bsa_greedy_order()      — the teacher ordering.
bsa_order_confidence()  — per-step margin between the chosen and runner-up
                          subunit; low margin ⇒ the teacher itself is unsure,
                          so the step should be down-weighted in distillation.
teacher_step_targets()  — soft per-step target distributions over candidates,
                          for KL distillation against the teacher.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def bsa_greedy_order(
    interface_matrix: np.ndarray,
    seed: Optional[int] = None,
    subcomplexes: Optional[list[list[int]]] = None,
) -> list[int]:
    """
    Marsh/Levy greedy: nucleate at the single largest interface, then repeatedly
    add the subunit burying the most surface against the current assembly.

    seed: force a starting subunit (else the larger partner of the top interface).
    subcomplexes: if given, obligate groups are completed before a new group is
                  opened — the "preformed subcomplex docks as a unit" rule.
    """
    w = np.array(interface_matrix, dtype=np.float64, copy=True)
    n = w.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]
    np.fill_diagonal(w, 0.0)

    if seed is None:
        i, j = np.unravel_index(int(np.argmax(w)), w.shape)
        # Start with the partner that has the larger total buried area.
        order = [int(i)] if w[i].sum() >= w[j].sum() else [int(j)]
    else:
        order = [int(seed)]

    group_of = {}
    if subcomplexes:
        for gi, g in enumerate(subcomplexes):
            for m in g:
                group_of[m] = gi

    remaining = [k for k in range(n) if k not in order]
    while remaining:
        gains = np.array([w[c, order].sum() for c in remaining])
        if subcomplexes:
            open_groups = {group_of[m] for m in order if m in group_of}
            bonus = np.array([1e6 if group_of.get(c, -1) in open_groups else 0.0
                              for c in remaining])
            gains = gains + bonus
        pick = remaining[int(np.argmax(gains))]
        order.append(pick)
        remaining.remove(pick)
    return order


def bsa_order_confidence(interface_matrix: np.ndarray, order: Sequence[int]) -> list[float]:
    """
    Per-step normalised margin (best − runner-up) / (best + 1e-9) in [0, 1].
    Steps with margin ≈ 0 are ties the heuristic cannot resolve — exactly the
    steps where the weak label carries no information.
    """
    w = np.array(interface_matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(w, 0.0)
    n = w.shape[0]
    out = []
    for step in range(1, len(order)):
        assembled = list(order[:step])
        remaining = [k for k in range(n) if k not in assembled]
        if len(remaining) < 2:
            out.append(1.0)
            continue
        gains = np.sort(np.array([w[c, assembled].sum() for c in remaining]))[::-1]
        out.append(float((gains[0] - gains[1]) / (gains[0] + 1e-9)))
    return out


def teacher_step_targets(
    interface_matrix: np.ndarray,
    order: Sequence[int],
    temperature: float = 0.15,
) -> list[tuple[list[int], np.ndarray]]:
    """
    Soft targets for distillation: at each step, softmax over candidates of the
    normalised buried area. Returns [(candidate_indices, probs), ...].

    A soft target beats the hard argmax because it preserves the teacher's
    genuine uncertainty on near-ties instead of inventing a confident label.
    """
    w = np.array(interface_matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(w, 0.0)
    scale = w.max() + 1e-9
    n = w.shape[0]

    out = []
    for step in range(1, len(order)):
        assembled = list(order[:step])
        remaining = [k for k in range(n) if k not in assembled]
        if not remaining:
            break
        gains = np.array([w[c, assembled].sum() for c in remaining]) / scale
        z = gains / max(temperature, 1e-6)
        z -= z.max()
        p = np.exp(z)
        out.append((remaining, (p / p.sum()).astype(np.float32)))
    return out


def agreement_with_gt(interface_matrix: np.ndarray, gt_order: Sequence[int]) -> dict:
    """
    How good is the teacher on complexes where real labels exist?
    Reported so the distillation weight can be set from evidence rather than faith.
    """
    from scipy.stats import kendalltau

    pred = bsa_greedy_order(interface_matrix)
    gt = list(gt_order)
    sub = [x for x in pred if x in set(gt)]
    if len(sub) != len(gt) or len(gt) < 2:
        return {"tau": float("nan"), "top1": float("nan"), "n": len(gt)}
    rank = {v: i for i, v in enumerate(sub)}
    tau, _ = kendalltau([rank[v] for v in gt], list(range(len(gt))))
    top1 = float(np.mean([sub[i] == gt[i] for i in range(len(gt))]))
    return {"tau": 0.0 if np.isnan(tau) else float(tau), "top1": top1, "n": len(gt)}
