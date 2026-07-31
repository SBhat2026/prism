"""
orbit_loss.py — Symmetry-quotiented training objective for assembly order.

Lever 7 of PRISM_scaling_roadmap.md, the training-side counterpart of the
symmetry-aware *metrics* in `src/evaluation/symmetry.py`.

The problem this fixes
----------------------
PRISM already scores with `tau_mod_symmetry` / `top1_accuracy_mod_symmetry`,
which quotient out the exchange of interchangeable chains — because "chain C
before chain E" is not a fact about biology when C and E are the same protein
in a symmetric ring, it is a fact about how the depositor lettered the PDB.

But the *loss* never got the same treatment. `_order_loss_from_h` in
`train_gnn.py` puts all its probability mass on one specific chain index at each
step, so every homomeric step trains the network to distinguish copies that are
by construction indistinguishable. Cross-entropy against an unidentifiable label
is not a hard problem, it is an *impossible* one: the Bayes-optimal predictor
puts 1/k on each of the k copies and still eats −log k of loss forever. The
gradient that remains is pure noise, and it grows with symmetry — which is to
say it grows with n, exactly where the ablation found the model collapsing to
random.

`train_gnn._get_orders` does address this, but only for whole-complex homomers
(`complex_type == "homomer"`), by enumerating all 2N cyclic rotations and taking
a min over their losses. That is (a) restricted to the rare pure-homomer case,
(b) O(N) forward-equivalents per complex, and (c) silent on the far more common
situation — a heteromeric complex containing several symmetric sub-orbits, which
is what every large assembly actually is (GroEL/GroES, the proteasome, the NPC).

The fix
-------
Marginal likelihood over the orbit. If the label only identifies *which protein*
comes next, not *which copy*, then the correct likelihood of the observation is
the total probability the model assigns to any copy of that protein:

    L = −log Σ_{j ∈ orbit(correct) ∩ remaining} softmax(scores)_j
      = −( logsumexp(scores[S]) − logsumexp(scores) )

This is the exact MLE under a label that is censored within the orbit. It is
strictly better behaved than the uniform-soft-label alternative: soft labels
force the model to *spread* mass evenly over the orbit, marginal likelihood lets
it commit to whichever copy the geometry favours and only asks that the mass
land somewhere in the orbit. Cost is one logsumexp — no enumeration, no min over
orderings, and it applies to homomers and heteromers alike.

Contact restriction
-------------------
The orbit is an upper bound on the ambiguity, not the truth. Once part of a ring
is placed, its copies are no longer interchangeable: the ones touching the
placed arc can extend it and the ones across the ring cannot. So when a contact
matrix is available we narrow the target set to orbit members that actually
touch the assembled core:

    S' = { j ∈ S : contact[j, assembled].max() > 0 }     (fall back to S if empty)

That is "predict on the asymmetric unit and tile by symmetry" implemented in the
*loss* rather than only at inference — the model is asked the well-posed
question (which entity extends the frontier) and is never graded on the
ill-posed one (which of the 8 identical copies the depositor called chain E).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Target-set construction
# ──────────────────────────────────────────────────────────────────────────────

def orbit_target_positions(
    remaining: Sequence[int],
    correct: int,
    orbit_of: dict[int, int],
    assembled: Optional[Sequence[int]] = None,
    contact: Optional[np.ndarray] = None,
    restrict_by_contact: bool = True,
) -> tuple[list[int], str]:
    """
    Positions *within `remaining`* that count as correct at this step.

    Returns (positions, mode) where mode is one of:
      "exact"    — the orbit is a singleton; identical to plain cross-entropy
      "orbit"    — several interchangeable copies, no contact info to narrow it
      "frontier" — narrowed to the orbit members touching the assembled core
    """
    idx = {v: p for p, v in enumerate(remaining)}
    if correct not in idx:
        return [], "exact"

    o = orbit_of.get(correct, None)
    if o is None:
        return [idx[correct]], "exact"

    members = [v for v in remaining if orbit_of.get(v, -1 - v) == o]
    if len(members) <= 1:
        return [idx[correct]], "exact"

    if restrict_by_contact and contact is not None and assembled is not None and len(assembled):
        a = list(assembled)
        touching = [v for v in members if float(np.max(contact[v, a])) > 0.0]
        # Only accept the narrowing if it keeps the annotated chain — otherwise
        # the contact matrix disagrees with the label and the label wins.
        if touching and correct in touching:
            return [idx[v] for v in touching], "frontier"

    return [idx[v] for v in members], "orbit"


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def orbit_marginal_ce(
    scores: torch.Tensor,          # (C,) raw logits over `remaining`
    target_positions: Sequence[int],
    label_smooth: float = 0.1,
) -> torch.Tensor:
    """
    −log P(next ∈ target set), with the same ε-smoothing the exact loss uses so
    the two are directly comparable across ablation conditions.

    With a singleton target set this reduces *exactly* to smoothed cross-entropy,
    which is what makes it a drop-in: nothing changes for asymmetric complexes.
    """
    if scores.numel() == 0 or not len(target_positions):
        return scores.sum() * 0.0

    log_probs = F.log_softmax(scores, dim=0)
    tgt = torch.as_tensor(list(target_positions), device=scores.device, dtype=torch.long)
    log_marginal = torch.logsumexp(log_probs.index_select(0, tgt), dim=0)

    if label_smooth <= 0.0:
        return -log_marginal
    # Uniform background over every candidate, mirroring the exact-loss smoothing.
    return -((1.0 - label_smooth) * log_marginal + label_smooth * log_probs.mean())


def order_loss_orbit(
    score_fn,
    n: int,
    order: Sequence[int],
    orbit_of: dict[int, int],
    contact: Optional[np.ndarray] = None,
    label_smooth: float = 0.1,
    restrict_by_contact: bool = True,
    device=None,
) -> torch.Tensor:
    """
    Whole-order loss, teacher-forced along `order`.

    `score_fn(assembled_mask, remaining) -> (C,) logits` keeps this module free
    of any dependency on the scorer's signature, so it can wrap either
    `train_gnn` or `run_ablation`'s call convention.
    """
    losses = []
    for step in range(1, len(order)):
        assembled = list(order[:step])
        correct = order[step]
        remaining = [i for i in range(n) if i not in set(assembled)]
        if correct not in remaining or len(remaining) < 2:
            continue

        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[assembled] = True
        scores = score_fn(mask, remaining)

        tgt, _mode = orbit_target_positions(
            remaining, correct, orbit_of, assembled, contact, restrict_by_contact
        )
        losses.append(orbit_marginal_ce(scores, tgt, label_smooth))

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics — how much label noise is the exact loss actually fitting?
# ──────────────────────────────────────────────────────────────────────────────

def degeneracy_stats(
    n: int,
    order: Sequence[int],
    orbit_of: dict[int, int],
    contact: Optional[np.ndarray] = None,
) -> dict:
    """
    Per-complex census of unidentifiable steps, computed without training
    anything. Answers: what fraction of supervised steps ask the model to pick
    one of k interchangeable copies, and what is the resulting irreducible
    cross-entropy floor?

    `ce_floor` is Σ log k / steps — the loss a *perfect* model still pays under
    the exact objective. It is the amount of the training signal that is pure
    noise, and it is what the marginal objective removes.
    """
    steps = ambiguous = frontier_resolved = 0
    ks: list[int] = []
    ks_frontier: list[int] = []

    for step in range(1, len(order)):
        assembled = list(order[:step])
        correct = order[step]
        remaining = [i for i in range(n) if i not in set(assembled)]
        if correct not in remaining or len(remaining) < 2:
            continue
        steps += 1

        orb, mode = orbit_target_positions(
            remaining, correct, orbit_of, assembled, contact, restrict_by_contact=False
        )
        fro, _ = orbit_target_positions(
            remaining, correct, orbit_of, assembled, contact, restrict_by_contact=True
        )
        k = len(orb)
        ks.append(k)
        ks_frontier.append(len(fro))
        if k > 1:
            ambiguous += 1
            if len(fro) < k:
                frontier_resolved += 1

    if steps == 0:
        return {"steps": 0}

    return {
        "steps": steps,
        "ambiguous_steps": ambiguous,
        "ambiguous_frac": ambiguous / steps,
        "mean_orbit_k": float(np.mean(ks)),
        "max_orbit_k": int(np.max(ks)),
        "ce_floor": float(np.mean(np.log(ks))),
        "mean_frontier_k": float(np.mean(ks_frontier)),
        "ce_floor_frontier": float(np.mean(np.log(ks_frontier))),
        "frontier_resolved_frac": frontier_resolved / max(ambiguous, 1),
    }
