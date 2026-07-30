"""
attribution.py — what is the GNN actually using to make this choice?

The weighted heuristic scorer decomposes trivially: its score IS
w_sasa·sasa + w_esm·esm + w_go·go + w_ppi·ppi, so per-channel contributions are
free. The GNN has no such decomposition, and the viewer used to paper over that
by reporting a flat 0.5 for every channel except SASA — which read as "the model
weighs ESM/GO/PPI at exactly 0.5" when it actually meant "nobody computed this".

This computes the honest analogue: occlusion attribution. Each edge channel is
neutralised in turn — replaced by its own off-diagonal mean, which removes the
variation while preserving scale and matrix shape — and we re-score. How far the
candidate's score moves is how much that channel mattered for this decision.

Neutralising with the mean rather than zeroing matters: zeroing conflates "this
channel is uninformative" with "this channel has been shifted off its training
distribution", and the latter produces large spurious attributions.

Attribution is signed. Positive means the channel was pushing this candidate up.
"""

from __future__ import annotations

import numpy as np
import torch


def _neutralise(edge_feats: torch.Tensor, ch: int) -> torch.Tensor:
    """Replace one edge channel by its off-diagonal mean."""
    out = edge_feats.clone()
    n = edge_feats.shape[0]
    if n < 2:
        return out
    off = ~torch.eye(n, dtype=torch.bool, device=edge_feats.device)
    out[:, :, ch] = torch.where(off, edge_feats[:, :, ch][off].mean(),
                                edge_feats[:, :, ch])
    return out


@torch.no_grad()
def edge_channel_attribution(
    model,
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    adj: torch.Tensor,
    assembled_mask: torch.Tensor,
    candidates: list[int],
    channels: dict[str, int],
    copy_indices: torch.Tensor | None = None,
    reembed: bool = True,
) -> dict[str, np.ndarray]:
    """Signed per-candidate attribution for each named edge channel.

    Returns {channel_name: (C,) array of (base_score - occluded_score)}.

    reembed=True re-runs the message-passing stack with the channel neutralised,
    so the attribution covers the channel's effect on the node embeddings as
    well as on the scoring head. That is the honest total; it costs one extra
    embed per channel, which is negligible at the handful-of-chains scale the
    viewer works at. reembed=False attributes only the scoring head.
    """
    base_h = model.embed(node_feats, adj, edge_feats, copy_indices)
    base = model.score_candidates(base_h, edge_feats, assembled_mask,
                                  candidates).detach().cpu().numpy()

    out: dict[str, np.ndarray] = {}
    for name, ch in channels.items():
        if ch >= edge_feats.shape[-1]:
            out[name] = np.zeros(len(candidates), dtype=np.float32)
            continue
        ef2 = _neutralise(edge_feats, ch)
        h2 = model.embed(node_feats, adj, ef2, copy_indices) if reembed else base_h
        occ = model.score_candidates(h2, ef2, assembled_mask,
                                     candidates).detach().cpu().numpy()
        out[name] = (base - occ).astype(np.float32)
    return out


def normalise_attribution(attr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scale attributions to fractions of the total absolute movement, per
    candidate, so the viewer can draw comparable bars. Channels that did nothing
    stay at 0 rather than being inflated to fill the bar."""
    if not attr:
        return {}
    names = list(attr)
    stacked = np.stack([attr[n] for n in names])           # (K, C)
    denom = np.abs(stacked).sum(axis=0, keepdims=True)     # (1, C)
    denom = np.where(denom < 1e-12, 1.0, denom)
    scaled = stacked / denom
    return {n: scaled[i] for i, n in enumerate(names)}
