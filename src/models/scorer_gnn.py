"""
GNN-based assembly compatibility scorer.

Architecture: Graph Attention Network (GAT) on the partial complex graph.
- Nodes: protein subunits, features = [ESM embedding (320d) | GO probs (K) | SASA norm (1)]
- Edges: interface contacts, features = [ppi_score | esm_cosine | go_cosine | sasa_buried]
- Output: per-candidate-addition compatibility score in [0,1]

Implemented in pure PyTorch (no PyG required) using sparse adjacency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """Single-head graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.a   = nn.Linear(2 * out_dim + edge_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,       # (N, in_dim)
        adj: torch.Tensor,     # (N, N) bool adjacency
        edge_feats: torch.Tensor,  # (N, N, edge_dim)
    ) -> torch.Tensor:
        N = x.size(0)
        Wx = self.W(x)  # (N, out_dim)

        # Attention coefficients
        src = Wx.unsqueeze(1).expand(N, N, -1)   # (N, N, out_dim)
        tgt = Wx.unsqueeze(0).expand(N, N, -1)   # (N, N, out_dim)
        cat = torch.cat([src, tgt, edge_feats], dim=-1)  # (N, N, 2*out+edge_dim)
        e   = self.leaky(self.a(cat).squeeze(-1))  # (N, N)

        # Mask non-edges
        mask = ~adj
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = self.drop(alpha)

        # Aggregate
        out = torch.bmm(alpha.unsqueeze(1), Wx.unsqueeze(0).expand(N, N, -1))
        # (N, 1, N) x (N, N, out_dim) → batched, need reshape
        # Use einsum instead
        out = torch.einsum("ij,jk->ik", alpha, Wx)  # (N, out_dim)
        return F.elu(out)


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 4,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0
        head_dim = out_dim // heads
        self.heads_list = nn.ModuleList([
            GATLayer(in_dim, head_dim, edge_dim, dropout) for _ in range(heads)
        ])

    def forward(self, x, adj, edge_feats):
        return torch.cat([h(x, adj, edge_feats) for h in self.heads_list], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Full scorer
# ──────────────────────────────────────────────────────────────────────────────

class AssemblyScorer(nn.Module):
    """
    Scores candidate subunit additions to a partial complex.

    Input state (per step):
        node_feats   : (N, node_dim)   — ESM emb + GO probs + SASA
        adj          : (N, N) bool     — which pairs share an interface
        edge_feats   : (N, N, 4)       — [ppi, esm_cos, go_cos, sasa_norm]
        assembled    : (N,) bool       — which subunits are already in complex
        candidate    : int             — index of subunit being scored

    Output: scalar compatibility score in [0,1]
    """

    def __init__(
        self,
        node_dim: int = 324,   # 320 ESM + 4 summary GO features (after projection)
        edge_dim: int = 4,
        hidden: int = 256,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList([
            MultiHeadGAT(hidden, hidden, edge_dim, heads, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        # Score head: [candidate_repr | mean_complex_repr | edge_feats_to_complex]
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            # No sigmoid — raw logits for ranking CE loss
        )

    def embed(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Run GNN layers once; returns node embeddings h (N, hidden)."""
        h = F.relu(self.node_proj(node_feats))
        for layer, norm in zip(self.layers, self.layer_norms):
            h = norm(h + layer(h, adj, edge_feats))
        return h

    def score_candidates(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
        assembled_mask: torch.Tensor,
        candidates: list,
    ) -> torch.Tensor:
        """Score all candidates given precomputed h. Returns (C,) logits.
        Batches all candidates through the score head in one pass."""
        C = len(candidates)
        if C == 0:
            return torch.zeros(0, device=h.device)
        cand_t = torch.tensor(candidates, device=h.device, dtype=torch.long)
        cand_reprs = h[cand_t]  # (C, hidden)
        if assembled_mask.any():
            complex_repr = h[assembled_mask].mean(0)  # (hidden,)
            edge_to_complex = edge_feats[cand_t][:, assembled_mask].mean(1)  # (C, 4)
        else:
            complex_repr = torch.zeros(h.size(1), device=h.device)
            edge_to_complex = torch.zeros(C, edge_feats.size(-1), device=h.device)
        complex_reprs = complex_repr.unsqueeze(0).expand(C, -1)  # (C, hidden)
        combined = torch.cat([cand_reprs, complex_reprs, edge_to_complex], dim=-1)
        return self.score_head(combined).squeeze(-1)  # (C,)

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        edge_feats: torch.Tensor,
        assembled_mask: torch.Tensor,
        candidate_idx: int,
    ) -> torch.Tensor:
        h = self.embed(node_feats, adj, edge_feats)
        return self.score_candidates(h, edge_feats, assembled_mask, [candidate_idx])[0]


# ──────────────────────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────────────────────

def build_node_features(
    esm_embeddings: np.ndarray,   # (N, 320)
    go_probs: Optional[np.ndarray] = None,  # (N, K)
    sasa_values: Optional[np.ndarray] = None,  # (N,) total SASA per subunit
) -> torch.Tensor:
    """
    Concatenate ESM embeddings + optional GO summary stats + SASA into node feature matrix.
    GO probs are summarized to 4 dims: [max, mean, std, n_active_frac].
    """
    parts = [esm_embeddings]

    if go_probs is not None:
        go_max  = go_probs.max(axis=1, keepdims=True)
        go_mean = go_probs.mean(axis=1, keepdims=True)
        go_std  = go_probs.std(axis=1, keepdims=True)
        go_act  = (go_probs > 0.1).mean(axis=1, keepdims=True)
        parts.append(np.hstack([go_max, go_mean, go_std, go_act]))

    if sasa_values is not None:
        sasa_norm = (sasa_values - sasa_values.mean()) / (sasa_values.std() + 1e-8)
        parts.append(sasa_norm.reshape(-1, 1))

    return torch.tensor(np.hstack(parts), dtype=torch.float32)


def build_edge_features(
    ppi_matrix: np.ndarray,    # (N, N) STRING PPI scores [0,1]
    esm_cos_matrix: np.ndarray,  # (N, N) ESM cosine similarity
    go_cos_matrix: np.ndarray,   # (N, N) GO cosine similarity
    sasa_matrix: np.ndarray,     # (N, N) buried SASA per pair (nm²), normalized
) -> torch.Tensor:
    """Stack four pairwise scoring matrices into edge feature tensor (N, N, 4)."""
    sasa_norm = sasa_matrix / (sasa_matrix.max() + 1e-8)
    stack = np.stack([ppi_matrix, esm_cos_matrix, go_cos_matrix, sasa_norm], axis=-1)
    return torch.tensor(stack, dtype=torch.float32)


def build_adjacency(contact_matrix: np.ndarray, threshold: float = 0.0) -> torch.Tensor:
    """Binary adjacency from contact/SASA matrix."""
    return torch.tensor(contact_matrix > threshold, dtype=torch.bool)


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic (non-GNN) scorers for ablation
# ──────────────────────────────────────────────────────────────────────────────

class WeightedLinearScorer:
    """
    Interpretable weighted sum scorer for ablation study.
    This is what the full GNN is competing against.
    """

    def __init__(
        self,
        w_sasa: float = 1.0,
        w_ppi:  float = 0.0,
        w_esm:  float = 0.0,
        w_go:   float = 0.0,
    ):
        self.w_sasa = w_sasa
        self.w_ppi  = w_ppi
        self.w_esm  = w_esm
        self.w_go   = w_go

    def score(
        self,
        candidate_idx: int,
        assembled_indices: list[int],
        sasa_matrix: np.ndarray,
        ppi_matrix: np.ndarray,
        esm_matrix: np.ndarray,
        go_matrix: np.ndarray,
    ) -> float:
        if not assembled_indices:
            return 0.0

        sasa = sasa_matrix[candidate_idx, assembled_indices].max()
        ppi  = ppi_matrix[candidate_idx, assembled_indices].max()
        esm  = esm_matrix[candidate_idx, assembled_indices].max()
        go   = go_matrix[candidate_idx, assembled_indices].max()

        sasa_n = sasa / (sasa_matrix.max() + 1e-8)
        return (self.w_sasa * sasa_n + self.w_ppi * ppi +
                self.w_esm * esm + self.w_go * go)
