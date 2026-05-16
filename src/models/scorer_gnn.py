"""
GNN-based assembly compatibility scorer.

Architecture: Graph Attention Network v2 (GATv2) on the partial complex graph.
- Nodes: protein subunits, features = [ESM emb (640d) | pLDDT (5d) | SASA (1d) | chirality (6d)] = 652d
- Edges: interface contacts, features = [ppi | esm_cos | go_cos | sasa_norm | handedness] = 5d
- Output: per-candidate-addition compatibility score (raw logit for ranking CE loss)

Implemented in pure PyTorch (no PyG required).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# GATv2 building blocks
# ──────────────────────────────────────────────────────────────────────────────

class GATv2Layer(nn.Module):
    """Single-head GATv2 layer. W acts on the full concatenation before LeakyReLU,
    enabling cross-term interactions between source and target features.
    Attention: e_ij = a^T LeakyReLU(W_attn·[h_i || h_j || edge_ij])
    Value:     h_i' = σ(Σ_j alpha_ij · W_val·h_j)
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.W_attn = nn.Linear(2 * in_dim + edge_dim, out_dim, bias=True)
        self.W_val  = nn.Linear(in_dim, out_dim, bias=False)
        self.a      = nn.Linear(out_dim, 1, bias=False)
        self.drop   = nn.Dropout(dropout)
        self.leaky  = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,             # (N, in_dim)
        adj: torch.Tensor,           # (N, N) bool adjacency
        edge_feats: torch.Tensor,    # (N, N, edge_dim)
    ) -> torch.Tensor:
        N = x.size(0)
        src = x.unsqueeze(1).expand(N, N, -1)  # (N, N, in_dim)
        tgt = x.unsqueeze(0).expand(N, N, -1)  # (N, N, in_dim)
        cat = torch.cat([src, tgt, edge_feats], dim=-1)  # (N, N, 2*in+edge)

        # GATv2: W_attn acts on full concat before LeakyReLU
        e = self.a(self.leaky(self.W_attn(cat))).squeeze(-1)  # (N, N)
        e = e.masked_fill(~adj, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = self.drop(alpha)

        # Value projection then attention-weighted aggregation → (N, out_dim)
        val = self.W_val(x)                            # (N, out_dim)
        out = torch.einsum("ij,jk->ik", alpha, val)   # (N, out_dim)
        return F.elu(out)


class MultiHeadGATv2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 5,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0
        head_dim = out_dim // heads
        self.heads_list = nn.ModuleList([
            GATv2Layer(in_dim, head_dim, edge_dim, dropout) for _ in range(heads)
        ])
        # Projection to merge multi-head output back to in_dim for residual
        self.proj = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x, adj, edge_feats):
        heads = torch.cat([h(x, adj, edge_feats) for h in self.heads_list], dim=-1)
        return self.proj(heads)


# ──────────────────────────────────────────────────────────────────────────────
# Sparse binding probability MLP (Step 12)
# ──────────────────────────────────────────────────────────────────────────────

class BindingProb(nn.Module):
    """Fast pairwise binding probability from edge features alone.
    Used to prune to top-k neighbours before GATv2 message passing.
    ~200 parameters, independently interpretable.
    """

    def __init__(self, edge_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """edge_feats: (N, N, edge_dim) → (N, N) binding probabilities."""
        return self.net(edge_feats).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Full scorer
# ──────────────────────────────────────────────────────────────────────────────

MAX_COPIES = 16  # maximum copy-index for symmetry breaking embedding

class AssemblyScorer(nn.Module):
    """
    Scores candidate subunit additions to a partial complex.

    Input state (per step):
        node_feats   : (N, node_dim)   — ESM + pLDDT + SASA + chirality
        adj          : (N, N) bool     — interface adjacency
        edge_feats   : (N, N, 5)       — [ppi, esm_cos, go_cos, sasa_norm, handedness]
        assembled    : (N,) bool       — which subunits are already placed
        copy_indices : (N,) long       — 0-indexed copy count per unique UniProt
        candidates   : list[int]       — indices to score

    anchor_mode:
        "max"    — score(c) = max_a score(c, a) [original]
        "pooled" — score(c) = attention-pooled over assembled context [new default]
    """

    def __init__(
        self,
        node_dim: int = 652,
        edge_dim: int = 5,
        hidden: int = 128,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        anchor_mode: str = "pooled",
        sparse_k: int = 0,           # 0 = no pruning; >0 = top-k neighbour pruning
    ):
        super().__init__()
        self.anchor_mode = anchor_mode
        self.sparse_k = sparse_k
        self.edge_dim = edge_dim

        self.node_proj = nn.Linear(node_dim, hidden)

        # Copy-index embedding for symmetry breaking (Step 10)
        self.copy_embed = nn.Embedding(MAX_COPIES, 16)
        self.copy_proj  = nn.Linear(hidden + 16, hidden)

        # Sparse binding probability pre-filter (Step 12)
        self.binding_prob = BindingProb(edge_dim)

        # GATv2 layers with residual + layer norm
        self.layers = nn.ModuleList([
            MultiHeadGATv2(hidden, hidden, edge_dim, heads, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        # Context-gated anchor pooling attention gate (Step 11, anchor_mode="pooled")
        self.anchor_gate = nn.Linear(hidden, 1, bias=False)

        # Score head: [candidate_repr | complex_repr | mean_edge_to_complex]
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _sparse_adj(
        self,
        adj: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Prune adj to top-k neighbours per node by binding probability."""
        if self.sparse_k <= 0:
            return adj
        N = adj.size(0)
        probs = self.binding_prob(edge_feats)  # (N, N)
        probs = probs * adj.float()
        topk_vals, topk_idx = probs.topk(min(self.sparse_k, N), dim=-1)
        sparse = torch.zeros_like(adj, dtype=torch.bool)
        sparse.scatter_(1, topk_idx, True)
        # Always keep self and existing adj entries that made topk
        return sparse & adj

    def embed(
        self,
        node_feats: torch.Tensor,          # (N, node_dim)
        adj: torch.Tensor,                  # (N, N) bool
        edge_feats: torch.Tensor,           # (N, N, edge_dim)
        copy_indices: Optional[torch.Tensor] = None,  # (N,) long
    ) -> torch.Tensor:
        """Run GATv2 layers once; returns node embeddings h (N, hidden)."""
        h = F.relu(self.node_proj(node_feats))

        # Inject copy-index embedding for symmetry breaking
        if copy_indices is not None:
            ci = self.copy_embed(copy_indices.clamp(0, MAX_COPIES - 1))  # (N, 16)
            h = F.relu(self.copy_proj(torch.cat([h, ci], dim=-1)))

        # Sparse adjacency pruning
        sparse_adj = self._sparse_adj(adj, edge_feats)

        for layer, norm in zip(self.layers, self.layer_norms):
            h = norm(h + layer(h, sparse_adj, edge_feats))
        return h

    def score_candidates(
        self,
        h: torch.Tensor,                   # (N, hidden)
        edge_feats: torch.Tensor,           # (N, N, edge_dim)
        assembled_mask: torch.Tensor,       # (N,) bool
        candidates: list,
    ) -> torch.Tensor:
        """Score all candidates given precomputed h. Returns (C,) logits."""
        C = len(candidates)
        if C == 0:
            return torch.zeros(0, device=h.device)

        cand_t = torch.tensor(candidates, device=h.device, dtype=torch.long)
        cand_reprs = h[cand_t]  # (C, hidden)

        if assembled_mask.any():
            assembled_h = h[assembled_mask]  # (A, hidden)
            if self.anchor_mode == "pooled":
                # Context-gated anchor pooling: soft attention over assembled subunits
                gates = torch.softmax(self.anchor_gate(assembled_h), dim=0)  # (A, 1)
                complex_repr = (gates * assembled_h).sum(0)                  # (hidden,)
            else:
                # max: max-pool over assembled representations
                complex_repr = assembled_h.max(0).values                     # (hidden,)
            edge_to_complex = edge_feats[cand_t][:, assembled_mask].mean(1)  # (C, edge_dim)
        else:
            complex_repr = torch.zeros(h.size(1), device=h.device)
            edge_to_complex = torch.zeros(C, self.edge_dim, device=h.device)

        complex_reprs = complex_repr.unsqueeze(0).expand(C, -1)
        combined = torch.cat([cand_reprs, complex_reprs, edge_to_complex], dim=-1)
        return self.score_head(combined).squeeze(-1)  # (C,)

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        edge_feats: torch.Tensor,
        assembled_mask: torch.Tensor,
        candidate_idx: int,
        copy_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed(node_feats, adj, edge_feats, copy_indices)
        return self.score_candidates(h, edge_feats, assembled_mask, [candidate_idx])[0]


# ──────────────────────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────────────────────

def build_node_features(
    esm_embeddings: np.ndarray,            # (N, 640) ESM-2 150M
    go_probs: Optional[np.ndarray] = None, # (N, K) GO-MF logit probabilities
    sasa_values: Optional[np.ndarray] = None,  # (N,) total SASA per subunit
    plddt_stats: Optional[np.ndarray] = None,  # (N, 5) [mean, frac>90, frac<50, q10, q90]
    chirality_feats: Optional[np.ndarray] = None,  # (N, 6) [Cb_mean, Cb_var, phi_mean, phi_std, psi_mean, psi_std]
) -> torch.Tensor:
    """
    Concatenate ESM + optional GO summary + pLDDT + SASA + chirality into node feature matrix.
    GO probs summarised to 4 stats: [max, mean, std, n_active_frac].
    Total dim: 640 + 4 + 5 + 1 + 6 = 656d (or subset if optional features absent).
    """
    parts = [esm_embeddings]

    if go_probs is not None:
        go_max  = go_probs.max(axis=1, keepdims=True)
        go_mean = go_probs.mean(axis=1, keepdims=True)
        go_std  = go_probs.std(axis=1, keepdims=True)
        go_act  = (go_probs > 0.1).mean(axis=1, keepdims=True)
        parts.append(np.hstack([go_max, go_mean, go_std, go_act]))

    if plddt_stats is not None:
        parts.append(plddt_stats)

    if sasa_values is not None:
        sasa_norm = (sasa_values - sasa_values.mean()) / (sasa_values.std() + 1e-8)
        parts.append(sasa_norm.reshape(-1, 1))

    # CHIRALITY-SENSITIVE FEATURE — do not remove without ablation
    if chirality_feats is not None:
        parts.append(chirality_feats)

    return torch.tensor(np.hstack(parts), dtype=torch.float32)


def build_edge_features(
    ppi_matrix: np.ndarray,      # (N, N) STRING PPI scores [0,1]
    esm_cos_matrix: np.ndarray,  # (N, N) ESM cosine similarity
    go_cos_matrix: np.ndarray,   # (N, N) GO cosine similarity
    sasa_matrix: np.ndarray,     # (N, N) buried SASA per pair (nm²)
    handedness: Optional[np.ndarray] = None,  # (N, N) interface handedness scalar [-1, 1] — CHIRALITY-SENSITIVE FEATURE
) -> torch.Tensor:
    """Stack pairwise scoring matrices into edge feature tensor (N, N, 4 or 5)."""
    N = ppi_matrix.shape[0]
    sasa_norm = sasa_matrix / (sasa_matrix.max() + 1e-8)
    stacks = [ppi_matrix, esm_cos_matrix, go_cos_matrix, sasa_norm]
    if handedness is not None:
        stacks.append(handedness)  # CHIRALITY-SENSITIVE FEATURE
    return torch.tensor(np.stack(stacks, axis=-1), dtype=torch.float32)


def build_adjacency(contact_matrix: np.ndarray, threshold: float = 0.0) -> torch.Tensor:
    """Binary adjacency from contact/SASA matrix."""
    return torch.tensor(contact_matrix > threshold, dtype=torch.bool)


def build_copy_indices(uniprot_ids: list[str]) -> torch.Tensor:
    """
    Assign 0-indexed copy count per unique UniProt ID for symmetry breaking.
    e.g. [P69905, P68871, P69905, P68871] → [0, 0, 1, 1]
    """
    seen: dict[str, int] = {}
    indices = []
    for uid in uniprot_ids:
        idx = seen.get(uid, 0)
        seen[uid] = idx + 1
        indices.append(idx)
    return torch.tensor(indices, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic (non-GNN) scorers for ablation
# ──────────────────────────────────────────────────────────────────────────────

class WeightedLinearScorer:
    """Interpretable weighted sum scorer for ablation study."""

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
