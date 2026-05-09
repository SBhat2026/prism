"""PRISM GAT assembly scorer — self-contained copy for HF Space."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=4, dropout=0.1):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.a    = nn.Linear(2 * out_dim + edge_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, adj, edge_feats):
        N = x.size(0)
        Wx  = self.W(x)
        src = Wx.unsqueeze(1).expand(N, N, -1)
        tgt = Wx.unsqueeze(0).expand(N, N, -1)
        e   = self.leaky(self.a(torch.cat([src, tgt, edge_feats], dim=-1)).squeeze(-1))
        e   = e.masked_fill(~adj, float("-inf"))
        alpha = self.drop(F.softmax(e, dim=-1))
        return F.elu(torch.einsum("ij,jk->ik", alpha, Wx))


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=4, heads=4, dropout=0.1):
        super().__init__()
        head_dim = out_dim // heads
        self.heads_list = nn.ModuleList([
            GATLayer(in_dim, head_dim, edge_dim, dropout) for _ in range(heads)
        ])

    def forward(self, x, adj, edge_feats):
        return torch.cat([h(x, adj, edge_feats) for h in self.heads_list], dim=-1)


class AssemblyScorer(nn.Module):
    def __init__(self, node_dim=646, edge_dim=4, hidden=128,
                 n_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.layers    = nn.ModuleList([
            MultiHeadGAT(hidden, hidden, edge_dim, heads, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.score_head  = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_feats, adj, edge_feats, assembled_mask, candidate_idx):
        h = F.relu(self.node_proj(node_feats))
        for layer, norm in zip(self.layers, self.layer_norms):
            h = norm(h + layer(h, adj, edge_feats))
        cand_repr    = h[candidate_idx]
        complex_repr = h[assembled_mask].mean(0) if assembled_mask.any() \
                       else torch.zeros_like(cand_repr)
        # Mean edge features from candidate to assembled set
        N = node_feats.size(0)
        asm_idx = assembled_mask.nonzero(as_tuple=True)[0]
        edge_summary = edge_feats[candidate_idx, asm_idx].mean(0) \
                       if len(asm_idx) > 0 \
                       else edge_feats[candidate_idx].mean(0)
        score_in = torch.cat([cand_repr, complex_repr, edge_summary])
        return self.score_head(score_in).squeeze(-1)


def build_edge_features(ppi, esm_cos, go_cos, sasa_norm):
    N = ppi.shape[0]
    ef = np.stack([ppi, esm_cos, go_cos, sasa_norm], axis=-1).astype(np.float32)
    return torch.tensor(ef, dtype=torch.float32)
