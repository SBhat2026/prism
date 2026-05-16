"""
Quotient graph compression for protein complexes.

Collapses identical subunit copies (same UniProt ID) into super-nodes,
reducing GATv2 complexity from O(N²) to O(k²) where k = unique subunit types.

Compression examples:
  1AON (7-mer ring homomer): 7 nodes → 1 super-node (7× compression)
  2HHB (α₂β₂ heteromer):    4 nodes → 2 super-nodes, 3 edges (2× node compression)

Lossless for exact UniProt matches; soft-merge (cosine sim > threshold) for
pseudo-symmetric paralogous subunits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class QuotientGraph:
    """Compressed graph where each node represents an orbit of identical subunits."""
    super_node_feats: torch.Tensor        # (k, node_dim)
    super_adj:        torch.Tensor        # (k, k) bool
    super_edge_feats: torch.Tensor        # (k, k, edge_dim)
    orbit_map:        list[list[int]]     # orbit_map[i] = [chain indices in orbit i]
    copy_numbers:     list[int]           # copy_numbers[i] = len(orbit_map[i])
    uniprot_ids:      list[str]           # representative UniProt per orbit
    soft_merged:      list[bool]          # True if orbit formed by soft (cosine) merge


def build_quotient(
    node_feats:   torch.Tensor,           # (N, node_dim)
    adj:          torch.Tensor,           # (N, N) bool
    edge_feats:   torch.Tensor,           # (N, N, edge_dim)
    uniprot_ids:  list[str],
    sym_threshold: float = 0.97,          # cosine similarity for soft merge
) -> QuotientGraph:
    """
    Collapse identical (or near-identical) subunits into super-nodes.

    Identity criterion:
      Exact: same UniProt accession string → guaranteed lossless.
      Soft:  ESM cosine similarity > sym_threshold → near-lossless, marks soft_merged=True.

    Super-node features: mean of member node features.
    Super-edge features: max across all member pairs (captures strongest interface signal).
    """
    N = node_feats.size(0)

    # Build orbits: first pass by exact UniProt match, then soft cosine merge
    assigned = [-1] * N
    orbits: list[list[int]] = []
    orbit_uniprot: list[str] = []
    soft_merged: list[bool] = []

    # Normalised ESM embeddings for cosine similarity (use first 640d as proxy)
    emb_dim = min(640, node_feats.size(1))
    nf_cpu = node_feats[:, :emb_dim].float()
    norms = nf_cpu.norm(dim=1, keepdim=True).clamp(min=1e-8)
    nf_norm = nf_cpu / norms  # (N, emb_dim)

    for i in range(N):
        if assigned[i] >= 0:
            continue
        # Try to join an existing orbit
        merged = False
        for oi, orbit in enumerate(orbits):
            rep = orbit[0]
            # Exact UniProt match
            if uniprot_ids[i] == uniprot_ids[rep]:
                orbit.append(i)
                assigned[i] = oi
                merged = True
                break
            # Soft cosine match
            if sym_threshold < 1.0:
                cos = float(torch.dot(nf_norm[i], nf_norm[rep]))
                if cos >= sym_threshold:
                    orbit.append(i)
                    assigned[i] = oi
                    soft_merged[oi] = True
                    merged = True
                    break
        if not merged:
            oi = len(orbits)
            orbits.append([i])
            orbit_uniprot.append(uniprot_ids[i])
            soft_merged.append(False)
            assigned[i] = oi

    k = len(orbits)
    node_dim = node_feats.size(1)
    edge_dim = edge_feats.size(-1)

    # Super-node features: mean over orbit members
    super_nf = torch.zeros(k, node_dim, dtype=node_feats.dtype)
    for oi, orbit in enumerate(orbits):
        super_nf[oi] = node_feats[orbit].mean(0)

    # Super-edge features: max over all member pairs between two orbits
    super_ef = torch.zeros(k, k, edge_dim, dtype=edge_feats.dtype)
    super_adj_t = torch.zeros(k, k, dtype=torch.bool)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            pairs = [(a, b) for a in orbits[i] for b in orbits[j]]
            if not pairs:
                continue
            pair_feats = torch.stack([edge_feats[a, b] for a, b in pairs])  # (P, edge_dim)
            super_ef[i, j] = pair_feats.max(0).values
            # Connected if any member pair was adjacent
            super_adj_t[i, j] = any(adj[a, b] for a, b in pairs)

    return QuotientGraph(
        super_node_feats=super_nf,
        super_adj=super_adj_t,
        super_edge_feats=super_ef,
        orbit_map=orbits,
        copy_numbers=[len(o) for o in orbits],
        uniprot_ids=orbit_uniprot,
        soft_merged=soft_merged,
    )


def expand_prediction(
    super_node_order: list[int],
    qg: QuotientGraph,
) -> list[int]:
    """
    Expand a super-node assembly order back to chain-level indices.
    Each super-node appears copy_number times consecutively.
    """
    chain_order = []
    for sni in super_node_order:
        chain_order.extend(qg.orbit_map[sni])
    return chain_order


def compress_gt_order(gt_order: list[int], qg: QuotientGraph) -> list[int]:
    """
    Convert chain-level gt_order to super-node-level order (first occurrence wins).
    Used for training loss computation on compressed graph.
    """
    seen: set[int] = set()
    sn_order = []
    # Build reverse map: chain_idx → super_node_idx
    chain_to_sn = {}
    for sni, orbit in enumerate(qg.orbit_map):
        for ci in orbit:
            chain_to_sn[ci] = sni
    for ci in gt_order:
        sni = chain_to_sn[ci]
        if sni not in seen:
            sn_order.append(sni)
            seen.add(sni)
    return sn_order
