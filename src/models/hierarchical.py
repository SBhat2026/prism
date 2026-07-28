"""
hierarchical.py — Two-level (module → complex) assembly-order prediction.

Lever 2 of PRISM_scaling_roadmap.md.

Large complexes do not assemble subunit-by-subunit; preformed subcomplexes dock
as units (the NPC Y-complex, the flagellar C-ring / rod / hook). Mirroring that
caps the effective n per prediction at ~5-15 even for a 500-subunit target,
which bounds both the receptive field and the factorial output space.

Pipeline
--------
1. detect_modules()      — community detection on the buried-surface-area
                           interface graph (greedy modularity, resolution-tuned
                           to hit a target module size).
2. coarsen()             — build a module-level pseudo-complex: node features
                           are orbit-aware means, edge features are cross-module
                           interface means. The *same* AssemblyScorer runs on it.
3. hierarchical_order()  — predict the module order on the coarse graph, then
                           the within-module order on each induced subgraph,
                           and splice.

The scorer is shape-agnostic (dense N×N attention), so no retraining is needed
to run it at either level — the coarse graph is just another complex.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Module detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_modules(
    interface_matrix: np.ndarray,
    target_size: int = 8,
    min_modules: int = 2,
    max_modules: int = 24,
) -> list[list[int]]:
    """
    Partition subunits into modules from the weighted interface (BSA) graph.

    Returns a list of index lists covering 0..N-1. Falls back to a single module
    (i.e. flat prediction) when the graph is too small or degenerate.
    """
    n = interface_matrix.shape[0]
    if n <= max(target_size, 4):
        return [list(range(n))]

    w = np.array(interface_matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(w, 0.0)
    if w.max() <= 0:
        return [list(range(n))]
    w /= w.max()

    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
    except Exception:
        return _spectral_modules(w, target_size)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if w[i, j] > 1e-6:
                g.add_edge(i, j, weight=float(w[i, j]))
    if g.number_of_edges() == 0:
        return [list(range(n))]

    want = int(np.clip(round(n / target_size), min_modules, min(max_modules, n)))
    best: Optional[list[list[int]]] = None
    for res in (0.6, 1.0, 1.6, 2.4, 3.5):
        try:
            comms = greedy_modularity_communities(g, weight="weight", resolution=res)
        except Exception:
            continue
        parts = [sorted(int(x) for x in c) for c in comms]
        if best is None or abs(len(parts) - want) < abs(len(best) - want):
            best = parts
        if len(parts) == want:
            break

    if not best:
        return _spectral_modules(w, target_size)

    # Absorb singletons into their strongest-interface neighbour module.
    parts = [p for p in best if len(p) > 1]
    singles = [p[0] for p in best if len(p) == 1]
    if not parts:
        return [list(range(n))]
    for s in singles:
        scores = [w[s, p].sum() for p in parts]
        parts[int(np.argmax(scores))].append(s)
    return [sorted(p) for p in parts]


def _spectral_modules(w: np.ndarray, target_size: int) -> list[list[int]]:
    """networkx-free fallback: sign-of-Fiedler-vector recursive bisection."""
    n = w.shape[0]

    def _split(idx: list[int]) -> list[list[int]]:
        if len(idx) <= target_size:
            return [idx]
        sub = w[np.ix_(idx, idx)]
        deg = np.diag(sub.sum(1))
        try:
            vals, vecs = np.linalg.eigh(deg - sub)
        except np.linalg.LinAlgError:
            return [idx]
        f = vecs[:, 1] if vecs.shape[1] > 1 else vecs[:, 0]
        a = [idx[i] for i in range(len(idx)) if f[i] >= 0]
        b = [idx[i] for i in range(len(idx)) if f[i] < 0]
        if not a or not b:
            return [idx]
        return _split(a) + _split(b)

    return _split(list(range(n)))


# ──────────────────────────────────────────────────────────────────────────────
# Coarsening
# ──────────────────────────────────────────────────────────────────────────────

def coarsen(
    node_feats: torch.Tensor,     # (N, F)
    edge_feats: torch.Tensor,     # (N, N, E)
    modules: list[list[int]],
    copy_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the module-level pseudo-complex.

    Node m := mean of its members' node features.
    Edge (m, m') := mean of the member-pair edge features across the two modules
                    (i.e. the aggregated inter-module interface).
    """
    device = node_feats.device
    m = len(modules)
    cn = torch.stack([node_feats[torch.tensor(mod, device=device)].mean(0) for mod in modules])

    ce = torch.zeros(m, m, edge_feats.size(-1), device=device)
    for a in range(m):
        ia = torch.tensor(modules[a], device=device)
        for b in range(m):
            if a == b:
                continue
            ib = torch.tensor(modules[b], device=device)
            ce[a, b] = edge_feats[ia][:, ib].reshape(-1, edge_feats.size(-1)).mean(0)

    adj = torch.ones(m, m, dtype=torch.bool, device=device)
    if copy_indices is not None:
        ci = torch.tensor([int(copy_indices[mod[0]].item()) for mod in modules],
                          dtype=torch.long, device=device)
    else:
        ci = torch.zeros(m, dtype=torch.long, device=device)
    return cn, ce, adj, ci


def induce(
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    members: Sequence[int],
    copy_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Induced subgraph tensors for one module."""
    device = node_feats.device
    idx = torch.tensor(list(members), device=device)
    nf = node_feats[idx]
    ef = edge_feats[idx][:, idx]
    adj = torch.ones(len(members), len(members), dtype=torch.bool, device=device)
    ci = copy_indices[idx] if copy_indices is not None else torch.zeros(len(members), dtype=torch.long, device=device)
    return nf, ef, adj, ci


# ──────────────────────────────────────────────────────────────────────────────
# Two-level greedy
# ──────────────────────────────────────────────────────────────────────────────

def _greedy(model, nf, ef, adj, ci, seed: int = 0) -> list[int]:
    """Greedy assembly on an arbitrary (sub)graph using the shared scorer."""
    n = nf.size(0)
    if n <= 1:
        return list(range(n))
    model.eval()
    with torch.no_grad():
        h = model.embed(nf, adj, ef, ci)
        order = [seed]
        remaining = [i for i in range(n) if i != seed]
        while remaining:
            mask = torch.zeros(n, dtype=torch.bool, device=nf.device)
            mask[order] = True
            scores = model.score_candidates(h, ef, mask, remaining)
            pick = remaining[int(scores.argmax().item())]
            order.append(pick)
            remaining.remove(pick)
    return order


def hierarchical_order(
    model,
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    interface_matrix: np.ndarray,
    copy_indices: Optional[torch.Tensor] = None,
    target_size: int = 8,
    modules: Optional[list[list[int]]] = None,
) -> tuple[list[int], dict]:
    """
    Predict a full assembly order via module-order ∘ within-module-order.

    Returns (order, info) where info reports the module partition and the
    effective n actually seen by the network at each level.
    """
    n = node_feats.size(0)
    if modules is None:
        modules = detect_modules(interface_matrix, target_size=target_size)

    if len(modules) <= 1:
        return _greedy(model, node_feats, edge_feats,
                       torch.ones(n, n, dtype=torch.bool, device=node_feats.device),
                       copy_indices), {
            "n_modules": 1, "module_sizes": [n], "effective_n": n, "flat_fallback": True,
        }

    cn, ce, cadj, cci = coarsen(node_feats, edge_feats, modules, copy_indices)
    module_order = _greedy(model, cn, ce, cadj, cci)

    order: list[int] = []
    for m in module_order:
        members = modules[m]
        nf, ef, adj, ci = induce(node_feats, edge_feats, members, copy_indices)
        local = _greedy(model, nf, ef, adj, ci)
        order.extend(members[i] for i in local)

    info = {
        "n_modules": len(modules),
        "module_sizes": [len(m) for m in modules],
        "effective_n": max(len(modules), max(len(m) for m in modules)),
        "flat_fallback": False,
    }
    return order, info
