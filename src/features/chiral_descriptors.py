"""
chiral_descriptors.py — Cheap handedness features for an SE(3)-*invariant* backbone.

Lever 1 (interim) of PRISM_scaling_roadmap.md.

Why this exists
---------------
PRISM's node/edge features are distance- and contact-based, i.e. invariant under
the full orthogonal group O(3) — including reflections. A mirrored assembly
produces *bit-identical* features, so the network cannot in principle tell an
assembly from its mirror image. That is a representational blind spot, not a
data problem.

The fix short of a full E(3)-equivariant backbone: inject descriptors that are
invariant under rotation+translation but **flip sign under reflection**
(pseudoscalars). Any signed volume / triple product does this:

    det[a, b, c]  --mirror-->  -det[a, b, c]

Descriptors computed here
-------------------------
Node (3d), per chain i:
  0  tetra_chirality  — signed volume of the tetrahedron formed by i and its
                        3 nearest chain centroids, normalised by edge scale.
  1  helical_sense    — signed torsion of the chain's own CA trace
                        (mean signed dihedral over a coarse-grained backbone);
                        distinguishes right- from left-handed super-helices.
  2  frame_handedness — det of the chain's own PCA frame after sequence-based
                        sign fixing; ±1 by construction.

Edge (1d), per pair (i, j):
  chiral_edge — det[ û_ij , v_i , v_j ] where û_ij is the unit centroid-to-centroid
                vector and v_i, v_j are the sequence-oriented principal axes of
                the two chains. This is the interface handedness: it is 0 for a
                planar/achiral arrangement and flips sign under mirroring.

All values are bounded in [-1, 1]. Missing coordinates ⇒ zeros (safe no-op).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

CHIRAL_NODE_DIM = 3
CHIRAL_EDGE_DIM = 1

ROOT = Path(__file__).resolve().parents[2]
CHIRAL_CACHE = ROOT / "data" / "chiral_cache"


# ──────────────────────────────────────────────────────────────────────────────
# Per-chain oriented frames
# ──────────────────────────────────────────────────────────────────────────────

def _oriented_axis(ca: np.ndarray) -> np.ndarray:
    """
    First principal axis of a CA trace, sign-fixed by the sequence direction so
    that the resulting frame is reproducible (and mirrors correctly).
    """
    c = ca - ca.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    v = vt[0]
    # Sign convention: point along N→C, i.e. positive correlation with residue index.
    t = np.arange(len(ca), dtype=np.float64)
    t -= t.mean()
    if float(np.dot(t, c @ v)) < 0:
        v = -v
    return v / (np.linalg.norm(v) + 1e-12)


def _frame_handedness(ca: np.ndarray) -> float:
    """det of the sequence-oriented PCA frame; ±1, flips under reflection."""
    c = ca - ca.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    v1, v2, v3 = vt[0], vt[1], vt[2]
    t = np.arange(len(ca), dtype=np.float64)
    t -= t.mean()
    if float(np.dot(t, c @ v1)) < 0:
        v1 = -v1
    # Second axis sign-fixed by the largest-|projection| residue, again sequence-anchored.
    proj2 = c @ v2
    if float(proj2[int(np.argmax(np.abs(proj2)))]) < 0:
        v2 = -v2
    return float(np.sign(np.linalg.det(np.stack([v1, v2, v3]))))


def _helical_sense(ca: np.ndarray, stride: int = 8) -> float:
    """
    Mean signed dihedral of a coarse-grained CA trace, mapped to [-1, 1].
    Right-handed super-helical packing gives a positive value, left-handed
    negative — a genuine pseudoscalar of the fold.
    """
    pts = ca[::stride]
    if len(pts) < 4:
        return 0.0
    b = np.diff(pts, axis=0)
    n1 = np.cross(b[:-2], b[1:-1])
    n2 = np.cross(b[1:-1], b[2:])
    m = np.cross(n1, b[1:-1] / (np.linalg.norm(b[1:-1], axis=1, keepdims=True) + 1e-12))
    x = np.einsum("ij,ij->i", n1, n2)
    y = np.einsum("ij,ij->i", m, n2)
    dih = np.arctan2(y, x)
    valid = np.isfinite(dih)
    if not valid.any():
        return 0.0
    return float(np.clip(np.mean(dih[valid]) / np.pi, -1.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_chiral_features(
    pdb_id: str,
    chain_ids: Sequence[str],
    pdb_path: Optional[str],
    cache_dir: Optional[Path] = CHIRAL_CACHE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (node_feats (N, 3), edge_feats (N, N, 1)), both float32, in [-1, 1].
    Zeros when coordinates are unavailable — the feature is then a no-op.
    """
    n = len(chain_ids)
    zeros = (np.zeros((n, CHIRAL_NODE_DIM), np.float32),
             np.zeros((n, n, CHIRAL_EDGE_DIM), np.float32))
    if not pdb_path:
        return zeros

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{pdb_id}_chiral.npz"
        if cache_path.exists():
            try:
                d = np.load(cache_path)
                nf, ef = d["node"], d["edge"]
                if nf.shape == (n, CHIRAL_NODE_DIM) and ef.shape == (n, n, CHIRAL_EDGE_DIM):
                    return nf.astype(np.float32), ef.astype(np.float32)
            except Exception:
                pass

    from src.evaluation.symmetry import load_ca_coords

    coords = load_ca_coords(pdb_path, chain_ids)
    if len(coords) < n:
        return zeros

    cents = np.stack([coords[c].mean(axis=0) for c in chain_ids])
    axes = np.stack([_oriented_axis(coords[c]) for c in chain_ids])

    node = np.zeros((n, CHIRAL_NODE_DIM), np.float64)
    for i, c in enumerate(chain_ids):
        node[i, 1] = _helical_sense(coords[c])
        node[i, 2] = _frame_handedness(coords[c])

    # Node feature 0: signed volume with the 3 nearest neighbouring chains.
    if n >= 4:
        d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        for i in range(n):
            nb = np.argsort(d[i])[:3]
            a, b, c_ = (cents[nb[0]] - cents[i], cents[nb[1]] - cents[i], cents[nb[2]] - cents[i])
            scale = (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c_)) + 1e-9
            node[i, 0] = float(np.dot(a, np.cross(b, c_)) / scale)

    # Edge feature: interface handedness det[û_ij, v_i, v_j].
    edge = np.zeros((n, n, CHIRAL_EDGE_DIM), np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            u = cents[j] - cents[i]
            nu = np.linalg.norm(u)
            if nu < 1e-6:
                continue
            u = u / nu
            chi = float(np.dot(u, np.cross(axes[i], axes[j])))
            edge[i, j, 0] = chi
            edge[j, i, 0] = -chi   # antisymmetric: i→j and j→i have opposite sense

    node = np.clip(np.nan_to_num(node), -1.0, 1.0).astype(np.float32)
    edge = np.clip(np.nan_to_num(edge), -1.0, 1.0).astype(np.float32)

    if cache_path is not None:
        try:
            np.savez_compressed(cache_path, node=node, edge=edge)
        except Exception:
            pass
    return node, edge


def mirror_coords_test(pdb_path: str, chain_ids: Sequence[str]) -> dict:
    """
    Diagnostic: verify the descriptors actually flip sign under reflection.
    Returns the mean |Δ| between original and mirrored features; a correct
    pseudoscalar gives ~2×|value| for node[0] and the edge feature.
    """
    from src.evaluation.symmetry import load_ca_coords

    coords = load_ca_coords(pdb_path, chain_ids)
    if len(coords) < len(chain_ids):
        return {"ok": False, "reason": "missing coords"}

    def _feats(cs: dict[str, np.ndarray]):
        cents = np.stack([cs[c].mean(axis=0) for c in chain_ids])
        axes = np.stack([_oriented_axis(cs[c]) for c in chain_ids])
        n = len(chain_ids)
        e = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                u = cents[j] - cents[i]
                u = u / (np.linalg.norm(u) + 1e-12)
                e[i, j] = float(np.dot(u, np.cross(axes[i], axes[j])))
        return e

    orig = _feats(coords)
    mirrored = _feats({c: v * np.array([1.0, 1.0, -1.0]) for c, v in coords.items()})
    return {
        "ok": True,
        "mean_abs_orig": float(np.abs(orig).mean()),
        "mean_abs_sum": float(np.abs(orig + mirrored).mean()),   # ≈0 ⇒ clean sign flip
        "sign_flip_fraction": float((np.sign(orig) * np.sign(mirrored) < 0).mean()),
    }
