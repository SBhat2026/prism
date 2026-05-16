"""
Chirality-sensitive geometric features derived from PDB 3D coordinates.

All features are self-supervised from PDB geometry — no database chirality labels needed.
The model learns to associate these patterns with binding outcomes through gradient descent.

NOTE: ESM-2 is trained exclusively on L-amino acid proteins. D-proteins are OOD.
SASA is the one existing feature that is chirality-aware. These features augment it.
For D-protein prediction, a mirror-image protein language model would be required.

CHIRALITY-SENSITIVE FEATURE — do not remove without ablation.

Three feature groups added to the pipeline:
  Edge (+1d): interface_handedness — determinant of rotation between chain frames
  Node (+2d): cbeta_orientation   — Cα→Cβ projection at interface residues
  Node (+4d): backbone_dihedral_stats — φ, ψ mean/std at interface residues
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate utilities
# ──────────────────────────────────────────────────────────────────────────────

def _parse_pdb_coords(pdb_path: str | Path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """
    Parse PDB ATOM records into {chain: {resnum: {atom_name: xyz}}}.
    Keeps only standard amino acids; skips HETATM.
    """
    _AA3 = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    }
    coords: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            if resname not in _AA3:
                continue
            chain = line[21].strip() or "_"
            resnum = int(line[22:26].strip())
            atom = line[12:16].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            coords.setdefault(chain, {}).setdefault(resnum, {})[atom] = np.array([x, y, z])
    return coords


def _interface_residues(
    coords_a: dict[int, dict[str, np.ndarray]],
    coords_b: dict[int, dict[str, np.ndarray]],
    cutoff: float = 8.0,
) -> tuple[list[int], list[int]]:
    """Return residue numbers within cutoff Å of the opposing chain (CA–CA distance)."""
    ca_a = {r: v["CA"] for r, v in coords_a.items() if "CA" in v}
    ca_b = {r: v["CA"] for r, v in coords_b.items() if "CA" in v}
    if not ca_a or not ca_b:
        return [], []
    arr_a = np.stack(list(ca_a.values()))  # (Ra, 3)
    arr_b = np.stack(list(ca_b.values()))  # (Rb, 3)
    dists = np.linalg.norm(arr_a[:, None] - arr_b[None, :], axis=-1)  # (Ra, Rb)
    close_a = np.where(dists.min(axis=1) < cutoff)[0]
    close_b = np.where(dists.min(axis=0) < cutoff)[0]
    resnums_a = list(ca_a.keys())
    resnums_b = list(ca_b.keys())
    return ([resnums_a[i] for i in close_a], [resnums_b[i] for i in close_b])


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: Interface handedness (new 5th edge feature)
# ──────────────────────────────────────────────────────────────────────────────

def interface_handedness(
    coords_a: dict[int, dict[str, np.ndarray]],
    coords_b: dict[int, dict[str, np.ndarray]],
    cutoff: float = 8.0,
) -> float:
    """
    Compute the handedness of the rigid body transformation from chain A's interface
    centroid frame to chain B's interface centroid frame.

    Returns det(R) ∈ {+1, -1, 0}:
      +1 : proper rotation — same chirality preserved at interface
      -1 : improper rotation — mirror relationship between chains
       0 : insufficient interface residues to compute
    """
    res_a, res_b = _interface_residues(coords_a, coords_b, cutoff)
    if len(res_a) < 3 or len(res_b) < 3:
        return 0.0

    # Centroid frames from interface Cα
    ca_a = np.array([coords_a[r]["CA"] for r in res_a if "CA" in coords_a[r]])
    ca_b = np.array([coords_b[r]["CA"] for r in res_b if "CA" in coords_b[r]])
    if len(ca_a) < 3 or len(ca_b) < 3:
        return 0.0

    cen_a = ca_a.mean(0)
    cen_b = ca_b.mean(0)
    ca_a_c = ca_a - cen_a
    ca_b_c = ca_b - cen_b

    # Truncate to same length for SVD-based rotation estimation
    n = min(len(ca_a_c), len(ca_b_c))
    H = ca_a_c[:n].T @ ca_b_c[:n]  # (3, 3) covariance
    try:
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        return float(np.sign(np.linalg.det(R)))
    except np.linalg.LinAlgError:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2: Cβ orientation at interface residues (2 new node features)
# ──────────────────────────────────────────────────────────────────────────────

def cbeta_orientation(
    chain_coords: dict[int, dict[str, np.ndarray]],
    interface_res: list[int],
) -> tuple[float, float]:
    """
    Compute mean and variance of Cα→Cβ unit vector projections onto the interface normal.

    Returns (mean_projection, circular_variance) ∈ [-1, 1] × [0, 1].
    L-amino acids cluster in a characteristic Ramachandran region; D-amino acids mirror it.
    """
    if not interface_res:
        return 0.0, 0.0

    # Interface normal: PCA of interface Cα positions
    ca_pts = np.array([chain_coords[r]["CA"] for r in interface_res
                       if r in chain_coords and "CA" in chain_coords[r]])
    if len(ca_pts) < 2:
        return 0.0, 0.0

    cen = ca_pts.mean(0)
    _, _, Vt = np.linalg.svd(ca_pts - cen)
    normal = Vt[-1]  # smallest singular value → normal to interface plane
    normal /= np.linalg.norm(normal) + 1e-8

    projections = []
    for r in interface_res:
        if r not in chain_coords:
            continue
        res = chain_coords[r]
        if "CA" not in res:
            continue
        if "CB" in res:
            vec = res["CB"] - res["CA"]
        elif "C" in res and "N" in res:
            # GLY: estimate Cβ direction from backbone
            vec = np.cross(res["N"] - res["CA"], res["C"] - res["CA"])
        else:
            continue
        norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-8:
            continue
        proj = float(np.dot(vec / norm_vec, normal))
        projections.append(proj)

    if not projections:
        return 0.0, 0.0
    arr = np.array(projections)
    return float(arr.mean()), float(arr.var())


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3: Backbone dihedral statistics at interface residues (4 new node features)
# ──────────────────────────────────────────────────────────────────────────────

def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute dihedral angle (in radians) defined by four points."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = math.acos(cos_angle)
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    return angle


def backbone_dihedral_stats(
    chain_coords: dict[int, dict[str, np.ndarray]],
    interface_res: list[int],
) -> tuple[float, float, float, float]:
    """
    Compute mean and std of φ and ψ backbone dihedral angles at interface residues.

    L-amino acids cluster at φ∈[-180°,0°], ψ∈[-180°,0°].
    D-amino acids mirror to opposite quadrant — model learns this without labels.

    Returns (mean_phi, std_phi, mean_psi, std_psi) in radians, normalised to [-1, 1].
    """
    sorted_res = sorted(chain_coords.keys())
    res_set = set(interface_res)
    phis, psis = [], []

    for i, r in enumerate(sorted_res):
        if r not in res_set:
            continue
        res = chain_coords[r]
        if "N" not in res or "CA" not in res or "C" not in res:
            continue

        # φ = dihedral(C_prev, N, CA, C)
        if i > 0:
            r_prev = sorted_res[i - 1]
            prev = chain_coords.get(r_prev, {})
            if "C" in prev:
                phi = _dihedral(prev["C"], res["N"], res["CA"], res["C"])
                phis.append(phi)

        # ψ = dihedral(N, CA, C, N_next)
        if i < len(sorted_res) - 1:
            r_next = sorted_res[i + 1]
            nxt = chain_coords.get(r_next, {})
            if "N" in nxt:
                psi = _dihedral(res["N"], res["CA"], res["C"], nxt["N"])
                psis.append(psi)

    def _stats(angles: list[float]) -> tuple[float, float]:
        if not angles:
            return 0.0, 0.0
        a = np.array(angles) / math.pi  # normalise to [-1, 1]
        return float(a.mean()), float(a.std())

    mu_phi, sig_phi = _stats(phis)
    mu_psi, sig_psi = _stats(psis)
    return mu_phi, sig_phi, mu_psi, sig_psi


# ──────────────────────────────────────────────────────────────────────────────
# High-level: compute all chirality features for a complex
# ──────────────────────────────────────────────────────────────────────────────

def compute_chirality_features(
    pdb_path: str | Path,
    chains: list[str],
    cutoff: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute all chirality features for a complex given its PDB file and chain list.

    Returns:
        node_feats : (N, 6) float32 — [Cb_mean, Cb_var, phi_mean, phi_std, psi_mean, psi_std]
        edge_feats : (N, N) float32 — interface_handedness (use as 5th edge dim)
    """
    coords = _parse_pdb_coords(pdb_path)
    N = len(chains)
    node_feats = np.zeros((N, 6), dtype=np.float32)
    edge_feats = np.zeros((N, N), dtype=np.float32)

    chain_coords_list = [coords.get(c, {}) for c in chains]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ci = chain_coords_list[i]
            cj = chain_coords_list[j]
            if not ci or not cj:
                continue
            h = interface_handedness(ci, cj, cutoff)
            edge_feats[i, j] = h

        # Node features: interface residues against ALL other chains
        all_interface_res: list[int] = []
        for j in range(N):
            if i == j:
                continue
            cj = chain_coords_list[j]
            if not cj:
                continue
            res_i, _ = _interface_residues(chain_coords_list[i], cj, cutoff)
            all_interface_res.extend(res_i)
        all_interface_res = list(set(all_interface_res))

        ci = chain_coords_list[i]
        if ci and all_interface_res:
            cb_mean, cb_var = cbeta_orientation(ci, all_interface_res)
            mu_phi, sig_phi, mu_psi, sig_psi = backbone_dihedral_stats(ci, all_interface_res)
            node_feats[i] = [cb_mean, cb_var, mu_phi, sig_phi, mu_psi, sig_psi]

    return node_feats, edge_feats
