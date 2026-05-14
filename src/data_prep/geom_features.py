"""
geom_features.py — Monomer-level geometric node features for AssemblyScorer v4.

Backed by: "Dissecting the Black Box of AlphaFold in Protein-Protein Complex Assembly"
(bioRxiv 2026.04.03.716280) which shows complex assembly is governed by monomer-level
structural geometry, not inter-protein coevolution.

Computes 9 features per chain from AlphaFold/PDB CA coordinates:
  0: radius_of_gyration      — pLDDT-weighted Rg (Å)
  1: sphericity               — 1 - (I_max - I_min) / (I_max + I_min); 1=sphere
  2: asphericity              — Ω = I_z - 0.5*(I_x + I_y); 0=sphere
  3: prolate_param            — b/a ratio of principal axes; 1=sphere
  4: principal_axis_ratio_1   — a/c (long/short)
  5: principal_axis_ratio_2   — b/c (mid/short)
  6: residue_count_norm       — log10(N_res) / log10(2000)  normalised to ~[0,1]
  7: mean_pLDDT_norm          — mean pLDDT / 100
  8: pLDDT_std_norm           — std of pLDDT / 50

If coordinates aren't available, returns zeros (safe default).

Usage:
    from src.data_prep.geom_features import compute_geom_features, GEOM_DIM
    node_feats = np.hstack([esm_emb, plddt_feats, sasa_diag, geom_feats])
"""

import numpy as np
import os
import json
import requests
import time
from pathlib import Path
from typing import Optional

GEOM_DIM = 9


def _fetch_af_pdb(uniprot_id: str, cache_dir: Path, timeout: int = 20) -> Optional[str]:
    """Download AlphaFold PDB (CA only sufficient) and return local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cache_dir / f"{uniprot_id}_AF.pdb"
    if pdb_path.exists():
        return str(pdb_path)
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            pdb_path.write_bytes(r.content)
            return str(pdb_path)
    except Exception:
        pass
    return None


def _parse_ca_plddt(pdb_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse ATOM CA lines from PDB.
    Returns (coords, plddt_per_residue) where coords is (N,3) and plddt is (N,).
    pLDDT is stored in the B-factor column in AlphaFold PDBs.
    """
    coords = []
    plddts = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    b = float(line[60:66])
                    coords.append([x, y, z])
                    plddts.append(b)
                except ValueError:
                    continue
    if not coords:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(coords, dtype=np.float32), np.array(plddts, dtype=np.float32)


def _inertia_tensor_eigenvalues(coords: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute principal moments of inertia (sorted ascending) from (N,3) coords."""
    if weights is None:
        weights = np.ones(len(coords))
    weights = weights / weights.sum()
    com = (coords * weights[:, None]).sum(0)
    r = coords - com
    # Inertia tensor
    I = np.zeros((3, 3))
    for k, w in zip(r, weights):
        I[0, 0] += w * (k[1]**2 + k[2]**2)
        I[1, 1] += w * (k[0]**2 + k[2]**2)
        I[2, 2] += w * (k[0]**2 + k[1]**2)
        I[0, 1] -= w * k[0] * k[1]
        I[0, 2] -= w * k[0] * k[2]
        I[1, 2] -= w * k[1] * k[2]
    I[1, 0] = I[0, 1]; I[2, 0] = I[0, 2]; I[2, 1] = I[1, 2]
    evals = np.linalg.eigvalsh(I)
    return np.sort(evals)  # ascending: Ix ≤ Iy ≤ Iz


def _radius_of_gyration(coords: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if weights is None:
        weights = np.ones(len(coords))
    weights = weights / weights.sum()
    com = (coords * weights[:, None]).sum(0)
    r2 = ((coords - com)**2).sum(1)
    return float(np.sqrt((r2 * weights).sum()))


def compute_geom_features(
    coords: np.ndarray,       # (N, 3) CA coordinates
    plddt: np.ndarray,        # (N,) pLDDT per residue
) -> np.ndarray:
    """Compute GEOM_DIM=9 geometric features from CA coords + pLDDT."""
    feats = np.zeros(GEOM_DIM, dtype=np.float32)
    N = len(coords)
    if N == 0:
        return feats

    w = np.clip(plddt / 100.0, 0.01, 1.0)

    # 0: Rg
    feats[0] = min(_radius_of_gyration(coords, w) / 50.0, 5.0)  # normalise; 50Å ≈ mid-size

    # 1-5: shape from inertia tensor
    if N >= 4:
        evals = _inertia_tensor_eigenvalues(coords, w)
        Ix, Iy, Iz = evals
        total = Ix + Iy + Iz + 1e-8
        # sphericity: 1 = perfect sphere
        feats[1] = float(1.0 - (Iz - Ix) / total)
        # asphericity (relative anisotropy)
        feats[2] = float((Iz - 0.5 * (Ix + Iy)) / (total + 1e-8))
        # prolate/oblate param: b/a where a,b,c are semi-axes
        # semi-axes: a = sqrt(5/(2m) * (Iy+Iz-Ix))  etc.
        a2 = (Iy + Iz - Ix) / 2 + 1e-8
        b2 = (Ix + Iz - Iy) / 2 + 1e-8
        c2 = (Ix + Iy - Iz) / 2 + 1e-8
        a2, b2, c2 = max(a2, 1e-8), max(b2, 1e-8), max(c2, 1e-8)
        a, b, c = a2**0.5, b2**0.5, c2**0.5
        axes = sorted([a, b, c])         # short, mid, long
        cs, cm, cl = axes
        feats[3] = float(min(cm / (cl + 1e-8), 1.0))   # prolate param b/a
        feats[4] = float(min(cl / (cs + 1e-8) / 10.0, 1.0))  # a/c ratio (long/short), capped at 10
        feats[5] = float(min(cm / (cs + 1e-8) / 10.0, 1.0))  # b/c ratio (mid/short), capped at 10

    # 6: residue count (log-normalised)
    feats[6] = float(min(np.log10(N + 1) / np.log10(2001), 1.0))

    # 7-8: pLDDT summary
    feats[7] = float(np.mean(plddt) / 100.0)
    feats[8] = float(min(np.std(plddt) / 50.0, 1.0))

    return feats


def load_geom_for_chain(
    uniprot_id: str,
    cache_dir: Path,
    sequence_len: int = 100,
) -> np.ndarray:
    """
    Fetch AlphaFold structure for a UniProt ID and return GEOM_DIM features.
    Falls back to a sequence-length-based stub if structure unavailable.
    Results are cached to <cache_dir>/<uniprot_id>_geom.npy.
    """
    geom_cache = cache_dir / f"{uniprot_id}_geom.npy"
    if geom_cache.exists():
        return np.load(str(geom_cache))

    if uniprot_id:
        pdb_path = _fetch_af_pdb(uniprot_id, cache_dir / "af_pdbs")
        if pdb_path:
            coords, plddt = _parse_ca_plddt(pdb_path)
            if len(coords) > 0:
                feats = compute_geom_features(coords, plddt)
                np.save(str(geom_cache), feats)
                return feats
        time.sleep(0.3)

    # Stub: use sequence length proxy
    feats = np.zeros(GEOM_DIM, dtype=np.float32)
    feats[0] = float(min(np.sqrt(sequence_len) / 10.0, 5.0))   # Rg proxy
    feats[6] = float(min(np.log10(sequence_len + 1) / np.log10(2001), 1.0))
    feats[7] = 0.70  # default pLDDT assumption
    np.save(str(geom_cache), feats)
    return feats


def build_geom_matrix(
    uniprot_ids: list[str],
    cache_dir: Path,
    sequences: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Return (N, GEOM_DIM) feature matrix for all chains in a complex.
    """
    rows = []
    for i, uid in enumerate(uniprot_ids):
        seq_len = len(sequences[i]) if sequences and i < len(sequences) else 100
        rows.append(load_geom_for_chain(uid, cache_dir, seq_len))
    return np.stack(rows).astype(np.float32)
