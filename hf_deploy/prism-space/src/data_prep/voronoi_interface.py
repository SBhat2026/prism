"""
voronoi_interface.py — Interface contact density edge feature for AssemblyScorer v4.

Inspired by VoroIF-GNN (2026) which uses Voronoi tessellation to assess interface
quality. This module provides a practical CA-CA contact density approximation that
doesn't require voronota:

    contact_density(i, j) = |{(a,b) : a∈chain_i, b∈chain_j, |a-b|_CA ≤ CUTOFF}|
                             / sqrt(N_i * N_j)

This is normalised by sqrt(N_i*N_j) rather than N_i*N_j to reduce bias toward
large chains. Result is clipped to [0, 1] after global normalisation.

Produces a 5th edge feature alongside [ppi, esm_cos, go_cos, sasa_norm].

Usage:
    from src.data_prep.voronoi_interface import compute_contact_density_matrix, CONTACT_CUTOFF
    density_mat = compute_contact_density_matrix(pdb_path, chain_ids)
    # Returns (N, N) float32 array in [0, 1]
"""

import numpy as np
from pathlib import Path
from typing import Optional

CONTACT_CUTOFF = 8.0   # Å CA–CA distance for interface contact
CACHE_SUFFIX   = "_contact_density.npy"


def _parse_chain_ca(pdb_path: str) -> dict[str, np.ndarray]:
    """
    Parse PDB and return {chain_id: (M, 3)} array of CA coordinates per chain.
    """
    chain_coords: dict[str, list] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21]
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            chain_coords.setdefault(chain, []).append([x, y, z])
    return {c: np.array(v, dtype=np.float32) for c, v in chain_coords.items()}


def compute_contact_density_matrix(
    pdb_path: str,
    chain_ids: list[str],
    cutoff: float = CONTACT_CUTOFF,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute N×N CA-contact-density matrix for a complex.

    Each off-diagonal entry (i, j) is the number of inter-chain CA pairs within
    `cutoff` Å, normalised by sqrt(N_i * N_j), then globally normalised to [0, 1].

    Returns (N, N) float32 with diagonal = 0.
    """
    N = len(chain_ids)

    if cache_path and Path(cache_path).exists():
        mat = np.load(cache_path).astype(np.float32)
        if mat.shape == (N, N):
            return mat

    chain_ca = _parse_chain_ca(pdb_path)
    density = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        ci = chain_ids[i]
        if ci not in chain_ca:
            continue
        ca_i = chain_ca[ci]          # (M_i, 3)
        Ni   = len(ca_i)
        for j in range(i + 1, N):
            cj = chain_ids[j]
            if cj not in chain_ca:
                continue
            ca_j = chain_ca[cj]      # (M_j, 3)
            Nj   = len(ca_j)

            # Pairwise distances (M_i, M_j)
            diff = ca_i[:, None, :] - ca_j[None, :, :]   # (M_i, M_j, 3)
            dists = np.linalg.norm(diff, axis=-1)
            n_contacts = int((dists <= cutoff).sum())
            norm = float(np.sqrt(Ni * Nj)) + 1e-8
            val = n_contacts / norm
            density[i, j] = val
            density[j, i] = val

    # Normalise globally to [0, 1] relative to max in this complex
    max_val = density.max()
    if max_val > 0:
        density = density / max_val

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, density)

    return density


def get_contact_density_cached(
    pdb_id: str,
    chain_ids: list[str],
    pdb_path: str,
    cache_dir: Path,
) -> np.ndarray:
    """Convenience wrapper with cache under cache_dir/{pdb_id}_contact_density.npy."""
    cache_path = str(cache_dir / f"{pdb_id}{CACHE_SUFFIX}")
    return compute_contact_density_matrix(pdb_path, chain_ids, cache_path=cache_path)
