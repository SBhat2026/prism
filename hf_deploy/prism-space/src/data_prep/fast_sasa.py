"""
Fast SASA-based buried surface area (BSA) computation using freesasa.

For each chain pair (i, j):
    BSA(i,j) = (SASA_i_isolated + SASA_j_isolated - SASA_ij_complex) / 2

Scales as O(N^2) substructure evaluations. Suitable for complexes up to ~20 chains.
Caches results per PDB to avoid recomputation.
"""

import os
import json
import tempfile
import numpy as np
import freesasa
from pathlib import Path
from typing import Optional


def _write_chain_subset_pdb(src_pdb: str, chain_set: set[str], dst: str):
    """Write a subset PDB containing only the specified chain IDs."""
    with open(src_pdb) as fin, open(dst, "w") as fout:
        for line in fin:
            rec = line[:4]
            if rec in ("ATOM", "HETA"):
                if line[21] in chain_set:
                    fout.write(line)
            elif rec in ("END ", "TER "):
                fout.write(line)


def _total_sasa(pdb_path: str) -> float:
    """Total solvent-accessible surface area of a PDB (Å²)."""
    try:
        struct = freesasa.Structure(pdb_path)
        result = freesasa.calc(struct)
        return result.totalArea()
    except Exception:
        return 0.0


def compute_bsa_matrix(
    pdb_path: str,
    chain_ids: list[str],
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute N×N buried surface area matrix (Å²) for a complex.

    Args:
        pdb_path:   path to PDB file containing all chains
        chain_ids:  ordered list of chain IDs to include
        cache_path: if given, load/save result to this .npy path

    Returns:
        (N, N) float32 array, symmetric, diagonal = 0
    """
    N = len(chain_ids)

    if cache_path and Path(cache_path).exists():
        mat = np.load(cache_path).astype(np.float32)
        if mat.shape == (N, N):
            return mat

    bsa_mat = np.zeros((N, N), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Isolated SASA per chain
        isolated = []
        for cid in chain_ids:
            tmp = os.path.join(tmpdir, f"chain_{cid}.pdb")
            _write_chain_subset_pdb(pdb_path, {cid}, tmp)
            isolated.append(_total_sasa(tmp))

        # Pairwise BSA
        for i in range(N):
            for j in range(i + 1, N):
                tmp_pair = os.path.join(tmpdir, f"pair_{i}_{j}.pdb")
                _write_chain_subset_pdb(
                    pdb_path, {chain_ids[i], chain_ids[j]}, tmp_pair
                )
                pair_sasa = _total_sasa(tmp_pair)
                bsa = max(0.0, (isolated[i] + isolated[j] - pair_sasa) / 2.0)
                bsa_mat[i, j] = bsa
                bsa_mat[j, i] = bsa

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, bsa_mat)

    return bsa_mat


def derive_gt_order_sasa_greedy(bsa_mat: np.ndarray) -> list[int]:
    """
    Derive assembly order via greedy BSA maximization (Marsh 2013 heuristic).

    Seed: chain with highest total BSA to all others (most interfacially connected).
    Each subsequent step: add the remaining chain with highest BSA to assembled set.

    Returns list of chain indices (0-indexed), first-to-last assembly order.
    """
    N = bsa_mat.shape[0]
    row_sums = bsa_mat.sum(axis=1)
    seed = int(np.argmax(row_sums))

    order = [seed]
    remaining = list(range(N))
    remaining.remove(seed)

    while remaining:
        scores = []
        for cand in remaining:
            # Max BSA from candidate to any already-assembled chain
            bsa_to_complex = bsa_mat[cand, order].max()
            scores.append(bsa_to_complex)
        best_local = int(np.argmax(scores))
        chosen = remaining[best_local]
        order.append(chosen)
        remaining.remove(chosen)

    return order
