"""
STRING PPI score fetcher.

Queries the STRING REST API for pairwise interaction scores between protein
UniProt accessions. Results cached locally.

Returns combined_score ∈ [0, 1] (STRING raw score / 1000).
"""

import json
import time
import requests
import numpy as np
from itertools import combinations
from pathlib import Path
from typing import Optional

STRING_NETWORK_URL = "https://string-db.org/api/json/network"
STRING_IDS_URL     = "https://string-db.org/api/json/get_string_ids"
CALLER_ID          = "protein_assembly_sim"


def normalise_string_score(raw) -> float:
    """STRING scores → [0, 1].

    `/api/json/network` returns `score` ALREADY normalised to [0, 1] (e.g.
    0.999). Only the *request* parameter `required_score` uses the 0-1000
    integer scale. An earlier version of this module divided the response by
    1000 as well, which crushed every PPI value to ~1e-3 — three orders of
    magnitude below the SASA channel — so the PPI edge channel became
    numerically invisible to the network. See scripts/repair_ppi_cache.py for
    the in-place repair of caches written before this fix.

    Both scales are tolerated here so a future API change cannot silently
    reintroduce the bug.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if v > 1.0:            # 0-1000 integer scale
        v /= 1000.0
    return float(min(max(v, 0.0), 1.0))


def _map_uniprot_to_string(
    uniprot_ids: list[str],
    species: int,
    sleep: float = 0.5,
) -> dict[str, str]:
    """Map UniProt IDs → STRING IDs for the given species."""
    if not uniprot_ids:
        return {}
    params = {
        "identifiers":     "\r".join(uniprot_ids),
        "species":         species,
        "limit":           1,
        "caller_identity": CALLER_ID,
    }
    try:
        r = requests.post(STRING_IDS_URL, data=params, timeout=30)
        time.sleep(sleep)
        if r.status_code != 200:
            return {}
        mapping = {}
        for item in r.json():
            uid = item.get("queryItem", "").strip()
            sid = item.get("stringId", "").strip()
            if uid and sid:
                mapping[uid] = sid
        return mapping
    except Exception:
        return {}


def fetch_ppi_matrix(
    uniprot_ids: list[str],
    chain_ids: list[str],
    species: int = 9606,
    cache_dir: Optional[Path] = None,
    min_score: float = 0.0,
    sleep: float = 0.5,
    per_chain_taxids: "list[int] | None" = None,
) -> np.ndarray:
    """
    Build N×N PPI score matrix for the given UniProt IDs.

    Args:
        uniprot_ids:      UniProt accession per chain (parallel to chain_ids)
        chain_ids:        chain labels (used for indexing only)
        species:          NCBI taxon ID (default 9606 = human); used when
                          per_chain_taxids is None
        cache_dir:        cache directory; cached per frozenset of UniProt IDs
        min_score:        filter out scores below this threshold (set to 0 to keep all)
        sleep:            seconds between API calls
        per_chain_taxids: per-chain taxon IDs for cross-species complexes.
                          When provided, each pair (i,j) is queried with
                          taxid[i] first, then taxid[j]; the max score is kept.
                          Missing pairs default to 0.0 (not 0.5).

    Returns:
        (N, N) float32 array, symmetric, values ∈ [0, 1]
    """
    N = len(uniprot_ids)
    mat = np.zeros((N, N), dtype=np.float32)

    # Cross-species path: delegate pairwise to per-taxid queries
    if per_chain_taxids is not None:
        if len(per_chain_taxids) != N:
            raise ValueError("per_chain_taxids must have one entry per chain")
        for i in range(N):
            for j in range(i + 1, N):
                best = 0.0
                for taxid in {per_chain_taxids[i], per_chain_taxids[j]}:
                    sub_mat = fetch_ppi_matrix(
                        [uniprot_ids[i], uniprot_ids[j]],
                        [chain_ids[i], chain_ids[j]],
                        species=taxid,
                        cache_dir=cache_dir,
                        min_score=min_score,
                        sleep=sleep,
                        per_chain_taxids=None,
                    )
                    best = max(best, float(sub_mat[0, 1]))
                mat[i, j] = best
                mat[j, i] = best
        return mat

    # Deduplicate UniProt IDs
    unique_ids = list(dict.fromkeys(uid for uid in uniprot_ids if uid))
    if len(unique_ids) < 2:
        return mat

    # Cache key
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = "_".join(sorted(unique_ids)) + f"_sp{species}"
        cache_file = cache_dir / f"{key[:120]}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            pair_scores = {tuple(k.split(":::")): v for k, v in cached.items()}
            _fill_matrix(mat, uniprot_ids, pair_scores)
            return mat

    # Map to STRING IDs
    str_map = _map_uniprot_to_string(unique_ids, species, sleep)
    if not str_map:
        return mat

    string_ids = [str_map[uid] for uid in unique_ids if uid in str_map]
    if len(string_ids) < 2:
        return mat

    # Query network
    params = {
        "identifiers":     "\r".join(string_ids),
        "species":         species,
        "required_score":  int(min_score * 1000),
        "caller_identity": CALLER_ID,
    }
    try:
        r = requests.post(STRING_NETWORK_URL, data=params, timeout=60)
        time.sleep(sleep)
        if r.status_code != 200:
            return mat

        pair_scores: dict[tuple, float] = {}
        for item in r.json():
            a = item.get("preferredName_A", "")
            b = item.get("preferredName_B", "")
            score = normalise_string_score(item.get("score", 0))

            # Map back to UniProt; STRING preferred names may not equal UniProt
            # so also store by STRING ID
            sid_a = item.get("stringId_A", "")
            sid_b = item.get("stringId_B", "")

            # Find UniProt IDs for these STRING IDs
            rev_map = {v: k for k, v in str_map.items()}
            uid_a = rev_map.get(sid_a, sid_a)
            uid_b = rev_map.get(sid_b, sid_b)
            pair_scores[(uid_a, uid_b)] = score
            pair_scores[(uid_b, uid_a)] = score

        if cache_file is not None:
            serialisable = {":::".join(k): v for k, v in pair_scores.items()}
            cache_file.write_text(json.dumps(serialisable, indent=2))

        _fill_matrix(mat, uniprot_ids, pair_scores)

    except Exception:
        pass

    return mat


def _fill_matrix(
    mat: np.ndarray,
    uniprot_ids: list[str],
    pair_scores: dict,
):
    N = len(uniprot_ids)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            score = pair_scores.get((uniprot_ids[i], uniprot_ids[j]), 0.0)
            mat[i, j] = max(mat[i, j], float(score))


def fetch_ppi_matrix_multispecies(
    uniprot_ids: list[str],
    chain_ids: list[str],
    taxids: list[int],  # per-chain taxid
    cache_dir=None,
    sleep: float = 0.5,
) -> np.ndarray:
    """Multi-species PPI: group chains by taxid, query each group separately, merge.

    Chains within the same organism are queried together via STRING for that taxid.
    Pairs from different organisms get score 0.0 (no STRING data across organisms).
    """
    N = len(uniprot_ids)
    mat = np.zeros((N, N), dtype=np.float32)

    if N < 2:
        return mat

    # Group chain indices by taxid
    taxid_to_indices: dict[int, list[int]] = {}
    for idx, tid in enumerate(taxids):
        taxid_to_indices.setdefault(tid, []).append(idx)

    # Query STRING for each intra-species group and fill the sub-matrix
    for tid, indices in taxid_to_indices.items():
        if len(indices) < 2:
            continue
        sub_uids   = [uniprot_ids[i] for i in indices]
        sub_chains = [chain_ids[i]   for i in indices]
        sub_mat = fetch_ppi_matrix(
            sub_uids,
            sub_chains,
            species=tid,
            cache_dir=cache_dir,
            sleep=sleep,
            per_chain_taxids=None,
        )
        # Scatter sub_mat back into the full N×N matrix
        for si, gi in enumerate(indices):
            for sj, gj in enumerate(indices):
                if si != sj:
                    mat[gi, gj] = sub_mat[si, sj]

    # Cross-species pairs remain 0.0 — no STRING data across organisms
    return mat
