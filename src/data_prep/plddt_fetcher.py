"""
AlphaFold per-chain pLDDT feature fetcher.

Downloads AlphaFold PDB from EBI (pLDDT stored in B-factor column) and extracts
per-residue pLDDT statistics for a given UniProt accession.

Output stats (7 scalar features per chain):
    plddt_mean, plddt_std, plddt_q10, plddt_q50, plddt_q90,
    plddt_frac_gt90, plddt_frac_lt50

Use for node feature augmentation in GNN training.
"""

import json
import time
import numpy as np
import requests
from pathlib import Path
from typing import Optional

AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"
AF_URL_FALLBACK = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
RCSB_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
RCSB_ENTRY_URL  = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

# 5 features used as node features in GNN (subset of 7 stats)
PLDDT_NODE_DIM = 5
PLDDT_NODE_KEYS = ["plddt_mean", "plddt_frac_gt90", "plddt_frac_lt50",
                   "plddt_q10", "plddt_q90"]


def fetch_plddt_stats(
    uniprot_id: str,
    cache_dir: Path,
    sleep: float = 0.3,
) -> Optional[dict]:
    """
    Fetch pLDDT statistics for a UniProt accession from the AlphaFold DB.

    Returns dict with keys in PLDDT_NODE_KEYS (+ plddt_std, plddt_q50),
    or None if unavailable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{uniprot_id}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    # Resolve the latest AF model URL via the EBI API
    try:
        api_r = requests.get(AF_API.format(uid=uniprot_id), timeout=15)
        time.sleep(sleep / 2)
        if api_r.status_code == 200 and api_r.json():
            url = api_r.json()[0]["pdbUrl"]
        else:
            url = AF_URL_FALLBACK.format(uid=uniprot_id)
    except Exception:
        url = AF_URL_FALLBACK.format(uid=uniprot_id)

    try:
        r = requests.get(url, timeout=30)
        time.sleep(sleep)
        if r.status_code != 200:
            return None

        plddt = []
        for line in r.text.splitlines():
            if line[:4] == "ATOM" and line[13:15].strip() == "CA":
                try:
                    plddt.append(float(line[60:66].strip()))
                except ValueError:
                    pass

        if not plddt:
            return None

        arr = np.array(plddt, dtype=np.float32)
        stats = {
            "plddt_mean":       float(arr.mean()),
            "plddt_std":        float(arr.std()),
            "plddt_q10":        float(np.percentile(arr, 10)),
            "plddt_q50":        float(np.percentile(arr, 50)),
            "plddt_q90":        float(np.percentile(arr, 90)),
            "plddt_frac_gt90":  float((arr > 90).mean()),
            "plddt_frac_lt50":  float((arr < 50).mean()),
            "n_residues":       int(len(arr)),
        }
        cache_file.write_text(json.dumps(stats, indent=2))
        return stats

    except Exception:
        return None


def stats_to_node_vec(stats: Optional[dict]) -> np.ndarray:
    """Convert pLDDT stats dict → (5,) float32 node feature vector."""
    if stats is None:
        # Neutral defaults: moderate confidence
        return np.array([70.0, 0.3, 0.1, 50.0, 90.0], dtype=np.float32)
    return np.array(
        [stats.get(k, 0.0) for k in PLDDT_NODE_KEYS],
        dtype=np.float32,
    )


def get_chain_uniprot_ids(pdb_id: str, sleep: float = 0.2) -> dict[str, str]:
    """
    Query RCSB data API to get {auth_chain_id: uniprot_id} for all protein chains.
    Returns empty dict on failure.
    """
    pdb_id = pdb_id.upper()
    try:
        r = requests.get(RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=20)
        time.sleep(sleep)
        if r.status_code != 200:
            return {}
        entry = r.json()
        entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get(
            "polymer_entity_ids", []
        )
    except Exception:
        return {}

    chain_uniprot: dict[str, str] = {}

    for eid in entity_ids:
        try:
            r2 = requests.get(
                RCSB_ENTITY_URL.format(pdb_id=pdb_id, entity_id=eid), timeout=20
            )
            time.sleep(sleep)
            if r2.status_code != 200:
                continue
            entity = r2.json()

            # UniProt accession(s) for this entity
            uniprot_ids = []
            for ref in entity.get("rcsb_polymer_entity_container_identifiers", {}).get(
                "uniprot_ids", []
            ):
                uniprot_ids.append(ref)

            if not uniprot_ids:
                continue
            uid = uniprot_ids[0]

            # Auth chain IDs for this entity
            auth_chains = entity.get(
                "rcsb_polymer_entity_container_identifiers", {}
            ).get("auth_asym_ids", [])
            for c in auth_chains:
                chain_uniprot[c] = uid

        except Exception:
            continue

    return chain_uniprot
