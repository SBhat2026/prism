"""
Build expanded training dataset from PDB.

Pipeline:
  1. Query RCSB for hetero-oligomers (≥2 unique protein entities, 4–14 chains,
     resolution ≤ 2.0 Å, X-ray only)
  2. Download PDB files
  3. Extract per-chain sequences from ATOM records
  4. Compute pairwise BSA matrix via freesasa
  5. Derive GT assembly order (SASA-greedy, same heuristic as Marsh 2013)
  6. Compute ESM-150M embeddings for each unique chain sequence
  7. Fetch AlphaFold pLDDT stats for each chain (via UniProt ID from RCSB)
  8. Optionally query STRING PPI
  9. Save to data/training_complexes/

Usage:
    python src/data_prep/build_pdb_training_set.py --n_complexes 200

Output:
    data/training_complexes/index.json           — master index of all complexes
    data/training_complexes/{pdb_id}/sasa.npy    — (N,N) BSA matrix
    data/training_complexes/{pdb_id}/emb.npy     — (N, 640) ESM embeddings
    data/training_complexes/{pdb_id}/meta.json   — chains, gt_order, type, pLDDT
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_prep.fast_sasa import compute_bsa_matrix, derive_gt_order_sasa_greedy
from src.data_prep.plddt_fetcher import (
    get_chain_uniprot_ids, fetch_plddt_stats, stats_to_node_vec
)

OUT_DIR     = ROOT / "data" / "training_complexes"
PDB_DIR     = OUT_DIR / "pdb"
PLDDT_CACHE = ROOT / "data" / "plddt_cache"
STRING_CACHE= ROOT / "data" / "string"

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILE   = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Exclude Marsh 2013 benchmark PDBs (kept as held-out eval set)
MARSH_EXCLUDE = {
    "2HHB","1BRS","1AY7","1TGS","2PTC","1A2K","1SBB","3GBN","1AON","1GRU"
}

# ──────────────────────────────────────────────────────────────────────────────
# RCSB query
# ──────────────────────────────────────────────────────────────────────────────

def query_rcsb_hetero_oligomers(
    n_chains_min: int = 2,
    n_chains_max: int = 14,
    resolution_max: float = 2.5,
    n_results: int = 500,
) -> list[str]:
    """
    Query RCSB for hetero-oligomeric protein complexes.

    Filters:
      - ≥2 unique protein polymer entities (heteromer)
      - n_chains_min ≤ deposited chains ≤ n_chains_max
      - resolution ≤ resolution_max Å
      - X-ray crystallography only
      - No nucleic acid chains
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "negation": False,
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "greater_or_equal",
                        "negation": False,
                        "value": n_chains_min,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "less_or_equal",
                        "negation": False,
                        "value": n_chains_max,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "negation": False,
                        "value": resolution_max,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "negation": False,
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_nucleic_acid",
                        "operator": "equals",
                        "negation": False,
                        "value": 0,
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": 0, "rows": n_results},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
        "return_type": "entry",
    }

    try:
        r = requests.post(RCSB_SEARCH, json=query, timeout=30)
        r.raise_for_status()
        hits = r.json().get("result_set", [])
        pdb_ids = [h["identifier"].upper() for h in hits]
        return [p for p in pdb_ids if p not in MARSH_EXCLUDE]
    except Exception as e:
        print(f"[rcsb] query failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# PDB download + chain extraction
# ──────────────────────────────────────────────────────────────────────────────

_AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def download_pdb(pdb_id: str) -> Optional[Path]:
    """Download PDB file to PDB_DIR/{pdb_id}.pdb. Returns path or None on error."""
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    out = PDB_DIR / f"{pdb_id}.pdb"
    if out.exists():
        return out
    try:
        r = requests.get(RCSB_FILE.format(pdb_id=pdb_id), timeout=30)
        r.raise_for_status()
        out.write_text(r.text)
        time.sleep(0.3)
        return out
    except Exception as e:
        print(f"  [dl] {pdb_id} failed: {e}")
        return None


def extract_chain_sequences(pdb_path: Path) -> dict[str, str]:
    """Extract per-chain amino acid sequences from ATOM CA records."""
    chains: dict[str, list[str]] = {}
    seen: dict[str, set] = {}
    with open(pdb_path) as f:
        for line in f:
            if line[:4] != "ATOM" or line[13:15].strip() != "CA":
                continue
            ch = line[21]
            resnum = line[22:26].strip()
            resname = line[17:20].strip()
            key = (resnum, resname)
            if ch not in seen:
                seen[ch] = set()
                chains[ch] = []
            if key not in seen[ch]:
                seen[ch].add(key)
                chains[ch].append(_AA3TO1.get(resname, "X"))
    return {c: "".join(s) for c, s in chains.items()}


def filter_protein_chains(
    seq_map: dict[str, str],
    min_len: int = 30,
) -> dict[str, str]:
    """Keep only chains with ≥min_len amino acids (discard small linkers/tags)."""
    return {c: s for c, s in seq_map.items() if len(s) >= min_len}


def has_unique_sequences(seq_map: dict[str, str], min_unique: int = 2) -> bool:
    """Check that the complex has ≥min_unique distinct sequences (is a heteromer)."""
    unique_seqs = set(seq_map.values())
    return len(unique_seqs) >= min_unique


# ──────────────────────────────────────────────────────────────────────────────
# ESM-150M embeddings
# ──────────────────────────────────────────────────────────────────────────────

_ESM_MODEL = None
_ESM_BCONV = None
_ESM_LAYER = 30  # t30 = layer 30


def _load_esm():
    global _ESM_MODEL, _ESM_BCONV
    if _ESM_MODEL is not None:
        return
    import torch
    import esm as esm_lib
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    _ESM_MODEL = model.to(device).eval()
    _ESM_BCONV = alphabet.get_batch_converter()
    print(f"[esm] loaded esm2_t30_150M_UR50D on {device}")


def embed_sequences(sequences: dict[str, str], batch_size: int = 4) -> dict[str, np.ndarray]:
    """
    Compute ESM-150M mean-pool embeddings.
    Returns {name: (640,) float32 array}.
    """
    import torch
    _load_esm()
    device = next(_ESM_MODEL.parameters()).device
    items = list(sequences.items())
    embeddings: dict[str, np.ndarray] = {}

    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        _, _, tokens = _ESM_BCONV([(n, s) for n, s in batch])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = _ESM_MODEL(tokens, repr_layers=[_ESM_LAYER], return_contacts=False)
        reps = out["representations"][_ESM_LAYER]
        for j, (name, seq) in enumerate(batch):
            emb = reps[j, 1: len(seq) + 1].mean(0).cpu().numpy()
            embeddings[name] = emb.astype(np.float32)

    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Per-complex processing
# ──────────────────────────────────────────────────────────────────────────────

def process_complex(pdb_id: str, fetch_string: bool = False,
                    min_chains: int = 2) -> Optional[dict]:
    """
    Full pipeline for one PDB entry.

    Returns metadata dict ready to append to index.json, or None on failure.
    """
    print(f"\n  [{pdb_id}] downloading...")
    pdb_path = download_pdb(pdb_id)
    if pdb_path is None:
        return None

    # Chain sequences
    seq_map = filter_protein_chains(extract_chain_sequences(pdb_path))
    if len(seq_map) < min_chains:
        print(f"  [{pdb_id}] skip: {len(seq_map)} chains < min {min_chains}")
        return None
    if not has_unique_sequences(seq_map):
        print(f"  [{pdb_id}] skip: homomer")
        return None

    chain_ids = sorted(seq_map.keys())
    N = len(chain_ids)
    sequences = [seq_map[c] for c in chain_ids]

    # Output directory for this complex
    cplx_dir = OUT_DIR / pdb_id
    cplx_dir.mkdir(parents=True, exist_ok=True)

    # BSA matrix
    sasa_cache = str(cplx_dir / "sasa.npy")
    print(f"  [{pdb_id}] computing SASA ({N} chains)...")
    bsa_mat = compute_bsa_matrix(str(pdb_path), chain_ids, cache_path=sasa_cache)

    if bsa_mat.max() == 0:
        print(f"  [{pdb_id}] skip: zero BSA (chains may not contact)")
        return None

    # GT order from SASA-greedy
    gt_order = derive_gt_order_sasa_greedy(bsa_mat)

    # Detect complex type (homomer if all sequences identical)
    unique_seqs = set(sequences)
    complex_type = "homomer" if len(unique_seqs) == 1 else "heteromer"

    # ESM embeddings
    print(f"  [{pdb_id}] computing ESM-150M embeddings...")
    seq_dict = {f"{pdb_id}_{c}": s for c, s in zip(chain_ids, sequences)}
    try:
        emb_dict = embed_sequences(seq_dict)
        emb_mat = np.stack([emb_dict[f"{pdb_id}_{c}"] for c in chain_ids])
    except Exception as e:
        print(f"  [{pdb_id}] ESM failed: {e}")
        return None
    np.save(str(cplx_dir / "emb.npy"), emb_mat)

    # pLDDT via AlphaFold
    print(f"  [{pdb_id}] fetching pLDDT from AlphaFold...")
    uniprot_map = get_chain_uniprot_ids(pdb_id)
    plddt_vecs = []
    uniprot_ids = []
    for c in chain_ids:
        uid = uniprot_map.get(c, "")
        uniprot_ids.append(uid)
        if uid:
            stats = fetch_plddt_stats(uid, PLDDT_CACHE)
            vec = stats_to_node_vec(stats)
        else:
            vec = stats_to_node_vec(None)  # defaults
        plddt_vecs.append(vec)
    plddt_mat = np.stack(plddt_vecs)  # (N, 5)
    np.save(str(cplx_dir / "plddt.npy"), plddt_mat)

    # STRING PPI
    ppi_mat = np.zeros((N, N), dtype=np.float32)
    if fetch_string and any(uniprot_ids):
        from src.data_prep.string_api import fetch_ppi_matrix
        print(f"  [{pdb_id}] querying STRING PPI...")
        ppi_mat = fetch_ppi_matrix(
            uniprot_ids, chain_ids, cache_dir=STRING_CACHE
        )
    np.save(str(cplx_dir / "ppi.npy"), ppi_mat)

    # ESM cosine matrix
    norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    emb_n = emb_mat / (norms + 1e-8)
    esm_cos_mat = (emb_n @ emb_n.T).clip(-1, 1).astype(np.float32)

    # Save metadata
    meta = {
        "pdb_id": pdb_id,
        "chain_ids": chain_ids,
        "sequences": sequences,
        "gt_order": gt_order,
        "complex_type": complex_type,
        "n_chains": N,
        "uniprot_ids": uniprot_ids,
        "bsa_max": float(bsa_mat.max()),
        "gt_source": "sasa_greedy",
    }
    (cplx_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"  [{pdb_id}] done. type={complex_type} N={N} bsa_max={bsa_mat.max():.1f}")
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_complexes", type=int, default=600,
                        help="Max number of PDB entries to query")
    parser.add_argument("--n_target", type=int, default=200,
                        help="Target number of successfully processed complexes")
    parser.add_argument("--min_chains", type=int, default=2)
    parser.add_argument("--max_chains", type=int, default=14)
    parser.add_argument("--resolution", type=float, default=2.5)
    parser.add_argument("--string", action="store_true",
                        help="Query STRING PPI (slow; ~1s per complex)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed complexes")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "index.json"

    # Load existing index if resuming
    index: list[dict] = []
    done_ids: set[str] = set()
    if args.resume and index_path.exists():
        index = json.loads(index_path.read_text())
        done_ids = {m["pdb_id"] for m in index}
        print(f"[resume] {len(done_ids)} already processed")

    print(f"[rcsb] querying for {args.n_complexes} hetero-oligomers "
          f"({args.min_chains}–{args.max_chains} chains, ≤{args.resolution}Å)...")
    pdb_ids = query_rcsb_hetero_oligomers(
        n_chains_min=args.min_chains,
        n_chains_max=args.max_chains,
        resolution_max=args.resolution,
        n_results=args.n_complexes,
    )
    print(f"[rcsb] {len(pdb_ids)} candidates (excluding Marsh 2013)")

    processed = 0
    for pdb_id in pdb_ids:
        if processed >= args.n_target:
            break
        if pdb_id in done_ids:
            continue
        try:
            meta = process_complex(pdb_id, fetch_string=args.string,
                                   min_chains=args.min_chains)
            if meta is not None:
                index.append(meta)
                done_ids.add(pdb_id)
                processed += 1
                index_path.write_text(json.dumps(index, indent=2))
        except Exception as e:
            print(f"  [{pdb_id}] error: {e}")
            continue

    print(f"\n[done] {processed} complexes built → {OUT_DIR}")
    print(f"[done] index: {index_path} ({len(index)} total)")


if __name__ == "__main__":
    main()
