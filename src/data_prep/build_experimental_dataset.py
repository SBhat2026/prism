"""
Build dataset from experimental_gt_v2.json complexes.

Downloads PDB files, extracts chain sequences, computes BSA matrix (freesasa),
ESM-150M embeddings, AlphaFold pLDDT, and STRING PPI.

Saves to data/experimental_complexes/{pdb_id}/ with same layout as training_complexes.

Usage:
    python src/data_prep/build_experimental_dataset.py [--string]
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

from src.data_prep.fast_sasa import compute_bsa_matrix
from src.data_prep.plddt_fetcher import (
    get_chain_uniprot_ids, fetch_plddt_stats, stats_to_node_vec, PLDDT_NODE_DIM
)

GT_V2_PATH  = ROOT / "data" / "experimental_gt_v2.json"
OUT_DIR     = ROOT / "data" / "experimental_complexes"
PDB_DIR     = OUT_DIR / "pdb"
PLDDT_CACHE = ROOT / "data" / "plddt_cache"
STRING_CACHE= ROOT / "data" / "string"

RCSB_FILE   = "https://files.rcsb.org/download/{pdb_id}.pdb"

_AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

# ── ESM model (singleton) ────────────────────────────────────────────────────

_ESM_MODEL = None
_ESM_BCONV = None
_ESM_LAYER = 30
MAX_SEQ_LEN = 1022  # ESM-2 hard limit (1024 - BOS/EOS)


def _load_esm():
    global _ESM_MODEL, _ESM_BCONV
    if _ESM_MODEL is not None:
        return
    try:
        import torch
        import esm as esm_lib
    except ModuleNotFoundError:
        print("[error] 'esm' or 'torch' not found in this Python environment.")
        print("[error] Run with: /Users/siddhantbhat/miniforge3/bin/python3 src/data_prep/build_experimental_dataset.py --resume")
        raise
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    _ESM_MODEL = model.to(device).eval()
    _ESM_BCONV = alphabet.get_batch_converter()
    print(f"[esm] loaded esm2_t30_150M_UR50D on {device}")


def embed_sequences(sequences: dict[str, str]) -> dict[str, np.ndarray]:
    import torch
    _load_esm()
    device = next(_ESM_MODEL.parameters()).device
    items = [(n, s[:MAX_SEQ_LEN]) for n, s in sequences.items()]
    embeddings: dict[str, np.ndarray] = {}
    for i in range(0, len(items), 4):
        batch = items[i: i + 4]
        _, _, tokens = _ESM_BCONV([(n, s) for n, s in batch])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = _ESM_MODEL(tokens, repr_layers=[_ESM_LAYER], return_contacts=False)
        reps = out["representations"][_ESM_LAYER]
        for j, (name, seq) in enumerate(batch):
            emb = reps[j, 1: len(seq) + 1].mean(0).cpu().numpy()
            embeddings[name] = emb.astype(np.float32)
    return embeddings


# ── PDB helpers ──────────────────────────────────────────────────────────────

def download_pdb(pdb_id: str) -> Optional[Path]:
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
        print(f"  [{pdb_id}] download failed: {e}")
        return None


def extract_chain_sequences(pdb_path: Path) -> dict[str, str]:
    chains: dict[str, list] = {}
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


# ── Per-complex processing ───────────────────────────────────────────────────

def process_complex(entry: dict, fetch_string: bool = False) -> Optional[dict]:
    pdb_id    = entry["pdb_id"]
    req_chains = entry["chains"]
    gt_order  = entry["gt_order"]
    complex_type = entry["type"]
    is_ring   = entry.get("is_ring", False)

    print(f"\n  [{pdb_id}] downloading...")
    pdb_path = download_pdb(pdb_id)
    if pdb_path is None:
        return None

    seq_map = extract_chain_sequences(pdb_path)

    # Validate required chains exist
    missing = [c for c in req_chains if c not in seq_map]
    if missing:
        print(f"  [{pdb_id}] skip: chains {missing} not in PDB (got {list(seq_map.keys())})")
        return None

    chain_ids = req_chains
    sequences = [seq_map[c] for c in chain_ids]
    N = len(chain_ids)

    cplx_dir = OUT_DIR / pdb_id
    cplx_dir.mkdir(parents=True, exist_ok=True)

    # BSA matrix
    sasa_cache = str(cplx_dir / "sasa.npy")
    print(f"  [{pdb_id}] computing SASA ({N} chains)...")
    try:
        bsa_mat = compute_bsa_matrix(str(pdb_path), chain_ids, cache_path=sasa_cache)
    except Exception as e:
        print(f"  [{pdb_id}] SASA failed: {e}, using zeros")
        bsa_mat = np.zeros((N, N), dtype=np.float32)
        np.save(sasa_cache, bsa_mat)

    # ESM-150M embeddings
    print(f"  [{pdb_id}] computing ESM-150M embeddings...")
    seq_dict = {f"{pdb_id}_{c}": s for c, s in zip(chain_ids, sequences)}
    try:
        emb_dict = embed_sequences(seq_dict)
        emb_mat = np.stack([emb_dict[f"{pdb_id}_{c}"] for c in chain_ids])
    except Exception as e:
        print(f"  [{pdb_id}] ESM failed: {e}")
        return None
    np.save(str(cplx_dir / "emb.npy"), emb_mat)

    # AlphaFold pLDDT
    print(f"  [{pdb_id}] fetching pLDDT...")
    uniprot_map = get_chain_uniprot_ids(pdb_id)
    plddt_vecs = []
    uniprot_ids = []
    for c in chain_ids:
        uid = uniprot_map.get(c, "")
        uniprot_ids.append(uid)
        stats = fetch_plddt_stats(uid, PLDDT_CACHE) if uid else None
        plddt_vecs.append(stats_to_node_vec(stats))
    plddt_mat = np.stack(plddt_vecs).astype(np.float32)
    np.save(str(cplx_dir / "plddt.npy"), plddt_mat)

    # STRING PPI
    ppi_mat = np.zeros((N, N), dtype=np.float32)
    if fetch_string and any(uniprot_ids):
        try:
            from src.data_prep.string_api import fetch_ppi_matrix
            print(f"  [{pdb_id}] querying STRING PPI...")
            ppi_mat = fetch_ppi_matrix(uniprot_ids, chain_ids, cache_dir=STRING_CACHE)
        except Exception as e:
            print(f"  [{pdb_id}] STRING failed: {e}")
    np.save(str(cplx_dir / "ppi.npy"), ppi_mat)

    meta = {
        "pdb_id":       pdb_id,
        "chain_ids":    chain_ids,
        "sequences":    sequences,
        "gt_order":     gt_order,
        "complex_type": complex_type,
        "is_ring":      is_ring,
        "n_chains":     N,
        "uniprot_ids":  uniprot_ids,
        "bsa_max":      float(bsa_mat.max()),
        "gt_source":    "experimental",
        "gt_evidence":  entry.get("gt_evidence", ""),
        "ref":          entry.get("ref", ""),
        "name":         entry.get("name", ""),
        "category":     entry.get("category", ""),
    }
    (cplx_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [{pdb_id}] done. type={complex_type} N={N} bsa_max={bsa_mat.max():.1f}")
    return meta


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--string", action="store_true", help="Query STRING PPI")
    parser.add_argument("--resume", action="store_true", help="Skip completed complexes")
    parser.add_argument("--pdb_ids", nargs="*", help="Process specific PDB IDs only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "index.json"

    gt_data = json.loads(GT_V2_PATH.read_text())
    entries = gt_data["complexes"]

    if args.pdb_ids:
        entries = [e for e in entries if e["pdb_id"] in args.pdb_ids]
        print(f"[filter] processing {len(entries)} specified complexes")

    # Load existing index if resuming
    index: list[dict] = []
    done_ids: set[str] = set()
    if args.resume and index_path.exists():
        index = json.loads(index_path.read_text())
        done_ids = {m["pdb_id"] for m in index}
        print(f"[resume] {len(done_ids)} already processed: {sorted(done_ids)}")

    print(f"[start] {len(entries)} complexes in experimental_gt_v2.json")

    for entry in entries:
        pdb_id = entry["pdb_id"]
        if pdb_id in done_ids:
            print(f"  [{pdb_id}] already done, skipping")
            continue
        try:
            meta = process_complex(entry, fetch_string=args.string)
            if meta is not None:
                index.append(meta)
                done_ids.add(pdb_id)
                index_path.write_text(json.dumps(index, indent=2))
        except Exception as e:
            print(f"  [{pdb_id}] error: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"\n[done] {len(index)} complexes built → {OUT_DIR}")
    print(f"[done] index: {index_path}")


if __name__ == "__main__":
    main()
