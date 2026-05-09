"""
Phase B — Feature precomputation for the labeled 15K corpus.

Reads data/interim/labeled_dataset.tsv (Tier 1 + Tier 2 rows that have a PDB),
downloads PDB structures, computes per-complex feature bundles, and writes:

    data/labeled_15k/{pdb_id}/
        meta.json      — pdb_id, uniprot_ids, gt_order, n_subunits, tier, source
        emb.npy        — (N, 640) ESM-2 t30_150M mean-pool per chain
        sasa.npy       — (N, N) BSA matrix (Å²)
        plddt.npy      — (N, 5) AlphaFold pLDDT stats per chain
        ppi.npy        — (N, N) STRING combined_score / 1000 (zeros if unavailable)
        go.npy         — (N, N) GO Jaccard (zeros — reserved for future)

    data/labeled_15k/index.json  — {pdb_id: {tier, n_subunits, source, has_features}}

Usage:
    python -m src.data_pipeline.precompute_features --tiers 1,2 --limit 10  # smoke test
    python -m src.data_pipeline.precompute_features --tiers 1,2             # full run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_prep.fast_sasa import compute_bsa_matrix
from src.data_prep.plddt_fetcher import (
    fetch_plddt_stats, get_chain_uniprot_ids, stats_to_node_vec,
)

INTERIM_DIR = ROOT / "data" / "interim"
OUT_DIR     = ROOT / "data" / "labeled_15k"
PDB_CACHE   = ROOT / "data" / "cache" / "pdb"
SEQ_CACHE   = ROOT / "data" / "cache" / "sequences"
PLDDT_CACHE = ROOT / "data" / "plddt_cache"
STRING_CACHE= ROOT / "data" / "string"

RCSB_PDB_URL  = "https://files.rcsb.org/download/{pdb_id}.pdb"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"

_AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "PRISM-Precompute/1.0"

# ESM globals (lazy-loaded once)
_ESM_MODEL = None
_ESM_BCONV = None


# ── ESM ──────────────────────────────────────────────────────────────────────

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
    print(f"[esm] loaded on {device}")


def _embed_sequences(seq_dict: dict[str, str], batch_size: int = 4) -> dict[str, np.ndarray]:
    import torch
    _load_esm()
    device = next(_ESM_MODEL.parameters()).device
    items  = list(seq_dict.items())
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        _, _, tokens = _ESM_BCONV([(n, s[:1022]) for n, s in batch])
        with torch.no_grad():
            reps = _ESM_MODEL(tokens.to(device), repr_layers=[30],
                              return_contacts=False)["representations"][30]
        for j, (name, seq) in enumerate(batch):
            emb = reps[j, 1: min(len(seq), 1022) + 1].mean(0).cpu().numpy()
            out[name] = emb.astype(np.float32)
    return out


# ── PDB ──────────────────────────────────────────────────────────────────────

def _download_pdb(pdb_id: str) -> Optional[Path]:
    PDB_CACHE.mkdir(parents=True, exist_ok=True)
    path = PDB_CACHE / f"{pdb_id.upper()}.pdb"
    if path.exists() and path.stat().st_size > 100:
        return path
    try:
        r = _SESSION.get(RCSB_PDB_URL.format(pdb_id=pdb_id.upper()), timeout=30)
        r.raise_for_status()
        path.write_text(r.text)
        time.sleep(0.3)
        return path
    except Exception as e:
        print(f"    [pdb] {pdb_id} download failed: {e}")
        return None


def _extract_chain_seqs(pdb_path: Path) -> dict[str, str]:
    chains: dict[str, list[str]] = {}
    seen: dict[str, set] = {}
    with open(pdb_path) as f:
        for line in f:
            if line[:4] != "ATOM" or line[13:15].strip() != "CA":
                continue
            ch = line[21]
            key = (line[22:26].strip(), line[17:20].strip())
            chains.setdefault(ch, [])
            seen.setdefault(ch, set())
            if key not in seen[ch]:
                seen[ch].add(key)
                chains[ch].append(_AA3TO1.get(line[17:20].strip(), "X"))
    return {c: "".join(s) for c, s in chains.items() if len(s) >= 20}


# ── UniProt sequence fetch (for Tier 3 fallback, not used in Tier 1/2) ───────

def _fetch_uniprot_seq(uid: str) -> Optional[str]:
    SEQ_CACHE.mkdir(parents=True, exist_ok=True)
    cache = SEQ_CACHE / f"{uid}.fa"
    if cache.exists():
        lines = cache.read_text().splitlines()
        return "".join(l for l in lines if not l.startswith(">"))
    try:
        r = _SESSION.get(UNIPROT_FASTA.format(uid=uid), timeout=10)
        time.sleep(0.1)
        if r.status_code != 200:
            return None
        cache.write_text(r.text)
        lines = r.text.splitlines()
        return "".join(l for l in lines if not l.startswith(">"))
    except Exception:
        return None


# ── STRING PPI ────────────────────────────────────────────────────────────────

def _fetch_ppi_matrix(uniprot_ids: list[str]) -> np.ndarray:
    N = len(uniprot_ids)
    mat = np.zeros((N, N), dtype=np.float32)
    try:
        from src.data_prep.string_api import fetch_ppi_matrix
        mat = fetch_ppi_matrix(uniprot_ids, list(range(N)), cache_dir=STRING_CACHE)
    except Exception:
        pass
    return mat


# ── Per-complex processing ────────────────────────────────────────────────────

def process_one(row: pd.Series, out_dir: Path) -> bool:
    """
    Build feature bundle for one labeled_dataset row.
    Returns True on success, False on skip/error.
    """
    pdb_id = str(row.get("pdb_id", "")).strip().upper()
    if not pdb_id or pdb_id in ("", "NAN", "NO_STRUCTURE"):
        return False

    key = pdb_id
    cplx_dir = out_dir / key
    if (cplx_dir / "meta.json").exists():
        return True  # already done — skip (resumable)

    uid_json = row.get("uniprot_ids", "[]")
    uid_list  = json.loads(uid_json) if isinstance(uid_json, str) else list(uid_json)
    uid_list  = [u for u in uid_list if u]
    gt_order  = json.loads(row.get("gt_order", "[]")) if isinstance(row.get("gt_order"), str) else []

    print(f"  [{pdb_id}] n={row['n_subunits']} tier={row['gt_tier']}")

    # 1. Download PDB
    pdb_path = _download_pdb(pdb_id)
    if pdb_path is None:
        return False

    # 2. Extract chain sequences from PDB
    seq_map = _extract_chain_seqs(pdb_path)
    if len(seq_map) < 2:
        print(f"    [{pdb_id}] skip: only {len(seq_map)} chains in PDB")
        return False

    chain_ids = sorted(seq_map.keys())
    N = len(chain_ids)

    # 3. SASA / BSA matrix
    sasa_cache = str(cplx_dir / "sasa.npy") if False else None  # write after mkdir
    cplx_dir.mkdir(parents=True, exist_ok=True)
    sasa_cache = str(cplx_dir / "sasa.npy")
    try:
        bsa_mat = compute_bsa_matrix(str(pdb_path), chain_ids, cache_path=sasa_cache)
    except Exception as e:
        print(f"    [{pdb_id}] SASA failed: {e}")
        bsa_mat = np.zeros((N, N), dtype=np.float32)
    np.save(sasa_cache, bsa_mat)

    # 4. ESM-2 embeddings
    seq_dict = {c: seq_map[c] for c in chain_ids}
    try:
        emb_dict = _embed_sequences(seq_dict)
        emb_mat  = np.stack([emb_dict[c] for c in chain_ids])
    except Exception as e:
        print(f"    [{pdb_id}] ESM failed: {e}")
        return False
    np.save(str(cplx_dir / "emb.npy"), emb_mat)

    # 5. pLDDT (chain → UniProt via SIFTS, then AlphaFold)
    try:
        chain_uid = get_chain_uniprot_ids(pdb_id)
    except Exception:
        chain_uid = {}
    plddt_rows = []
    resolved_uids = []
    for c in chain_ids:
        uid = chain_uid.get(c, "")
        resolved_uids.append(uid)
        if uid:
            stats = fetch_plddt_stats(uid, PLDDT_CACHE)
            vec   = stats_to_node_vec(stats)
        else:
            vec = stats_to_node_vec(None)
        plddt_rows.append(vec)
    plddt_mat = np.stack(plddt_rows)
    np.save(str(cplx_dir / "plddt.npy"), plddt_mat)

    # 6. STRING PPI (best-effort; zeros on failure/rate-limit)
    effective_uids = resolved_uids if any(resolved_uids) else uid_list[:N]
    ppi_mat = _fetch_ppi_matrix(effective_uids)
    np.save(str(cplx_dir / "ppi.npy"), ppi_mat)

    # 7. GO matrix placeholder (zeros; filled by downstream ProtFuncBridge if desired)
    np.save(str(cplx_dir / "go.npy"), np.zeros((N, N), dtype=np.float32))

    # 8. meta.json
    meta = {
        "pdb_id":      pdb_id,
        "chain_ids":   chain_ids,
        "uniprot_ids": resolved_uids,
        "gt_order":    gt_order,
        "n_subunits":  N,
        "gt_tier":     int(row.get("gt_tier", 3)),
        "source":      str(row.get("source", "")),
        "complex_type": str(row.get("type", "heteromer")),
        "is_ring":     bool(row.get("is_ring", False)),
    }
    (cplx_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="1,2",
                    help="Comma-separated tier numbers to include (default: 1,2)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N complexes (0 = unlimited)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers (currently unused — sequential for MPS safety)")
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "index.json"

    # Load existing index for resumability
    index: dict = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except Exception:
            index = {}

    # Load labeled dataset
    tsv_path = INTERIM_DIR / "labeled_dataset.tsv"
    if not tsv_path.exists():
        print(f"[error] {tsv_path} not found — run the data pipeline first")
        sys.exit(1)

    df = pd.read_csv(tsv_path, sep="\t")
    df_filtered = df[df["gt_tier"].isin(tiers)].copy()

    # Filter to rows with a valid pdb_id
    has_pdb = df_filtered["pdb_id"].notna() & \
              ~df_filtered["pdb_id"].astype(str).str.upper().isin(["", "NAN", "NO_STRUCTURE"])
    df_filtered = df_filtered[has_pdb].reset_index(drop=True)

    # Deduplicate by pdb_id (keep first occurrence)
    df_filtered = df_filtered.drop_duplicates(subset=["pdb_id"]).reset_index(drop=True)

    print(f"[precompute] {len(df_filtered)} complexes (tiers {tiers}, has PDB)")
    if args.limit:
        df_filtered = df_filtered.head(args.limit)
        print(f"[precompute] limited to {len(df_filtered)} for smoke test")

    ok = skip = err = 0
    for i, row in df_filtered.iterrows():
        pdb_id = str(row.get("pdb_id", "")).strip().upper()
        # Already in index and has_features → skip
        if index.get(pdb_id, {}).get("has_features"):
            skip += 1
            continue

        try:
            success = process_one(row, OUT_DIR)
        except Exception as e:
            print(f"  [{pdb_id}] unhandled error: {e}")
            success = False

        index[pdb_id] = {
            "tier":        int(row.get("gt_tier", 3)),
            "n_subunits":  int(row.get("n_subunits", 0)),
            "source":      str(row.get("source", "")),
            "has_features": success,
        }

        if success:
            ok += 1
        else:
            err += 1

        # Flush index every 100 rows
        if (ok + err) % 100 == 0:
            index_path.write_text(json.dumps(index, indent=2))
            print(f"[progress] {ok} ok / {err} err / {skip} skip  "
                  f"({ok + err + skip}/{len(df_filtered)})")

    index_path.write_text(json.dumps(index, indent=2))
    print(f"\n[done] {ok} built, {err} failed, {skip} skipped → {OUT_DIR}")
    print(f"[done] index: {index_path}  ({len(index)} total entries)")


if __name__ == "__main__":
    main()
