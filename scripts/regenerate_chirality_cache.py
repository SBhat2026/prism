#!/usr/bin/env python3
"""
Regenerate chirality features for all complexes in data/labeled_15k/.

Reads each complex's meta.json to get PDB ID and chain list, fetches the PDB
file (from data/complexes/ cache or RCSB), computes chirality features, and
writes emb_v2.npy (652d = old 646d + 6 chirality node features) and
edges_v2.npy (5d = ppi/esm_cos/go_cos/sasa_norm + handedness).

Run time: ~3-5 hours for 6,921 complexes on M4 Mac (geometry only, no ESM).
Usage: python scripts/regenerate_chirality_cache.py [--workers 4] [--limit N]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.chirality_features import compute_chirality_features

LABELED_DIR   = ROOT / "data" / "labeled_15k"
COMPLEXES_DIR = ROOT / "data" / "complexes"
PDB_CACHE_DIR = ROOT / "data" / "cache" / "pdb"
PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_pdb(pdb_id: str) -> Path | None:
    """Return path to PDB file, downloading from RCSB if necessary."""
    # Check local complex dirs first
    for subdir in [COMPLEXES_DIR / "marsh2013", COMPLEXES_DIR]:
        candidate = subdir / f"{pdb_id}.pdb"
        if candidate.exists():
            return candidate
    # Check cache
    cached = PDB_CACHE_DIR / f"{pdb_id}.pdb"
    if cached.exists():
        return cached
    # Download
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            cached.write_text(r.text)
            return cached
    except Exception as e:
        print(f"  [warn] Failed to fetch {pdb_id}: {e}")
    return None


def process_complex(complex_dir: Path, force: bool = False) -> dict:
    """Process one complex directory, returning status dict."""
    meta_path = complex_dir / "meta.json"
    if not meta_path.exists():
        return {"dir": str(complex_dir), "status": "skip_no_meta"}

    meta = json.loads(meta_path.read_text())
    pdb_id = meta.get("pdb_id", complex_dir.name.split("_")[0]).upper()
    chains = meta.get("chains", [])
    if not chains:
        return {"dir": str(complex_dir), "status": "skip_no_chains"}

    node_out = complex_dir / "chirality_node.npy"
    edge_out  = complex_dir / "chirality_edge.npy"
    emb_v2    = complex_dir / "emb_v2.npy"

    if emb_v2.exists() and not force:
        return {"dir": str(complex_dir), "status": "skip_exists"}

    pdb_path = fetch_pdb(pdb_id)
    if pdb_path is None:
        return {"dir": str(complex_dir), "status": "error_no_pdb", "pdb_id": pdb_id}

    try:
        node_feats, edge_feats = compute_chirality_features(pdb_path, chains)
        # Write chirality-only arrays for inspection
        np.save(str(node_out), node_feats)
        np.save(str(edge_out), edge_feats)

        # Extend existing emb.npy (N, 646) → emb_v2.npy (N, 652)
        emb_path = complex_dir / "emb.npy"
        if emb_path.exists():
            emb = np.load(str(emb_path))  # (N, 646)
            if emb.shape[0] == node_feats.shape[0]:
                emb_v2_arr = np.concatenate([emb, node_feats], axis=1)
                # Write atomically: .tmp then rename
                tmp = emb_v2.with_suffix(".npy.tmp")
                np.save(str(tmp), emb_v2_arr)
                tmp.rename(emb_v2)

        # Extend existing edges if present
        sasa_path = complex_dir / "sasa.npy"
        edges_out = complex_dir / "edges_v2.npy"
        if sasa_path.exists():
            sasa = np.load(str(sasa_path))  # (N, N) or (N, N, 4)
            if sasa.ndim == 3 and sasa.shape[2] >= 4:
                # Already (N, N, 4) edge features — append handedness as 5th
                h = edge_feats[:, :, np.newaxis] if edge_feats.ndim == 2 else edge_feats
                edges_v2_arr = np.concatenate([sasa, h], axis=2)
                tmp = edges_out.with_suffix(".npy.tmp")
                np.save(str(tmp), edges_v2_arr)
                tmp.rename(edges_out)

        return {"dir": str(complex_dir), "status": "ok", "pdb_id": pdb_id,
                "N": node_feats.shape[0]}
    except Exception as e:
        return {"dir": str(complex_dir), "status": f"error: {e}", "pdb_id": pdb_id}


def main():
    parser = argparse.ArgumentParser(description="Regenerate chirality feature cache")
    parser.add_argument("--workers", type=int, default=4,
                        help="Thread pool size for parallel PDB fetching")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N complexes (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if emb_v2.npy already exists")
    args = parser.parse_args()

    complex_dirs = sorted(LABELED_DIR.glob("*/"))
    if args.limit:
        complex_dirs = complex_dirs[:args.limit]

    total = len(complex_dirs)
    print(f"Processing {total} complex directories with {args.workers} workers...")
    t0 = time.time()

    counts = {"ok": 0, "skip_exists": 0, "skip_no_meta": 0,
              "skip_no_chains": 0, "error_no_pdb": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_complex, d, args.force): d
                   for d in complex_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            status = result["status"]
            key = status if status in counts else "error"
            counts[key] += 1
            if i % 500 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  [{i}/{total}] ok={counts['ok']} skip={counts['skip_exists']} "
                      f"err={counts['error']} | {rate:.1f}/s | ETA {eta/60:.1f}m")

    print(f"\nDone in {(time.time()-t0)/60:.1f}m")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
