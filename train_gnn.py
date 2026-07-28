"""
Full GAT-based AssemblyScorer training.

Architecture: 3-layer MultiHeadGAT (4 heads, 128 hidden)
Node features: ESM-2 8M (320d) + 4 GO summary stats + 1 SASA-diag = 325d
Edge features: [ppi, esm_cos, go_cos, sasa_norm] = 4d

Training: softmax CE at each assembly step (same as WeightLearner)
Eval:     Kendall's τ via greedy assembly using GNN-predicted scores

Outputs:
  results/gnn_training/epochs/epoch_NNNN.png  — per-epoch dashboard
  results/gnn_training/training_log.json       — full metrics history
  results/gnn_training/best_gnn.pt             — best τ checkpoint
  results/gnn_training/dashboard.html          — live browser dashboard
"""

import os
import sys
import json
import copy
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from pathlib import Path
from typing import Optional
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.scorer_gnn import (
    AssemblyScorer, build_node_features, build_edge_features, build_adjacency,
    build_copy_indices,
)
from src.training.dataset import ComplexData, build_examples, build_step_groups
from src.training.losses import kendall_tau_from_order
from src.data_prep.plddt_fetcher import (
    get_chain_uniprot_ids, fetch_plddt_stats, stats_to_node_vec, PLDDT_NODE_DIM
)
from src.evaluation.metrics import evaluate_dataset
from src.data_prep.geom_features import build_geom_matrix, GEOM_DIM
from src.data_prep.voronoi_interface import compute_contact_density_matrix

MODEL_VERSION = "v4"
GEOM_CACHE    = ROOT / "data" / "feature_cache"

GNN_DIR      = ROOT / "results" / "gnn_training"
EPOCH_DIR    = GNN_DIR / "epochs"
EPOCH_DIR.mkdir(parents=True, exist_ok=True)
PLDDT_CACHE  = ROOT / "data" / "plddt_cache"
TRAINING_DIR = ROOT / "data" / "training_complexes"
GO_CACHE     = ROOT / "data" / "go_cache"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Data loading (reuse cached outputs from run_alpha.py) ────────────────────

# PDB chain IDs and subcomplex topology — requires structural/domain knowledge, kept in code.
# gt_order is loaded from data/gt/experimental_gt.json (literature-sourced, authoritative).
_BENCHMARK_STRUCTURE = {
    "2HHB": {"chains": ["A","B","C","D"],        "type": "heteromer", "subcomplexes": [[0,1],[2,3]]},
    "1BRS": {"chains": ["A","B","C","D","E","F"],"type": "heteromer", "subcomplexes": [[0,1],[2,3],[4,5]]},
    "1AY7": {"chains": ["A","B"],                "type": "heteromer"},
    "1TGS": {"chains": ["Z","I"],                "type": "heteromer"},
    "2PTC": {"chains": ["E","I"],                "type": "heteromer"},
    "1A2K": {"chains": ["A","B","C","D","E"],    "type": "heteromer"},
    "1SBB": {"chains": ["A","B","C","D"],        "type": "heteromer"},
    "3GBN": {"chains": ["A","H","L"],            "type": "heteromer", "subcomplexes": [[1,2],[0]]},
    "1AON": {"chains": list("ABCDEFG"),          "type": "homomer"},
    "1GRU": {"chains": list("ABCDEFG"),          "type": "homomer"},
}

def _load_benchmark() -> dict:
    _gt_by_pdb = {}
    gt_path = ROOT / "data" / "gt" / "experimental_gt.json"
    if gt_path.exists():
        data = json.loads(gt_path.read_text())
        for entry in data.get("complexes", []):
            _gt_by_pdb[entry["pdb_id"]] = entry["gt_order"]
    result = {}
    for pdb_id, struct in _BENCHMARK_STRUCTURE.items():
        N = len(struct["chains"])
        entry = {"chains": struct["chains"], "type": struct["type"],
                 "gt_order": _gt_by_pdb.get(pdb_id, list(range(N)))}
        if "subcomplexes" in struct:
            entry["subcomplexes"] = struct["subcomplexes"]
        result[pdb_id] = entry
    return result

BENCHMARK = _load_benchmark()


def _load_plddt_for_complex(pdb_id: str, chains: list[str]) -> np.ndarray:
    """
    Fetch AlphaFold pLDDT stats for each chain via RCSB UniProt mapping.
    Returns (N, PLDDT_NODE_DIM) float32 array.
    Falls back to neutral defaults for chains without AF models.
    """
    uniprot_map = get_chain_uniprot_ids(pdb_id)
    vecs = []
    for c in chains:
        uid = uniprot_map.get(c, "")
        if uid:
            stats = fetch_plddt_stats(uid, PLDDT_CACHE)
        else:
            stats = None
        vecs.append(stats_to_node_vec(stats))
    return np.stack(vecs).astype(np.float32)  # (N, PLDDT_NODE_DIM)


def load_dataset_from_alpha(fetch_plddt: bool = True) -> list[ComplexData]:
    """
    Build ComplexData for Marsh 2013 benchmark from cached embeddings + GROMACS SASA.
    Uses ESM-2 150M (640d) if available, else 8M (320d).
    Optionally fetches AlphaFold pLDDT per chain (cached in data/plddt_cache/).
    """
    MARSH_DIR  = ROOT / "data" / "complexes" / "marsh2013"
    EMBED_8M   = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t6_8M_UR50D"
    EMBED_150M = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t30_150M_UR50D"
    STRING_CACHE = ROOT / "data" / "string"

    use_150m   = EMBED_150M.exists()
    embed_dir  = EMBED_150M if use_150m else EMBED_8M
    esm_dim    = 640 if use_150m else 320
    emb_data   = np.load(str(embed_dir / "embeddings.npz"))

    manifest_path = MARSH_DIR / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}

    dataset = []
    for pdb_id, meta in BENCHMARK.items():
        chains   = meta["chains"]
        N        = len(chains)
        pdb_path = str(MARSH_DIR / f"{pdb_id}.pdb")

        seqs = [manifest.get(pdb_id, {}).get("sequences", {}).get(c, "M"*50) for c in chains]

        embs = np.array([
            emb_data.get(f"{pdb_id}_{c}", np.zeros(esm_dim)) for c in chains
        ], dtype=np.float32)
        norms   = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_n  = embs / (norms + 1e-8)
        esm_mat = (embs_n @ embs_n.T).astype(np.float32).clip(-1, 1)

        # SASA — try cache first, then freesasa, then GROMACS fallback
        sasa_cache = ROOT / "data" / "sasa_cache" / f"{pdb_id}_sasa.npy"
        sasa_cache.parent.mkdir(parents=True, exist_ok=True)
        if sasa_cache.exists():
            sasa_mat = np.load(str(sasa_cache)).astype(np.float32)
            if sasa_mat.shape != (N, N):
                sasa_cache.unlink()  # stale — different chain count
                sasa_mat = None
        else:
            sasa_mat = None
        if sasa_mat is None:
            sasa_mat = np.zeros((N, N), dtype=np.float32)
            if Path(pdb_path).exists():
                try:
                    from src.data_prep.fast_sasa import compute_bsa_matrix
                    sasa_mat = compute_bsa_matrix(pdb_path, chains).astype(np.float32)
                    np.save(str(sasa_cache), sasa_mat)
                    print(f"  {pdb_id}: SASA computed via freesasa + cached")
                except Exception as e:
                    print(f"  {pdb_id}: freesasa failed ({e}), trying GROMACS")
                    try:
                        from src.gromacs.extract_sasa import GromacsSASA
                        calc = GromacsSASA()
                        for i, ci in enumerate(chains):
                            for j, cj in enumerate(chains):
                                if j <= i: continue
                                try:
                                    bsa = calc.interface_buried_area(pdb_path, ci, cj)
                                    sasa_mat[i, j] = float(bsa)
                                    sasa_mat[j, i] = float(bsa)
                                except Exception:
                                    pass
                        np.save(str(sasa_cache), sasa_mat)
                        print(f"  {pdb_id}: SASA computed via GROMACS + cached")
                    except Exception as e2:
                        print(f"  {pdb_id}: SASA unavailable ({e2})")

        # STRING PPI — load from cache if available
        ppi_cache = STRING_CACHE / f"{pdb_id}_ppi.npy"
        if ppi_cache.exists():
            ppi_mat = np.load(str(ppi_cache)).astype(np.float32)
        else:
            ppi_mat = np.zeros((N, N), dtype=np.float32)

        # AlphaFold pLDDT
        plddt_cache = PLDDT_CACHE / f"{pdb_id}_plddt.npy"
        if plddt_cache.exists():
            plddt = np.load(str(plddt_cache)).astype(np.float32)
        elif fetch_plddt:
            plddt = _load_plddt_for_complex(pdb_id, chains)
            PLDDT_CACHE.mkdir(parents=True, exist_ok=True)
            np.save(str(plddt_cache), plddt)
            print(f"  {pdb_id}: pLDDT fetched + cached")
        else:
            plddt = np.full((N, PLDDT_NODE_DIM), 70.0, dtype=np.float32)
            plddt[:, 1] = 0.3  # frac_gt90 default
            plddt[:, 2] = 0.1  # frac_lt50 default

        dataset.append(ComplexData(
            pdb_id=pdb_id,
            complex_type=meta["type"],
            chain_ids=chains,
            sequences=seqs,
            gt_order=meta["gt_order"],
            sasa_matrix=sasa_mat,
            esm_matrix=esm_mat,
            go_matrix=np.zeros((N, N), dtype=np.float32),
            ppi_matrix=ppi_mat,
            esm_embeddings=embs,
            plddt=plddt,
            gt_source="experimental",
            subcomplexes=meta.get("subcomplexes"),
            pdb_path=pdb_path if Path(pdb_path).exists() else None,
        ))
        print(f"  {pdb_id}: sasa_max={sasa_mat.max():.1f} "
              f"plddt_mean={plddt[:,0].mean():.1f} ppi_max={ppi_mat.max():.3f}")

    print(f"[data] loaded {len(dataset)} Marsh2013 complexes "
          f"(esm_dim={esm_dim}, use_150m={use_150m})")
    return dataset


def load_expanded_dataset() -> list[ComplexData]:
    """
    Load pre-built PDB training complexes from data/training_complexes/.
    Falls back gracefully if the directory doesn't exist yet.
    """
    index_path = TRAINING_DIR / "index.json"
    if not index_path.exists():
        print("[data] no expanded dataset found — run build_pdb_training_set.py first")
        return []

    index = json.loads(index_path.read_text())
    dataset = []

    for meta in index:
        pdb_id = meta["pdb_id"]
        cplx_dir = TRAINING_DIR / pdb_id

        emb_path  = cplx_dir / "emb.npy"
        sasa_path = cplx_dir / "sasa.npy"
        plddt_path= cplx_dir / "plddt.npy"
        ppi_path  = cplx_dir / "ppi.npy"

        if not (emb_path.exists() and sasa_path.exists()):
            continue

        chains    = meta["chain_ids"]
        N         = len(chains)
        emb_mat   = np.load(str(emb_path)).astype(np.float32)
        sasa_mat  = np.load(str(sasa_path)).astype(np.float32)
        plddt     = np.load(str(plddt_path)).astype(np.float32) if plddt_path.exists() \
                    else np.full((N, PLDDT_NODE_DIM), 70.0, dtype=np.float32)
        ppi_mat   = np.load(str(ppi_path)).astype(np.float32) if ppi_path.exists() \
                    else np.zeros((N, N), dtype=np.float32)

        norms  = np.linalg.norm(emb_mat, axis=1, keepdims=True)
        emb_n  = emb_mat / (norms + 1e-8)
        esm_cos= (emb_n @ emb_n.T).clip(-1, 1).astype(np.float32)

        dataset.append(ComplexData(
            pdb_id=pdb_id,
            complex_type=meta["complex_type"],
            chain_ids=chains,
            sequences=meta.get("sequences", [""] * N),
            gt_order=meta["gt_order"],
            sasa_matrix=sasa_mat,
            esm_matrix=esm_cos,
            go_matrix=np.zeros((N, N), dtype=np.float32),
            ppi_matrix=ppi_mat,
            esm_embeddings=emb_mat,
            plddt=plddt,
            gt_source=meta.get("gt_source", "sasa_greedy"),
            uniprot_ids=meta.get("uniprot_ids", []),
        ))

    print(f"[data] loaded {len(dataset)} expanded complexes")
    return dataset


def load_experimental_v2_dataset() -> list[ComplexData]:
    """
    Load pre-built experimental GT v2 complexes from data/experimental_complexes/.
    Run src/data_prep/build_experimental_dataset.py first.
    """
    exp_dir    = ROOT / "data" / "experimental_complexes"
    index_path = exp_dir / "index.json"
    if not index_path.exists():
        print("[data] no experimental_complexes/index.json — run build_experimental_dataset.py first")
        return []

    index = json.loads(index_path.read_text())
    dataset = []
    for meta in index:
        pdb_id   = meta["pdb_id"]
        cplx_dir = exp_dir / pdb_id
        emb_path  = cplx_dir / "emb.npy"
        sasa_path = cplx_dir / "sasa.npy"
        if not (emb_path.exists() and sasa_path.exists()):
            print(f"  [v2] {pdb_id} missing files, skipping")
            continue
        chains  = meta["chain_ids"]
        N       = len(chains)
        emb_mat = np.load(str(emb_path)).astype(np.float32)
        sasa_mat= np.load(str(sasa_path)).astype(np.float32)
        plddt   = np.load(str(cplx_dir / "plddt.npy")).astype(np.float32) \
                  if (cplx_dir / "plddt.npy").exists() \
                  else np.full((N, PLDDT_NODE_DIM), 70.0, dtype=np.float32)
        ppi_mat = np.load(str(cplx_dir / "ppi.npy")).astype(np.float32) \
                  if (cplx_dir / "ppi.npy").exists() \
                  else np.zeros((N, N), dtype=np.float32)
        norms   = np.linalg.norm(emb_mat, axis=1, keepdims=True)
        emb_n   = emb_mat / (norms + 1e-8)
        esm_cos = (emb_n @ emb_n.T).clip(-1, 1).astype(np.float32)
        dataset.append(ComplexData(
            pdb_id=pdb_id,
            complex_type=meta["complex_type"],
            chain_ids=chains,
            sequences=meta.get("sequences", [""] * N),
            gt_order=meta["gt_order"],
            sasa_matrix=sasa_mat,
            esm_matrix=esm_cos,
            go_matrix=np.zeros((N, N), dtype=np.float32),
            ppi_matrix=ppi_mat,
            esm_embeddings=emb_mat,
            plddt=plddt,
            gt_source="experimental",
            uniprot_ids=meta.get("uniprot_ids", []),
        ))

    print(f"[data] loaded {len(dataset)} experimental_v2 complexes")
    return dataset


def load_labeled_15k_dataset(split: str = "train", max_complexes: int = 0) -> list[ComplexData]:
    """Load labeled_15k complexes from data/labeled_15k/{pdb_id}/."""
    cplx_dir = ROOT / "data" / "labeled_15k"
    if not cplx_dir.exists():
        print("[data] data/labeled_15k/ not found")
        return []

    all_dirs = sorted([d for d in cplx_dir.iterdir() if d.is_dir()])
    if max_complexes > 0:
        all_dirs = all_dirs[:max_complexes]

    dataset = []
    skipped = 0
    for d in all_dirs:
        meta_path = d / "meta.json"
        emb_path  = d / "emb.npy"
        sasa_path = d / "sasa.npy"
        if not (meta_path.exists() and emb_path.exists() and sasa_path.exists()):
            skipped += 1; continue
        try:
            meta    = json.loads(meta_path.read_text())
            pdb_id  = meta["pdb_id"]
            chains  = meta["chain_ids"]
            N       = len(chains)
            emb_mat = np.load(str(emb_path)).astype(np.float32)
            sasa_mat= np.load(str(sasa_path)).astype(np.float32)
            if emb_mat.shape[0] != N or sasa_mat.shape != (N, N):
                skipped += 1; continue
            plddt   = np.load(str(d / "plddt.npy")).astype(np.float32) \
                      if (d / "plddt.npy").exists() \
                      else np.full((N, PLDDT_NODE_DIM), 70.0, dtype=np.float32)
            ppi_raw = np.load(str(d / "ppi.npy")).astype(np.float32) \
                      if (d / "ppi.npy").exists() else np.zeros((N, N), dtype=np.float32)
            ppi_mat = ppi_raw if ppi_raw.shape == (N, N) else np.zeros((N, N), dtype=np.float32)
            go_raw  = np.load(str(d / "go.npy")).astype(np.float32) \
                      if (d / "go.npy").exists() else np.zeros((N, N), dtype=np.float32)
            go_mat  = go_raw if go_raw.shape == (N, N) else np.zeros((N, N), dtype=np.float32)
            norms   = np.linalg.norm(emb_mat, axis=1, keepdims=True)
            esm_cos = (emb_mat / (norms + 1e-8) @ (emb_mat / (norms + 1e-8)).T).clip(-1,1).astype(np.float32)
            pdb_cache = ROOT / "data" / "cache" / "pdb" / f"{pdb_id}.pdb"
            dataset.append(ComplexData(
                pdb_id=pdb_id,
                complex_type=meta.get("complex_type", "heteromer"),
                chain_ids=chains,
                sequences=meta.get("sequences", [""] * N),
                gt_order=meta.get("gt_order", list(range(N))),
                sasa_matrix=sasa_mat,
                esm_matrix=esm_cos,
                go_matrix=go_mat,
                ppi_matrix=ppi_mat,
                esm_embeddings=emb_mat,
                plddt=plddt,
                gt_source=meta.get("source", "sasa_greedy"),
                uniprot_ids=meta.get("uniprot_ids", []),
                pdb_path=str(pdb_cache) if pdb_cache.exists() else None,
            ))
        except Exception:
            skipped += 1

    # Deduplicate by pdb_id
    seen, deduped = set(), []
    for c in dataset:
        if c.pdb_id not in seen:
            seen.add(c.pdb_id); deduped.append(c)
    dataset = deduped

    print(f"[data] loaded {len(dataset)} labeled_15k '{split}' complexes (skipped {skipped} missing)")
    return dataset


# ── GNN step-group scorer ────────────────────────────────────────────────────

def score_step_gnn(
    model: AssemblyScorer,
    cdata: ComplexData,
    assembled: list[int],
    candidates: list[int],
    node_feats: torch.Tensor,
    edge_feats: torch.Tensor,
    adj: torch.Tensor,
    h: Optional[torch.Tensor] = None,
    copy_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score all candidates at one step. Pass precomputed h to avoid redundant embed."""
    model.eval()
    with torch.no_grad():
        N = len(cdata.chain_ids)
        assembled_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        if assembled:
            assembled_mask[assembled] = True
        if h is None:
            h = model.embed(node_feats, adj, edge_feats, copy_indices)
        return model.score_candidates(h, edge_feats, assembled_mask, candidates)


def greedy_gnn(model: AssemblyScorer, cdata: ComplexData,
               node_feats, edge_feats, adj,
               copy_indices: Optional[torch.Tensor] = None) -> list[int]:
    """Greedy assembly order predicted by GNN. Embeds once, reuses h per step."""
    N = len(cdata.chain_ids)
    model.eval()
    with torch.no_grad():
        h = model.embed(node_feats, adj, edge_feats, copy_indices)
    order = [cdata.gt_order[0]]
    remaining = [i for i in range(N) if i != order[0]]
    for _ in range(N - 1):
        if not remaining:
            break
        scores = score_step_gnn(model, cdata, order, remaining,
                                 node_feats, edge_feats, adj, h=h,
                                 copy_indices=copy_indices)
        best   = remaining[int(scores.argmax().item())]
        order.append(best)
        remaining.remove(best)
    return order


# ── Build all precomputed tensors ────────────────────────────────────────────

def precompute(cdata: ComplexData, esm_dim: int = 320, include_chiral: bool = False):
    """
    Returns (node_feats, edge_feats, adj, copy_indices) for one complex.

    Node features: ESM (640d) + pLDDT (5d) + SASA_diag (1d) + geom (9d) = 655d
    Edge features: [ppi, esm_cos, go_cos, sasa_norm, contact_density] = 5d
    copy_indices: (N,) long — 0-indexed copy count per unique UniProt (Step 10)

    include_chiral=True appends the Lever-1 handedness descriptors:
      Node +3d [tetra_chirality, helical_sense, frame_handedness] → 658d
      Edge +1d [interface handedness det(û_ij, v_i, v_j)]          → 6d
    Off by default so v4/v5/v6 checkpoints keep loading unchanged.
    """
    N    = len(cdata.chain_ids)
    sasa = cdata.sasa_matrix / (cdata.sasa_matrix.max() + 1e-8)

    # pLDDT features: (N, 5). Normalise to ~[0,1] range.
    if cdata.plddt is not None:
        plddt_feats = cdata.plddt.astype(np.float32).copy()
        plddt_feats[:, 0] /= 100.0   # mean pLDDT: [0,100] → [0,1]
        plddt_feats[:, 3] /= 100.0   # q10
        plddt_feats[:, 4] /= 100.0   # q90
        # cols 1,2 (frac_gt90, frac_lt50) already in [0,1]
    else:
        plddt_feats = np.zeros((N, PLDDT_NODE_DIM), dtype=np.float32)
        plddt_feats[:, 0] = 0.70  # neutral default

    sasa_diag = np.diag(sasa).astype(np.float32)

    # Monomer geometry features (9d per chain from AlphaFold CA coords + pLDDT)
    geom_cache_dir = GEOM_CACHE / "af_pdbs"
    uniprot_ids    = cdata.uniprot_ids or [""] * N
    sequences      = cdata.sequences or ["M" * 100] * N
    geom_feats     = build_geom_matrix(uniprot_ids, geom_cache_dir, sequences)  # (N, 9)

    node_parts = [
        cdata.esm_embeddings.astype(np.float32),  # (N, esm_dim)
        plddt_feats,                               # (N, 5)
        sasa_diag.reshape(-1, 1),                  # (N, 1)
        geom_feats,                                # (N, 9)
    ]

    # CHIRALITY-SENSITIVE FEATURES (Lever 1 interim) — opt-in, appended last.
    chiral_edge = None
    if include_chiral:
        from src.features.chiral_descriptors import compute_chiral_features
        chiral_node, chiral_edge = compute_chiral_features(
            cdata.pdb_id, cdata.chain_ids, cdata.pdb_path
        )
        node_parts.append(chiral_node)             # (N, 3)

    node_feats = np.hstack(node_parts)
    node_feats_t = torch.tensor(node_feats, dtype=torch.float32)

    # Voronoi contact density (only available when full-complex PDB is present)
    contact_density = None
    if cdata.pdb_path:
        try:
            cache_path = str(GEOM_CACHE / f"{cdata.pdb_id}_contact.npy")
            contact_density = compute_contact_density_matrix(
                cdata.pdb_path, cdata.chain_ids, cache_path=cache_path
            )
        except Exception:
            contact_density = None

    # Always include contact_density slot even when None, so edge_dim=5 is consistent
    contact_slot = contact_density if contact_density is not None \
                   else np.zeros((N, N), dtype=np.float32)
    edge_feats_t = build_edge_features(
        cdata.ppi_matrix, cdata.esm_matrix, cdata.go_matrix, sasa, contact_slot
    )
    if chiral_edge is not None:
        edge_feats_t = torch.cat(
            [edge_feats_t, torch.tensor(chiral_edge, dtype=torch.float32)], dim=-1
        )
    adj_dense = torch.ones(N, N, dtype=torch.bool)

    # Copy-index embedding for symmetry breaking (Step 10)
    copy_indices_t = build_copy_indices(uniprot_ids).to(DEVICE)

    # Move to device once at precompute time — avoids per-forward transfer overhead
    return node_feats_t.to(DEVICE), edge_feats_t.to(DEVICE), adj_dense.to(DEVICE), copy_indices_t


# ── Training loss for GNN ────────────────────────────────────────────────────

def _subcomplex_orders(gt_order: list[int], subcomplexes: list[list[int]]) -> list[list[int]]:
    """
    Enumerate sequential assembly orderings consistent with known subcomplex topology.

    For heteromers that assemble via obligate subcomplexes (e.g. hemoglobin α₁β₁ + α₂β₂),
    any interleaving of subcomplex-internal orderings is valid.
    Returns up to 48 orderings to bound compute cost.
    """
    from itertools import permutations, product as iproduct

    # Within each subcomplex, enumerate orderings (cap at 2 per subcomplex for large ones)
    sub_perms = []
    for sc in subcomplexes:
        if len(sc) <= 2:
            sub_perms.append(list(permutations(sc)))
        else:
            # For larger subcomplexes, use only the GT-consistent ordering and its reverse
            gt_sub = [x for x in gt_order if x in sc]
            sub_perms.append([gt_sub, list(reversed(gt_sub))])

    results = set()
    for combo in iproduct(*sub_perms):
        # combo = tuple of ordered sublists; generate interleavings
        # Use a simple recursive interleaving approach
        def _interleave(seqs):
            if not seqs:
                yield []
                return
            if len(seqs) == 1:
                yield list(seqs[0])
                return
            for i, seq in enumerate(seqs):
                rest = seqs[:i] + seqs[i+1:]
                for tail in _interleave(rest):
                    yield list(seq) + tail

        for order in _interleave(list(combo)):
            results.add(tuple(order))
            if len(results) >= 48:
                break
        if len(results) >= 48:
            break

    return [list(r) for r in results]


def _get_orders(cdata: ComplexData) -> list[list[int]]:
    """
    Return list of valid assembly orders for training.

    Homomers: augment with all 2N cyclic rotations — N forward + N backward.
      Both ring directions are structurally valid; training on only one direction
      creates a false gradient ceiling (the 1GRU SASA contradiction).
    Heteromers: single GT order only, unless subcomplex topology is defined.
    """
    base = cdata.gt_order
    if cdata.complex_type == "homomer":
        N = len(base)
        rev = list(reversed(base))
        orders = []
        for i in range(N):
            orders.append(base[i:] + base[:i])   # N forward rotations
            orders.append(rev[i:] + rev[:i])      # N backward rotations
        return orders
    # Heteromer — check for subcomplex topology
    if getattr(cdata, 'subcomplexes', None):
        return _subcomplex_orders(base, cdata.subcomplexes)
    return [base]


def _order_loss_from_h(
    model: AssemblyScorer,
    N: int,
    order: list[int],
    h: torch.Tensor,
    edge_feats: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cross-entropy loss for one assembly order, given precomputed node embeddings h.
    Runs embed once per complex (not per candidate) for a ~5-10x speedup."""
    n_supervised = len(order)
    step_losses = []
    for step in range(1, n_supervised):
        assembled = order[:step]
        correct   = order[step]
        remaining = [i for i in range(N) if i not in assembled]
        if not remaining or correct not in remaining:
            continue
        assembled_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        assembled_mask[assembled] = True
        scores = model.score_candidates(h, edge_feats, assembled_mask, remaining) / temperature
        correct_idx = remaining.index(correct)
        # Label smoothing ε=0.1
        n_cands = scores.size(0)
        smooth = 0.1
        soft_labels = torch.full((n_cands,), smooth / n_cands, device=DEVICE)
        soft_labels[correct_idx] += (1.0 - smooth)
        log_probs = F.log_softmax(scores, dim=0)
        step_losses.append(-(soft_labels * log_probs).sum())
    if not step_losses:
        return torch.tensor(0.0, device=DEVICE)
    return torch.stack(step_losses).mean()


def gnn_ranking_loss(
    model: AssemblyScorer,
    dataset: list[ComplexData],
    precomputed: dict,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Ring-aware ranking loss. embed() is called ONCE per complex; all orderings
    and all steps reuse the same h — avoids O(N²) redundant GNN forward passes.
    """
    model.train()
    complex_losses = []

    for cdata in dataset:
        N = len(cdata.chain_ids)
        node_feats, edge_feats, adj, copy_indices = precomputed[cdata.pdb_id]
        # Single embed for this complex — shared across all orderings
        h = model.embed(node_feats, adj, edge_feats, copy_indices)
        orders = _get_orders(cdata)

        if cdata.complex_type == "homomer" or (getattr(cdata, 'subcomplexes', None) and len(orders) > 1):
            ring_losses = [
                _order_loss_from_h(model, N, order, h, edge_feats, temperature)
                for order in orders
            ]
            complex_losses.append(torch.stack(ring_losses).min())
        else:
            complex_losses.append(
                _order_loss_from_h(model, N, orders[0], h, edge_feats, temperature)
            )

    if not complex_losses:
        return torch.tensor(0.0, device=DEVICE)
    return torch.stack(complex_losses).mean()


# ── Evaluate τ ───────────────────────────────────────────────────────────────

def _circular_tau(pred_order: list[int], gt_order: list[int]) -> float:
    """
    Circular Kendall's τ for symmetric ring homomers.

    Takes max τ over all 2N canonical ring orderings (N forward + N backward
    cyclic rotations). Also tries all N starting seeds in greedy order
    by evaluating the prediction against every valid ring labelling.

    This is the correct metric when both ring direction and starting position
    are arbitrary (e.g., GroEL/GroES 7-mer rings).
    """
    N = len(gt_order)
    base = list(gt_order)
    rev  = list(reversed(gt_order))
    best = -1.0
    for i in range(N):
        best = max(best, kendall_tau_from_order(pred_order, base[i:] + base[:i]))
        best = max(best, kendall_tau_from_order(pred_order, rev[i:]  + rev[:i]))
    return best


def greedy_gnn_best_seed(
    model: AssemblyScorer,
    cdata: ComplexData,
    node_feats, edge_feats, adj,
    copy_indices: Optional[torch.Tensor] = None,
) -> list[int]:
    """For homomers: try all N seeds, return best-τ ordering. Embeds once for all seeds."""
    if cdata.complex_type != "homomer":
        return greedy_gnn(model, cdata, node_feats, edge_feats, adj, copy_indices)

    N = len(cdata.chain_ids)
    model.eval()
    with torch.no_grad():
        h = model.embed(node_feats, adj, edge_feats, copy_indices)

    best_tau = -2.0
    best_order = None
    for seed in range(N):
        remaining = [i for i in range(N) if i != seed]
        order = [seed]
        for _ in range(N - 1):
            if not remaining:
                break
            scores = score_step_gnn(model, cdata, order, remaining,
                                    node_feats, edge_feats, adj, h=h,
                                    copy_indices=copy_indices)
            best_c  = remaining[int(scores.argmax().item())]
            order.append(best_c)
            remaining.remove(best_c)
        tau = _circular_tau(order, cdata.gt_order)
        if tau > best_tau:
            best_tau  = tau
            best_order = order
    return best_order


def eval_tau(model, dataset, precomputed) -> dict:
    """
    Evaluate Kendall's τ for each complex.

    Homomers: circular τ over all 2N ring orderings, with best-seed greedy.
    Heteromers: standard τ against GT.
    """
    results = {}
    for cdata in dataset:
        node_feats, edge_feats, adj, copy_indices = precomputed[cdata.pdb_id]
        if cdata.complex_type == "homomer":
            pred = greedy_gnn_best_seed(model, cdata, node_feats, edge_feats, adj, copy_indices)
            tau  = _circular_tau(pred, cdata.gt_order)
        else:
            pred = greedy_gnn(model, cdata, node_feats, edge_feats, adj, copy_indices)
            tau  = kendall_tau_from_order(pred, cdata.gt_order)
        results[cdata.pdb_id] = tau
    return results


# ── GT probability ───────────────────────────────────────────────────────────

def eval_gt_prob(model, dataset, precomputed) -> float:
    model.eval()
    probs = []
    with torch.no_grad():
        for cdata in dataset:
            N            = len(cdata.chain_ids)
            order        = cdata.gt_order
            n_supervised = len(order)
            node_feats, edge_feats, adj, copy_indices = precomputed[cdata.pdb_id]
            h = model.embed(node_feats, adj, edge_feats, copy_indices)
            for step in range(1, n_supervised):
                assembled = order[:step]
                correct   = order[step]
                remaining = [i for i in range(N) if i not in assembled]
                if not remaining or correct not in remaining:
                    continue
                assembled_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
                assembled_mask[assembled] = True
                scores = model.score_candidates(h, edge_feats, assembled_mask, remaining)
                p = torch.softmax(scores, dim=0)
                correct_idx = remaining.index(correct)
                probs.append(p[correct_idx].item())
    return float(np.mean(probs)) if probs else 0.0


# ── Dashboard rendering ───────────────────────────────────────────────────────

def render_dashboard(epoch: int, history: dict, tau_per_complex: dict,
                     dataset: list[ComplexData]):
    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
    clr = {"ax": "#0d1117", "fg": "#e6edf3", "grid": "#21262d",
           "loss": "#f85149", "tau": "#58a6ff", "prob": "#3fb950",
           "bar": "#1f6feb"}

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(clr["ax"])
        ax.tick_params(colors=clr["fg"], labelsize=8)
        ax.xaxis.label.set_color(clr["fg"]); ax.yaxis.label.set_color(clr["fg"])
        ax.title.set_color(clr["fg"])
        for spine in ax.spines.values(): spine.set_edgecolor(clr["grid"])
        ax.grid(color=clr["grid"], alpha=0.5, linewidth=0.5)
        return ax

    E = list(range(1, len(history["loss"]) + 1))

    # A: Loss
    ax = _ax(gs[0, 0])
    ax.plot(E, history["loss"], color=clr["loss"], lw=1.5, label="Train loss")
    if history.get("val_loss"):
        ax.plot(E, [v if v is not None else float("nan") for v in history["val_loss"]],
                color="#ffa657", lw=1.5, ls="--", label="Val loss", alpha=0.8)
    ax.set_title(f"A. Loss (epoch {epoch})", fontsize=9)
    ax.legend(fontsize=7); ax.set_ylim(bottom=0)

    # B: τ + gt_prob
    ax  = _ax(gs[0, 1])
    ax2 = ax.twinx()
    ax2.set_facecolor(clr["ax"])
    ax2.tick_params(colors=clr["fg"], labelsize=8)
    ax2.yaxis.label.set_color(clr["fg"])
    for spine in ax2.spines.values(): spine.set_edgecolor(clr["grid"])

    taus = history.get("mean_tau", [])
    prbs = history.get("gt_prob", [])
    E2   = E[:len(taus)]
    E3   = E[:len(prbs)]
    ax.plot(E2, taus, color=clr["tau"], lw=2, label="Mean τ")
    ax2.plot(E3, prbs, color=clr["prob"], lw=1.5, ls="--", label="gt_prob", alpha=0.85)
    ax.set_ylabel("Kendall τ", color=clr["tau"]); ax2.set_ylabel("gt_prob", color=clr["prob"])
    ax.set_ylim(0, 1.05); ax2.set_ylim(0, 1.05)
    ax.set_title("B. τ + gt_prob", fontsize=9)
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=7)

    # C: Per-complex τ bar chart
    ax = _ax(gs[0, 2])
    if tau_per_complex:
        pdb_ids = list(tau_per_complex.keys())
        vals    = [tau_per_complex[p] for p in pdb_ids]
        bars    = ax.barh(pdb_ids, vals, color=clr["bar"], alpha=0.8)
        ax.set_xlim(0, 1.1)
        ax.axvline(np.mean(vals), color="white", ls="--", lw=1, alpha=0.5,
                   label=f"mean={np.mean(vals):.3f}")
        ax.legend(fontsize=7)
    ax.set_title("C. Per-complex τ", fontsize=9)

    # D: Model info / stats table
    ax = _ax(gs[1, 0])
    ax.axis("off")
    info_rows = [
        ["Epoch",        str(epoch)],
        ["Loss",         f"{history['loss'][-1]:.4f}"],
        ["Mean τ",       f"{np.mean(list(tau_per_complex.values())):.4f}" if tau_per_complex else "—"],
        ["gt_prob",      f"{history['gt_prob'][-1]:.4f}" if history.get('gt_prob') else "—"],
        ["Model",        "GAT 3-layer"],
        ["Node dim",     "646 (ESM+pLDDT+SASA)"],
        ["Edge dim",     "4"],
        ["Heads",        "4"],
        ["Hidden",       "128"],
        ["Device",       DEVICE],
    ]
    table = ax.table(cellText=info_rows, colLabels=["Metric", "Value"],
                     cellLoc="left", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#161b22"); cell.set_edgecolor(clr["grid"])
        cell.set_text_props(color=clr["fg"])
    ax.set_title("D. Model Info", fontsize=9, color=clr["fg"])

    # E: Assembly graph (2HHB or first complex)
    ax = _ax(gs[1, 1])
    ax.axis("off")
    _draw_complex_graph(ax, dataset, tau_per_complex, clr)
    ax.set_title("E. Assembly Graph", fontsize=9)

    # F: τ over epochs for homo/hetero split
    ax = _ax(gs[1, 2])
    homo_taus   = history.get("homo_tau", [])
    hetero_taus = history.get("hetero_tau", [])
    if homo_taus:
        ax.plot(E[:len(homo_taus)], homo_taus,
                color="#d2a679", lw=1.5, label="Homo")
    if hetero_taus:
        ax.plot(E[:len(hetero_taus)], hetero_taus,
                color="#bc8cff", lw=1.5, label="Hetero")
    ax.plot(E2, taus, color=clr["tau"], lw=2, label="All")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    ax.set_title("F. τ by Complex Type", fontsize=9)

    fig.suptitle(f"GNN Assembly Scorer — Epoch {epoch}", color=clr["fg"],
                 fontsize=13, fontweight="bold", y=0.98)
    out = EPOCH_DIR / f"epoch_{epoch:04d}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


def _draw_complex_graph(ax, dataset, tau_per_complex, clr):
    """Draw scored assembly graph for 2HHB or best complex."""
    target = None
    for cd in dataset:
        if "2HHB" in cd.pdb_id or "2hhb" in cd.pdb_id.lower():
            target = cd; break
    if target is None and dataset:
        target = dataset[0]
    if target is None:
        return

    N     = len(target.chain_ids)
    sasa  = target.sasa_matrix / (target.sasa_matrix.max() + 1e-8)
    order = target.gt_order

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1, N):
            if sasa[i, j] > 0.05:
                G.add_edge(i, j, weight=float(sasa[i, j]))

    pos = nx.spring_layout(G, seed=42)
    tau_val = tau_per_complex.get(target.pdb_id, 0.0)

    colors = []
    for i in range(N):
        rank = order.index(i) if i in order else N
        frac = rank / max(N - 1, 1)
        colors.append(plt.cm.RdYlGn(1 - frac))

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=400, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4,
                           edge_color=[clr["grid"]]*G.number_of_edges(), width=1.5)
    labels = {i: target.chain_ids[i] for i in range(N)}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color="white", ax=ax)
    ax.set_title(f"E. {target.pdb_id} (τ={tau_val:.3f})", fontsize=9,
                 color=clr["fg"])


# ── HTML dashboard ────────────────────────────────────────────────────────────

def write_html_dashboard(epoch_list: list[str]):
    epochs_js = json.dumps(epoch_list)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<title>GNN Training Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Courier New', monospace; }}
  header {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 24px; display: flex; align-items: center; gap: 12px; }}
  header h1 {{ font-size: 16px; }}
  .pill {{ background: #238636; color: #fff; border-radius: 12px; padding: 2px 10px; font-size: 11px; font-weight: 700; }}
  #status {{ margin-left: auto; font-size: 11px; color: #8b949e; }}
  .main {{ padding: 16px; }}
  #epoch-img {{ width: 100%; border: 1px solid #30363d; border-radius: 6px; margin-top: 10px; }}
  .epoch-nav {{ display: flex; align-items: center; gap: 8px; margin-top: 8px; }}
  .epoch-nav button {{ background: #21262d; border: 1px solid #30363d; color: #e6edf3; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .epoch-nav button:hover {{ background: #30363d; }}
  #epoch-label {{ font-size: 12px; color: #8b949e; flex: 1; text-align: center; }}
  #epoch-slider {{ flex: 3; accent-color: #1f6feb; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 16px; }}
  .metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; text-align: center; }}
  .metric-card .label {{ font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }}
  .metric-card .value {{ font-size: 22px; font-weight: 700; color: #58a6ff; margin-top: 4px; }}
  #log-div {{ background: #010409; border: 1px solid #21262d; border-radius: 4px; padding: 8px; height: 160px; overflow-y: auto; font-size: 11px; line-height: 1.6; margin-top: 16px; }}
  footer {{ background: #161b22; border-top: 1px solid #30363d; padding: 8px 24px; font-size: 11px; color: #8b949e; }}
</style>
</head>
<body>
<header>
  <h1>Protein Assembly — GNN (GAT) Training</h1>
  <span class="pill">AssemblyScorer</span>
  <span id="status">auto-refresh 5s</span>
</header>
<div class="main">
  <div class="metrics" id="metrics-grid">
    <div class="metric-card"><div class="label">Epoch</div><div class="value" id="m-epoch">—</div></div>
    <div class="metric-card"><div class="label">Loss</div><div class="value" id="m-loss">—</div></div>
    <div class="metric-card"><div class="label">Mean τ</div><div class="value" id="m-tau">—</div></div>
    <div class="metric-card"><div class="label">gt_prob</div><div class="value" id="m-prob">—</div></div>
  </div>

  <div class="epoch-nav" style="margin-top:12px;">
    <button onclick="stepEpoch(-1)">&#9664;</button>
    <input type="range" id="epoch-slider" min="0" value="0" step="1" oninput="sliderChange(this.value)">
    <span id="epoch-label">—</span>
    <button onclick="stepEpoch(1)">&#9654;</button>
    <button onclick="togglePlay()" id="play-btn">&#9654; Play</button>
    <button onclick="jumpLatest()">&#9654;&#9654; Latest</button>
  </div>
  <img id="epoch-img" src="" alt="epoch snapshot">

  <div id="log-div"></div>
</div>
<footer>results/gnn_training/ &nbsp;|&nbsp; <span id="footer-time"></span></footer>

<script>
const EPOCHS = {epochs_js};
let cur = EPOCHS.length - 1;
let playing = false, playTimer = null;

const slider = document.getElementById('epoch-slider');
slider.max = EPOCHS.length - 1;
slider.value = cur;

function showEpoch(idx) {{
  cur = Math.max(0, Math.min(EPOCHS.length-1, idx));
  const name = EPOCHS[cur];
  document.getElementById('epoch-img').src = `epochs/${{name}}.png?t=${{Date.now()}}`;
  document.getElementById('epoch-label').textContent = `epoch ${{name.split('_')[1]}}`;
  slider.value = cur;
}}

function stepEpoch(d) {{ showEpoch(cur + d); }}
function sliderChange(v) {{ showEpoch(parseInt(v)); }}
function jumpLatest() {{ showEpoch(EPOCHS.length - 1); }}

function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {{
    if (cur >= EPOCHS.length - 1) cur = 0;
    playTimer = setInterval(() => {{
      if (cur >= EPOCHS.length - 1) {{ playing = false; clearInterval(playTimer); return; }}
      showEpoch(cur + 1);
    }}, 350);
  }} else {{ clearInterval(playTimer); }}
}}

showEpoch(cur);

async function loadLog() {{
  try {{
    const r = await fetch(`training_log.json?t=${{Date.now()}}`);
    if (!r.ok) return;
    const data = await r.json();
    if (!data.length) return;
    const last = data[data.length - 1];
    document.getElementById('m-epoch').textContent = last.epoch || '—';
    document.getElementById('m-loss').textContent  = last.loss  != null ? last.loss.toFixed(4)  : '—';
    document.getElementById('m-tau').textContent   = last.mean_tau != null ? last.mean_tau.toFixed(4) : '—';
    document.getElementById('m-prob').textContent  = last.gt_prob != null ? last.gt_prob.toFixed(4) : '—';

    const logDiv = document.getElementById('log-div');
    logDiv.innerHTML = data.slice(-25).reverse().map(d =>
      `<div style="color:#8b949e"><span style="color:#30363d">ep ${{String(d.epoch).padStart(4,'0')}}</span>  loss=<span style="color:#f85149">${{d.loss.toFixed(4)}}</span>  τ=<span style="color:#58a6ff">${{d.mean_tau != null ? d.mean_tau.toFixed(4) : '—'}}</span>  p=<span style="color:#3fb950">${{d.gt_prob != null ? d.gt_prob.toFixed(4) : '—'}}</span></div>`
    ).join('');

    document.getElementById('footer-time').textContent = new Date().toLocaleTimeString();
  }} catch(e) {{}}
}}

loadLog();
setInterval(loadLog, 5000);
</script>
</body>
</html>"""
    (GNN_DIR / "dashboard.html").write_text(html)


# ── GO matrix population ─────────────────────────────────────────────────────

def populate_go_matrices(dataset: list[ComplexData], cache_dir: Path) -> None:
    """
    Populate cdata.go_matrix for every complex using FABLEBridge (logit cosine).
    Caches per-complex to data/go_cache/{pdb_id}_go.npy — subsequent runs are instant.
    Silently skips if FABLEBridge model files are unavailable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    bridge = None

    for cdata in dataset:
        N = len(cdata.chain_ids)
        cache_path = cache_dir / f"{cdata.pdb_id}_go.npy"

        # Load from cache if shape matches
        if cache_path.exists():
            try:
                mat = np.load(str(cache_path)).astype(np.float32)
                if mat.shape == (N, N):
                    cdata.go_matrix = mat
                    continue
            except Exception:
                pass

        # Lazy-load bridge on first miss
        if bridge is None:
            try:
                from src.protfunc_bridge.go_coherence import FABLEBridge, pairwise_go_matrix as _pwgo
                bridge = FABLEBridge()
                print("[go] FABLEBridge loaded for GO matrix population")
            except Exception as e:
                print(f"[go] FABLEBridge unavailable ({e}) — go_matrix will be zeros")
                return  # All remaining complexes stay zero

        try:
            mat = _pwgo(cdata.sequences, bridge).astype(np.float32)
            np.save(str(cache_path), mat)
            cdata.go_matrix = mat
        except Exception as e:
            print(f"  [{cdata.pdb_id}] GO matrix failed: {e}")


# ── Temperature calibration ───────────────────────────────────────────────────

def calibrate_temperature(
    model: AssemblyScorer,
    dataset: list[ComplexData],
    precomputed: dict,
    T_lo: float = 0.2,
    T_hi: float = 8.0,
    n_steps: int = 120,
) -> float:
    """
    Find scalar temperature T* that minimises mean NLL of GT next subunit
    across all assembly steps in dataset.  Returns T*.
    Uses ternary search (unimodal NLL over T).
    """
    model.eval()

    # Collect (raw_scores, gt_idx) for every step — embed once per complex
    step_data: list[tuple[torch.Tensor, int]] = []
    with torch.no_grad():
        for cdata in dataset:
            if cdata.pdb_id not in precomputed:
                continue
            nf, ef, adj, ci = precomputed[cdata.pdb_id]
            N            = len(cdata.chain_ids)
            gt           = cdata.gt_order
            n_supervised = len(gt)
            h  = model.embed(nf, adj, ef, ci)
            for step in range(1, n_supervised):
                assembled = gt[:step]
                correct   = gt[step]
                remaining = [i for i in range(N) if i not in assembled]
                if not remaining or correct not in remaining:
                    continue
                am = torch.zeros(N, dtype=torch.bool, device=DEVICE)
                am[assembled] = True
                scores = model.score_candidates(h, ef, am, remaining).cpu()
                step_data.append((scores, remaining.index(correct)))

    if not step_data:
        return 1.0

    def nll(T: float) -> float:
        total = 0.0
        for scores, gt_idx in step_data:
            total -= float(F.log_softmax(scores / T, dim=0)[gt_idx].item())
        return total / len(step_data)

    # Ternary search (NLL(T) is convex in T for fixed logits)
    lo, hi = T_lo, T_hi
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    best_T = (lo + hi) / 2.0
    best_nll = nll(best_T)
    baseline = nll(1.0)
    print(f"[calib] T*={best_T:.3f}  NLL {baseline:.4f}→{best_nll:.4f}  "
          f"(Δ={baseline-best_nll:+.4f})")
    return float(best_T)


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_experimental_v2", action="store_true")
    parser.add_argument("--use_labeled_15k", action="store_true",
                        help="Add labeled_15k complexes to training set")
    parser.add_argument("--labeled_15k_max", type=int, default=0,
                        help="Max labeled_15k complexes to load (0=all)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping: halt if gt_prob doesn't improve for N epochs")
    parser.add_argument("--anchor_mode", choices=["max", "pooled"], default="pooled",
                        help="Anchor aggregation: pooled=context-gated (Step 11), max=original")
    parser.add_argument("--sparse_k", type=int, default=0,
                        help="Top-k neighbour pruning for sparse GATv2 (Step 12); 0=no pruning")
    args, _ = parser.parse_known_args()

    print(f"[gnn] device={DEVICE}")

    # Marsh 2013 benchmark — experimental GT, primary eval set
    marsh_dataset = load_dataset_from_alpha(fetch_plddt=True)
    if not marsh_dataset:
        print("[gnn] no Marsh 2013 data — aborting")
        return

    # Expanded PDB dataset — SASA-greedy GT, training only
    expanded_dataset = load_expanded_dataset()

    # Experimental GT v2 — optional, high-quality training + eval supplement
    v2_dataset = load_experimental_v2_dataset() if args.use_experimental_v2 else []
    if v2_dataset:
        print(f"[gnn] using experimental_v2: {len(v2_dataset)} complexes added to train+eval")

    # Optional: labeled_15k large corpus
    labeled_15k_dataset = []
    if args.use_labeled_15k:
        labeled_15k_dataset = load_labeled_15k_dataset("train", args.labeled_15k_max)
        print(f"[gnn] labeled_15k: {len(labeled_15k_dataset)} unique complexes after dedup")

    # Full training set: Marsh 2013 + expanded + (optional) experimental v2 + (optional) labeled_15k
    # Eval (τ) uses Marsh 2013 + (optional) experimental v2 (both have experimental GT)
    train_dataset = marsh_dataset + expanded_dataset + v2_dataset + labeled_15k_dataset
    eval_dataset  = marsh_dataset + v2_dataset

    n_marsh    = len(marsh_dataset)
    n_expanded = len(expanded_dataset)
    n_v2       = len(v2_dataset)
    n_15k      = len(labeled_15k_dataset)
    print(f"[gnn] train={len(train_dataset)} "
          f"(marsh={n_marsh}, expanded={n_expanded}, v2={n_v2}, labeled_15k={n_15k}), "
          f"eval={len(eval_dataset)}")

    # Populate GO matrices (FABLEBridge logit cosine) — cached to data/go_cache/
    # Runs on all training complexes; skips if model files unavailable
    parser2 = argparse.ArgumentParser(add_help=False)
    parser2.add_argument("--no_go", action="store_true",
                         help="Skip GO matrix population (faster startup)")
    args2, _ = parser2.parse_known_args()
    if not args2.no_go:
        print("[go] populating GO matrices...")
        populate_go_matrices(train_dataset, GO_CACHE)
        print(f"[go] done (cache: {GO_CACHE})")

    esm_dim  = marsh_dataset[0].esm_embeddings.shape[1]
    node_dim = esm_dim + PLDDT_NODE_DIM + 1 + GEOM_DIM  # ESM + pLDDT (5d) + SASA (1d) + geom (9d)
    edge_dim = 5  # ppi + esm_cos + go_cos + sasa + contact_density

    print(f"[gnn] {MODEL_VERSION} node_dim={node_dim} "
          f"(esm={esm_dim} + plddt={PLDDT_NODE_DIM} + sasa=1 + geom={GEOM_DIM}) "
          f"edge_dim={edge_dim}")

    GEOM_CACHE.mkdir(parents=True, exist_ok=True)

    # Precompute all tensors (downloads AF PDBs for geom features on first run)
    print("[gnn] precomputing features (may download AlphaFold PDBs on first run)…")
    precomputed = {cd.pdb_id: precompute(cd, esm_dim) for cd in train_dataset}

    print(f"[gnn] anchor_mode={args.anchor_mode} sparse_k={args.sparse_k}")
    model = AssemblyScorer(
        node_dim=node_dim, edge_dim=edge_dim, hidden=128, n_layers=3, heads=4, dropout=0.1,
        anchor_mode=args.anchor_mode, sparse_k=args.sparse_k,
    ).to(DEVICE)
    print(f"[gnn] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCHS = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    EVAL_EVERY = 5
    SAVE_EVERY = 25

    history = {
        "loss": [], "val_loss": [], "mean_tau": [], "gt_prob": [],
        "homo_tau": [], "hetero_tau": [],
        "mean_top1": [], "mean_confidence": [], "mean_early_tau": [],
    }
    best_gt_prob = -1.0
    best_epoch   = 1
    patience_ctr = 0
    log_entries  = []
    epoch_files  = []

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = gnn_ranking_loss(model, train_dataset, precomputed)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        history["loss"].append(loss_val)
        history["val_loss"].append(None)

        # Eval on Marsh 2013 only (experimental GT)
        tau_per_complex: dict = {}
        gt_prob_val: Optional[float] = None

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            # Full multi-metric evaluation
            eval_results = evaluate_dataset(model, eval_dataset, precomputed, device=DEVICE)
            summary = eval_results.pop("_summary")
            mean_tau      = summary["mean_tau"]
            gt_prob_val   = eval_gt_prob(model, eval_dataset, precomputed)
            tau_per_complex = {k: v["tau"] for k, v in eval_results.items()}

            homo_taus   = [v for k, v in tau_per_complex.items()
                           if BENCHMARK.get(k, {}).get("type") == "homomer"]
            hetero_taus = [v for k, v in tau_per_complex.items()
                           if BENCHMARK.get(k, {}).get("type") == "heteromer"]

            history["mean_tau"].append(mean_tau)
            history["gt_prob"].append(gt_prob_val)
            history["homo_tau"].append(float(np.mean(homo_taus)) if homo_taus else None)
            history["hetero_tau"].append(float(np.mean(hetero_taus)) if hetero_taus else None)
            history["mean_top1"].append(summary["mean_top1"])
            history["mean_confidence"].append(summary["mean_confidence"])
            history["mean_early_tau"].append(summary["mean_early_tau"])

            if gt_prob_val > best_gt_prob:
                best_gt_prob = gt_prob_val
                best_epoch   = epoch
                patience_ctr = 0
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "node_dim":    node_dim,
                    "edge_dim":    edge_dim,
                    "version":     MODEL_VERSION,
                    "epoch":       epoch,
                    "gt_prob":     gt_prob_val,
                    "mean_tau":    mean_tau,
                    "anchor_mode": args.anchor_mode,
                    "sparse_k":    args.sparse_k,
                }
                torch.save(ckpt, GNN_DIR / "best_gnn.pt")
                torch.save(ckpt, GNN_DIR / "best_v4.pt")
            else:
                patience_ctr += EVAL_EVERY

            print(f"ep {epoch:4d} | loss={loss_val:.4f} | τ={mean_tau:.4f} | "
                  f"gt_prob={gt_prob_val:.4f} | top1={summary['mean_top1']:.4f} | "
                  f"early_τ={summary['mean_early_tau']:.4f} | best_gtp={best_gt_prob:.4f}@{best_epoch}")

            if patience_ctr >= args.patience:
                print(f"[gnn] early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break
        else:
            for key in ("mean_tau", "gt_prob", "homo_tau", "hetero_tau",
                        "mean_top1", "mean_confidence", "mean_early_tau"):
                history[key].append(history[key][-1] if history[key] else None)
            if not tau_per_complex and epoch > 1:
                tau_per_complex = {cd.pdb_id: history["mean_tau"][-2]
                                   for cd in eval_dataset} if len(history["mean_tau"]) > 1 else {}

        entry = {
            "epoch":          epoch,
            "loss":           loss_val,
            "val_loss":       None,
            "mean_tau":       history["mean_tau"][-1],
            "gt_prob":        history["gt_prob"][-1],
            "homo_tau":       history["homo_tau"][-1],
            "hetero_tau":     history["hetero_tau"][-1],
            "mean_top1":      history["mean_top1"][-1],
            "mean_confidence":history["mean_confidence"][-1],
            "mean_early_tau": history["mean_early_tau"][-1],
        }
        log_entries.append(entry)
        with open(GNN_DIR / "training_log.json", "w") as f:
            json.dump(log_entries, f, default=lambda x: float(x) if x is not None else None)

        if epoch % SAVE_EVERY == 0 or epoch == 1:
            render_dashboard(epoch, history, tau_per_complex, eval_dataset)
            fname = f"epoch_{epoch:04d}"
            epoch_files.append(fname)
            write_html_dashboard(epoch_files)

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "version":  MODEL_VERSION,
        "epoch": EPOCHS,
        "gt_prob": history["gt_prob"][-1],
        "mean_tau": history["mean_tau"][-1],
    }, GNN_DIR / "final_gnn.pt")
    print(f"\n[gnn] done. best gt_prob={best_gt_prob:.4f} @ epoch {best_epoch}")

    # Temperature calibration on eval set using best checkpoint
    print("[calib] loading best checkpoint for temperature calibration...")
    best_state = torch.load(str(GNN_DIR / "best_gnn.pt"), map_location=DEVICE)
    model.load_state_dict(best_state["model_state_dict"])
    best_T = calibrate_temperature(model, eval_dataset, precomputed)
    temp_path = GNN_DIR / "temperature.json"
    json.dump({"temperature": best_T, "epoch": best_epoch, "gt_prob": best_gt_prob},
              open(temp_path, "w"), indent=2)
    print(f"[calib] saved → {temp_path}")
    print(f"[gnn] dashboard: results/gnn_training/dashboard.html")


if __name__ == "__main__":
    main()
