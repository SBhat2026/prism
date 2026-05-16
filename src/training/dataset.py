"""
Training dataset for assembly order prediction.

Each training example is one assembly step:
  (assembled_set, candidate, is_correct_next, feature_vector)

Feature vector per candidate:
  [max_sasa_norm, max_esm_cos, max_go_cos, max_ppi,
   mean_sasa_norm, mean_esm_cos, mean_go_cos, mean_ppi]  ← 8 scalars

Also stores full matrices + ESM embeddings for GNN training.
"""

import json
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent


@dataclass
class AssemblyExample:
    """One step of one complex's ground-truth assembly."""
    pdb_id: str
    complex_type: str          # "homomer" | "heteromer"
    step: int                  # 0-indexed
    assembled: list[int]       # indices in complex already assembled
    candidate: int             # index being considered
    is_correct: bool           # is this the ground-truth next?
    features: np.ndarray       # (8,) scalar features for linear scorer
    n_subunits: int


@dataclass
class ComplexData:
    """All precomputed matrices for one complex."""
    pdb_id: str
    complex_type: str
    chain_ids: list[str]
    sequences: list[str]
    gt_order: list[int]
    sasa_matrix: np.ndarray    # (N, N) normalized to [0,1]
    esm_matrix: np.ndarray     # (N, N) cosine similarity
    go_matrix: np.ndarray      # (N, N) cosine similarity on logits
    ppi_matrix: np.ndarray     # (N, N) STRING scores
    esm_embeddings: np.ndarray # (N, D) raw ESM embeddings
    plddt: Optional[np.ndarray] = None  # (N, 5) AlphaFold pLDDT node features
    gt_source: str = "experimental"     # "experimental" | "sasa_greedy"
    subcomplexes: Optional[list] = None  # known obligate subcomplex groups, e.g. [[0,1],[2,3]] for Hb
    uniprot_ids: Optional[list] = None  # (N,) UniProt IDs per chain (empty str if unknown)
    pdb_path: Optional[str] = None      # path to full-complex PDB (Marsh only)


def _max_mean(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    return (float(arr.max()) if len(arr) else 0.0,
            float(arr.mean()) if len(arr) else 0.0)


def build_candidate_features(
    candidate: int,
    assembled: list[int],
    sasa: np.ndarray,
    esm: np.ndarray,
    go: np.ndarray,
    ppi: np.ndarray,
) -> np.ndarray:
    """8-dim feature vector for one (candidate, assembled_set) pair."""
    if not assembled:
        return np.zeros(8, dtype=np.float32)
    row = candidate
    cols = assembled
    sasa_max, sasa_mean = _max_mean(sasa[row, cols].tolist())
    esm_max,  esm_mean  = _max_mean(esm[row,  cols].tolist())
    go_max,   go_mean   = _max_mean(go[row,   cols].tolist())
    ppi_max,  ppi_mean  = _max_mean(ppi[row,  cols].tolist())
    return np.array([sasa_max, esm_max, go_max, ppi_max,
                     sasa_mean, esm_mean, go_mean, ppi_mean], dtype=np.float32)


def build_examples(cdata: ComplexData) -> list[AssemblyExample]:
    """
    Build all (assembled, candidate, is_correct) triples from ground-truth order.
    For each step t, there are (N - t) candidates, 1 correct.
    """
    N = len(cdata.chain_ids)
    order = cdata.gt_order
    sasa  = cdata.sasa_matrix / (cdata.sasa_matrix.max() + 1e-8)

    examples = []
    for step in range(1, N):
        assembled = order[:step]
        correct   = order[step]
        remaining = [i for i in range(N) if i not in assembled]

        for cand in remaining:
            feats = build_candidate_features(
                cand, assembled, sasa, cdata.esm_matrix, cdata.go_matrix, cdata.ppi_matrix
            )
            examples.append(AssemblyExample(
                pdb_id=cdata.pdb_id,
                complex_type=cdata.complex_type,
                step=step,
                assembled=assembled.copy(),
                candidate=cand,
                is_correct=(cand == correct),
                features=feats,
                n_subunits=N,
            ))
    return examples


def load_all_complex_data(
    benchmark: dict,
    marsh_dir: Path,
    embed_8m_dir: Path,
    embed_150m_dir: Optional[Path] = None,
    go_bridge=None,
) -> list[ComplexData]:
    """
    Load all ComplexData objects for benchmark complexes.
    Uses 150M ESM if available, else 8M.
    """
    with open(marsh_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Choose embedding source
    use_150m = embed_150m_dir is not None and embed_150m_dir.exists()
    embed_dir = embed_150m_dir if use_150m else embed_8m_dir
    esm_dim   = 640 if use_150m else 320
    embed_data = np.load(str(embed_dir / "embeddings.npz"))

    dataset = []
    for pdb_id, meta in benchmark.items():
        chains  = meta["chains"]
        seqs    = [manifest.get(pdb_id, {}).get("sequences", {}).get(c, "M"*50) for c in chains]
        N       = len(chains)

        # ESM embeddings (N, D)
        embs = np.array([
            embed_data.get(f"{pdb_id}_{c}", np.zeros(esm_dim))
            for c in chains
        ])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_n = embs / (norms + 1e-8)
        esm_mat = (embs_n @ embs_n.T).clip(-1, 1)

        # SASA (loaded from pre-computed alpha results if available, else zeros)
        sasa_path = ROOT / "results" / "alpha" / "alpha_results.json"
        sasa_mat  = np.zeros((N, N))
        if sasa_path.exists():
            with open(sasa_path) as f:
                ar = json.load(f)
            # Find matching complex
            for cname, conds in ar.get("per_complex", {}).items():
                if pdb_id in cname:
                    break

        # Recompute SASA using GROMACS (may take time; use cached if possible)
        from src.gromacs.extract_sasa import GromacsSASA
        pdb_path = str(marsh_dir / f"{pdb_id}.pdb")
        sasa_calc = GromacsSASA()
        for i, ci in enumerate(chains):
            for j, cj in enumerate(chains):
                if j <= i:
                    continue
                try:
                    bsa = sasa_calc.interface_buried_area(pdb_path, ci, cj)
                    sasa_mat[i, j] = bsa
                    sasa_mat[j, i] = bsa
                except Exception:
                    pass

        # GO matrix
        go_mat = np.zeros((N, N))
        if go_bridge is not None:
            try:
                vecs = [go_bridge.get_logits(s) for s in seqs]
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            a, b = vecs[i], vecs[j]
                            na, nb = np.linalg.norm(a), np.linalg.norm(b)
                            go_mat[i, j] = float(np.dot(a, b) / (na * nb + 1e-8))
            except Exception:
                pass

        dataset.append(ComplexData(
            pdb_id=pdb_id,
            complex_type=meta["type"],
            chain_ids=chains,
            sequences=seqs,
            gt_order=meta["gt_order"],
            sasa_matrix=sasa_mat,
            esm_matrix=esm_mat,
            go_matrix=go_mat,
            ppi_matrix=np.zeros((N, N)),
            esm_embeddings=embs,
        ))
        print(f"  loaded {pdb_id}: sasa_max={sasa_mat.max():.2f} esm_max={esm_mat.max():.3f}")

    return dataset


def to_torch_batch(examples: list[AssemblyExample]) -> tuple:
    """Stack features + labels for linear scorer training."""
    feats  = torch.tensor(np.stack([e.features for e in examples]), dtype=torch.float32)
    labels = torch.tensor([1.0 if e.is_correct else 0.0 for e in examples], dtype=torch.float32)
    return feats, labels


def build_step_groups(examples: list[AssemblyExample]) -> list[list[AssemblyExample]]:
    """
    Group examples by (pdb_id, step) for ranking loss:
    each group is all candidates at one assembly step.
    """
    groups: dict[tuple, list] = {}
    for ex in examples:
        key = (ex.pdb_id, ex.step)
        if key not in groups:
            groups[key] = []
        groups[key].append(ex)
    return list(groups.values())
