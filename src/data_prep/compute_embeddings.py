"""
ESM-2 embedding extraction for all subunits.
MPS-optimized for Apple Silicon (M4 Mac Air).

Requires macOS Sequoia + PyTorch ≥ 2.5 for accurate MPS results (see literature note).

Usage:
    python compute_embeddings.py --dataset marsh2013 --model esm2_t6_8M_UR50D
    python compute_embeddings.py --dataset marsh2013 --model esm2_t30_150M_UR50D
"""

import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
EMBED_DIR = DATA_DIR / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)


ESM_LAYER_MAP = {
    "esm2_t6_8M_UR50D":    6,
    "esm2_t12_35M_UR50D":  12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        # Verify MPS numerical accuracy (Sequoia + PyTorch 2.5 required)
        x = torch.tensor([1.0, 2.0, 3.0], device="mps")
        y = torch.tensor([1.0, 2.0, 3.0])
        if torch.allclose(x.cpu(), y, atol=1e-4):
            print("[ESM] Using MPS (Apple Silicon)")
            return torch.device("mps")
        else:
            print("[ESM] MPS numerical check failed — using CPU. Upgrade to macOS Sequoia + PyTorch 2.5")
            return torch.device("cpu")
    return torch.device("cpu")


def load_esm_model(model_name: str, device: torch.device):
    import esm as esm_lib
    model, alphabet = esm_lib.pretrained.__dict__[model_name]()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter, ESM_LAYER_MAP[model_name]


@torch.no_grad()
def embed_sequences(
    sequences: dict[str, str],  # {name: sequence}
    model_name: str = "esm2_t6_8M_UR50D",
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> dict[str, np.ndarray]:
    """
    Compute ESM-2 mean-pool embeddings for all sequences.
    Returns {name: embedding_array (D,)}.
    """
    if device is None:
        device = get_device()

    model, batch_converter, repr_layer = load_esm_model(model_name, device)
    items = list(sequences.items())
    embeddings: dict[str, np.ndarray] = {}

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        _, _, tokens = batch_converter([(n, s) for n, s in batch])
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]  # (B, L+2, D)
        for j, (name, seq) in enumerate(batch):
            # Mean pool over residue tokens (exclude BOS/EOS)
            emb = reps[j, 1 : len(seq) + 1].mean(0).cpu().numpy()
            embeddings[name] = emb
        if (i // batch_size + 1) % 5 == 0:
            print(f"  {i + len(batch)}/{len(items)} sequences embedded")

    return embeddings


def cosine_similarity_matrix(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Build pairwise ESM cosine similarity matrix."""
    names = sorted(embeddings.keys())
    mat = np.stack([embeddings[n] for n in names])  # (N, D)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norm = mat / (norms + 1e-8)
    sim = mat_norm @ mat_norm.T
    return sim.clip(-1, 1), names


def compute_and_save(
    manifest_path: str,
    model_name: str = "esm2_t6_8M_UR50D",
    force: bool = False,
):
    """Load manifest, compute embeddings, save .npy + cosine matrix."""
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    out_dir = EMBED_DIR / manifest_path.parent.name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all sequences across all complexes
    all_seqs: dict[str, str] = {}
    for pdb_id, meta in manifest.items():
        for chain, seq in meta.get("sequences", {}).items():
            key = f"{pdb_id}_{chain}"
            all_seqs[key] = seq

    embed_file = out_dir / "embeddings.npz"
    if embed_file.exists() and not force:
        print(f"Embeddings already exist: {embed_file}")
        return

    print(f"Computing {model_name} embeddings for {len(all_seqs)} sequences...")
    embeddings = embed_sequences(all_seqs, model_name=model_name)

    # Save individual embeddings
    np.savez(str(embed_file), **embeddings)
    print(f"  saved → {embed_file}")

    # Save cosine similarity matrix
    sim_matrix, names = cosine_similarity_matrix(embeddings)
    np.save(str(out_dir / "cosine_matrix.npy"), sim_matrix)
    with open(out_dir / "names.json", "w") as f:
        json.dump(names, f)
    print(f"  cosine matrix {sim_matrix.shape} saved → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="marsh2013",
                        choices=["marsh2013", "npc"])
    parser.add_argument("--model", default="esm2_t6_8M_UR50D",
                        choices=list(ESM_LAYER_MAP.keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = DATA_DIR / "complexes" / args.dataset / "manifest.json"
    if not manifest.exists():
        print(f"Run fetch_complexes.py --dataset {args.dataset} first")
        raise SystemExit(1)

    compute_and_save(str(manifest), model_name=args.model, force=args.force)
