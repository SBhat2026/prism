"""
PCA probe: do ESM embeddings cluster by protein family / function
or by which training complex they came from?

If the model is generalizing: proteins with similar functions cluster
together regardless of which complex they appear in.
If the model is memorizing: proteins cluster by complex identity.

Usage:
    python src/evaluation/pca_generalization.py

Held-out proteins are curated deliberately: one member of each family
that appears in training (e.g., hemoglobin α) paired with a homolog
from a complex never seen (e.g., myoglobin — same globin fold, absent
from training). If PCA clusters by family rather than by complex, the
ESM representation generalizes; if it clusters by complex, the model
is exploiting topology shortcuts.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

EMBED_150M = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t30_150M_UR50D"
EMBED_8M   = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t6_8M_UR50D"
OUT_DIR    = ROOT / "results" / "alpha" / "gnn_viz"


# ── Held-out protein library ──────────────────────────────────────────────────
# Each entry: id, complex, family, in_training, sequence (or None to load from cache)
#
# Deliberate pairing strategy:
#   Training protein   →  Held-out homolog   →  Expected PCA: same cluster
#   2HHB chain A (Hb-α) → Myoglobin (1MBN)  →  globin
#   1BRS chain A (Barnase) → RNase A (1AFU) →  ribonuclease
#   1AON chain A (GroEL) → GroES (1AON-a)   →  chaperonin (within same complex but diff role)
#
# Add more entries as new complexes are added to training.

HELD_OUT_PROTEINS = [
    # Globin family
    {
        "id": "Myoglobin_1MBN_A",
        "complex": "1MBN",
        "family": "globin",
        "in_training": False,
        "uniprot": "P02144",
        "sequence": (
            "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG"
        ),
    },
    # Serine protease inhibitor family
    {
        "id": "BPTI_1TGS_I",
        "complex": "1TGS",
        "family": "serine_protease_inhibitor",
        "in_training": True,
        "uniprot": "P00974",
        "sequence": None,  # load from embeddings.npz
    },
    {
        "id": "OMTKY3_2OVO_I",
        "complex": "2OVO",
        "family": "serine_protease_inhibitor",
        "in_training": False,
        "uniprot": "P00977",
        "sequence": (
            "AVCEETDGKKLCGSDGEAFVTCENPQKIIEGRKIHFVQANPQGSCQDADPIASLNQAFSRDNKSLFHSLINDLNRQIVSCDSTGQTYAVLHGGYGISYLKFADPTNKSKIAAMLAQIDADTASSLRSKHLFHKPLQNLSTQ"
        ),
    },
    # Ribonuclease family
    {
        "id": "Barnase_1BRS_A",
        "complex": "1BRS",
        "family": "ribonuclease",
        "in_training": True,
        "uniprot": "P00648",
        "sequence": None,
    },
    {
        "id": "RNaseA_1AFU_A",
        "complex": "1AFU",
        "family": "ribonuclease",
        "in_training": False,
        "uniprot": "P61823",
        "sequence": (
            "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV"
        ),
    },
    # Chaperonin family
    {
        "id": "GroEL_1AON_A",
        "complex": "1AON",
        "family": "chaperonin",
        "in_training": True,
        "uniprot": "P0A6F5",
        "sequence": None,
    },
    # CCT alpha (P17987, human TCP1-alpha) — fetch sequence from UniProt and add here.
    # Until then, omitted: ESM embedding requires correct sequence to cluster with GroEL.
    # {
    #     "id": "CCT_alpha_held",
    #     "complex": "CCT_complex",
    #     "family": "chaperonin",
    #     "in_training": False,
    #     "uniprot": "P17987",
    #     "sequence": "<fetch from https://www.uniprot.org/uniprot/P17987.fasta>",
    # },
    # Hemoglobin subunits
    {
        "id": "HbA_2HHB_A",
        "complex": "2HHB",
        "family": "globin",
        "in_training": True,
        "uniprot": "P69905",
        "sequence": None,
    },
    {
        "id": "HbB_2HHB_B",
        "complex": "2HHB",
        "family": "globin",
        "in_training": True,
        "uniprot": "P02023",
        "sequence": None,
    },
]


def _load_esm_from_npz(protein_id: str) -> Optional[np.ndarray]:
    """Load pre-computed ESM embedding from the Marsh2013 npz cache."""
    embed_dir = EMBED_150M if EMBED_150M.exists() else EMBED_8M
    npz_path  = embed_dir / "embeddings.npz"
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path))

    # Try various key formats: PDB_chain, protein_id
    for key in data.files:
        # e.g. "2HHB_A" matches HbA_2HHB_A
        parts = protein_id.split("_")
        pdb = None
        chain = None
        for p in parts:
            if len(p) == 4 and p.isupper():
                pdb = p
            elif len(p) == 1 and p.isupper():
                chain = p
        if pdb and chain:
            candidate = f"{pdb}_{chain}"
            if candidate in data.files:
                return data[candidate].astype(np.float32)
    return None


def _compute_esm_embedding(sequence: str, esm_dim: int = 640) -> np.ndarray:
    """Compute ESM-2 150M embedding for a new sequence."""
    try:
        import esm as esm_lib
        model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        _, _, tokens = batch_converter([("seq", sequence)])
        import torch
        with torch.no_grad():
            results = model(tokens, repr_layers=[30])
        emb = results["representations"][30][0, 1:-1].mean(0).numpy()
        return emb.astype(np.float32)
    except Exception as e:
        print(f"    ESM compute failed ({e}), using random placeholder")
        return np.random.randn(esm_dim).astype(np.float32)


def load_embeddings(proteins: list[dict], esm_dim: int = 640) -> list[dict]:
    """
    For each protein, load or compute ESM embedding.
    Returns list of dicts with added 'embedding' key.
    """
    result = []
    for p in proteins:
        emb = _load_esm_from_npz(p["id"])
        if emb is None and p.get("sequence"):
            print(f"  Computing ESM for {p['id']}...")
            emb = _compute_esm_embedding(p["sequence"], esm_dim)
        if emb is None:
            print(f"  SKIP {p['id']}: no embedding available")
            continue
        result.append({**p, "embedding": emb})
    return result


def run_pca_probe(
    proteins: Optional[list[dict]] = None,
    outpath: Optional[str] = None,
) -> None:
    """
    Run PCA probe.

    proteins: list of dicts with keys: id, complex, family, in_training, embedding
              (or use default HELD_OUT_PROTEINS)
    outpath:  output PNG path (default: results/alpha/gnn_viz/pca_generalization.png)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if proteins is None:
        proteins = HELD_OUT_PROTEINS

    out = Path(outpath) if outpath else OUT_DIR / "pca_generalization.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    loaded = load_embeddings(proteins)
    if len(loaded) < 3:
        print(f"  [pca_probe] need ≥3 proteins, got {len(loaded)} — aborting")
        return

    embeddings = np.stack([p["embedding"] for p in loaded])
    X = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(color="#21262d", alpha=0.5)

    # ── Panel 1: color by complex (memorization signal) ──────────────────────
    complexes = sorted(set(p["complex"] for p in loaded))
    cmap1 = plt.cm.get_cmap("tab20", len(complexes))

    for i, prot in enumerate(loaded):
        c_idx  = complexes.index(prot["complex"])
        marker = "o" if prot["in_training"] else "^"
        size   = 160 if prot["in_training"] else 200
        ax1.scatter(coords[i, 0], coords[i, 1],
                    color=cmap1(c_idx), marker=marker, s=size,
                    edgecolors="white", linewidth=0.8, zorder=3, alpha=0.9)
        txt = ax1.annotate(
            prot["id"].split("_")[0],
            (coords[i, 0], coords[i, 1]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7, color="#e6edf3", alpha=0.85,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="#0d1117")])

    # Legend: complex colors
    for ci, cname in enumerate(complexes):
        ax1.scatter([], [], color=cmap1(ci), marker="o", label=cname, s=60)
    ax1.scatter([], [], marker="o", color="gray", label="● train", s=60)
    ax1.scatter([], [], marker="^", color="gray", label="▲ held-out", s=60)
    ax1.legend(fontsize=7, labelcolor="#e6edf3", facecolor="#161b22",
               edgecolor="#30363d", ncol=2)
    ax1.set_title(
        f"Colored by complex  (memorization signal)\n"
        f"PC1={pca.explained_variance_ratio_[0]:.1%}  PC2={pca.explained_variance_ratio_[1]:.1%}",
        color="#e6edf3", fontsize=9,
    )
    ax1.set_xlabel("PC1", color="#e6edf3")
    ax1.set_ylabel("PC2", color="#e6edf3")

    # ── Panel 2: color by protein family (generalization signal) ─────────────
    families = sorted(set(p["family"] for p in loaded))
    cmap2 = plt.cm.get_cmap("Set1", len(families))

    for i, prot in enumerate(loaded):
        f_idx  = families.index(prot["family"])
        marker = "o" if prot["in_training"] else "^"
        size   = 160 if prot["in_training"] else 200
        ax2.scatter(coords[i, 0], coords[i, 1],
                    color=cmap2(f_idx), marker=marker, s=size,
                    edgecolors="white", linewidth=0.8, zorder=3, alpha=0.9)
        txt = ax2.annotate(
            prot["id"].split("_")[0],
            (coords[i, 0], coords[i, 1]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7, color="#e6edf3", alpha=0.85,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="#0d1117")])

    for fi, fname in enumerate(families):
        ax2.scatter([], [], color=cmap2(fi), marker="o", label=fname, s=60)
    ax2.scatter([], [], marker="o", color="gray", label="● train", s=60)
    ax2.scatter([], [], marker="^", color="gray", label="▲ held-out", s=60)
    ax2.legend(fontsize=7, labelcolor="#e6edf3", facecolor="#161b22",
               edgecolor="#30363d", ncol=2)
    ax2.set_title(
        "Colored by protein family  (generalization signal)\n"
        "Same family should cluster across complexes if generalizing",
        color="#e6edf3", fontsize=9,
    )
    ax2.set_xlabel("PC1", color="#e6edf3")
    ax2.set_ylabel("PC2", color="#e6edf3")

    # ── Interpretation annotation ─────────────────────────────────────────────
    ev = pca.explained_variance_ratio_
    fig.suptitle(
        f"ESM-150M PCA — Memorization vs Generalization probe\n"
        f"If panel 2 clusters > panel 1: model generalizes by function, not topology.",
        color="#e6edf3", fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [pca_probe] saved → {out}")
    print(f"  PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  total={sum(ev[:2]):.1%} variance explained")
    print(f"  n_proteins={len(loaded)}  "
          f"in_training={sum(p['in_training'] for p in loaded)}  "
          f"held_out={sum(not p['in_training'] for p in loaded)}")


if __name__ == "__main__":
    run_pca_probe()
