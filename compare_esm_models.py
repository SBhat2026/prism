"""
Compare 8M vs 150M ESM embeddings on assembly order prediction.
Quick ablation: does 150M ESM cosine give better Kendall's τ?
"""

import sys, json, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).parent
RESULTS = ROOT / "results" / "alpha"
sys.path.insert(0, str(ROOT))

from run_alpha import BENCHMARK, MARSH_DIR, compute_sasa_matrix, compute_go_matrix, get_bridge
from src.simulation.assembly_sim import Complex, greedy_assembly
from src.data_prep.fetch_complexes import _extract_sequences_from_pdb

EMBED_8M  = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t6_8M_UR50D"
EMBED_150 = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t30_150M_UR50D"

def load_esm(pdb_id, chains, embed_dir, expected_dim):
    data = np.load(str(embed_dir / "embeddings.npz"))
    embs = []
    for c in chains:
        key = f"{pdb_id}_{c}"
        embs.append(data[key] if key in data else np.zeros(expected_dim))
    mat = np.array(embs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / (norms + 1e-8) @ (mat / (norms + 1e-8)).T).clip(-1, 1)


def compare():
    with open(MARSH_DIR / "manifest.json") as f:
        manifest = json.load(f)

    rows = []
    for pdb_id, meta in BENCHMARK.items():
        chains = meta["chains"]
        seqs   = [manifest.get(pdb_id, {}).get("sequences", {}).get(c, "M"*50) for c in chains]
        pdb_path = str(MARSH_DIR / f"{pdb_id}.pdb")

        sasa_mat = compute_sasa_matrix(pdb_path, chains)
        go_mat   = compute_go_matrix(seqs)
        esm8_mat = load_esm(pdb_id, chains, EMBED_8M,  320)
        esm150_mat = load_esm(pdb_id, chains, EMBED_150, 640)

        base_cplx = Complex(
            name=pdb_id, sequences=seqs, chain_ids=chains,
            pdb_path=pdb_path,
            sasa_matrix=sasa_mat, ppi_matrix=np.zeros((len(chains),len(chains))),
            esm_matrix=esm8_mat,   go_matrix=go_mat,
            ground_truth_order=meta["gt_order"],
        )
        cplx_150 = Complex(
            name=pdb_id, sequences=seqs, chain_ids=chains,
            pdb_path=pdb_path,
            sasa_matrix=sasa_mat, ppi_matrix=np.zeros((len(chains),len(chains))),
            esm_matrix=esm150_mat, go_matrix=go_mat,
            ground_truth_order=meta["gt_order"],
        )

        r_sasa    = greedy_assembly(base_cplx, w_sasa=1.0, w_esm=0.0, w_go=0.0, w_ppi=0.0)
        r_esm8    = greedy_assembly(base_cplx, w_sasa=0.0, w_esm=1.0, w_go=0.0, w_ppi=0.0)
        r_esm150  = greedy_assembly(cplx_150,  w_sasa=0.0, w_esm=1.0, w_go=0.0, w_ppi=0.0)
        r_all8    = greedy_assembly(base_cplx, w_sasa=0.33,w_esm=0.33,w_go=0.33,w_ppi=0.0)
        r_all150  = greedy_assembly(cplx_150,  w_sasa=0.33,w_esm=0.33,w_go=0.33,w_ppi=0.0)

        rows.append({
            "pdb": pdb_id,
            "name": meta["name"],
            "type": meta["type"],
            "n": len(chains),
            "sasa_tau":    r_sasa.kendall_tau,
            "esm8_tau":    r_esm8.kendall_tau,
            "esm150_tau":  r_esm150.kendall_tau,
            "all8_tau":    r_all8.kendall_tau,
            "all150_tau":  r_all150.kendall_tau,
        })
        print(f"  {pdb_id:6s} ({meta['type']:10s}) SASA:{r_sasa.kendall_tau:.2f} "
              f"ESM8:{r_esm8.kendall_tau:.2f} ESM150:{r_esm150.kendall_tau:.2f} "
              f"ALL8:{r_all8.kendall_tau:.2f} ALL150:{r_all150.kendall_tau:.2f}")

    # Summary
    homomers   = [r for r in rows if r["type"] == "homomer"]
    heteromers = [r for r in rows if r["type"] == "heteromer"]

    print("\n── Summary ──")
    for label, group in [("Homomers", homomers), ("Heteromers", heteromers), ("All", rows)]:
        for col in ["sasa_tau", "esm8_tau", "esm150_tau", "all8_tau", "all150_tau"]:
            vals = [r[col] for r in group if r[col] is not None]
            print(f"  {label:12s} {col:15s}: {np.mean(vals):.3f}")
        print()

    # Plot comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    names  = [f"{r['pdb']} ({r['type'][:4]})" for r in rows]
    x      = np.arange(len(names))
    w      = 0.15

    ax.bar(x - 2*w, [r["sasa_tau"]   for r in rows], w, label="SASA",         color="#3498db")
    ax.bar(x - w,   [r["esm8_tau"]   for r in rows], w, label="ESM 8M",       color="#2ecc71")
    ax.bar(x,       [r["esm150_tau"] for r in rows], w, label="ESM 150M",     color="#27ae60")
    ax.bar(x + w,   [r["all8_tau"]   for r in rows], w, label="All + ESM 8M", color="#e67e22")
    ax.bar(x + 2*w, [r["all150_tau"] for r in rows], w, label="All + ESM 150M",color="#d35400")

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Kendall's τ"); ax.set_ylim(-0.2, 1.2)
    ax.set_title("ESM 8M vs 150M — Assembly Order Kendall's τ")
    ax.legend(fontsize=8, ncol=5)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(RESULTS / "esm_8m_vs_150m.png"), dpi=150)
    plt.close()
    print(f"  Saved → results/alpha/esm_8m_vs_150m.png")


if __name__ == "__main__":
    compare()
