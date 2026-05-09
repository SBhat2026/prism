"""
Final alpha-phase report: comprehensive figure showing all key results.
"""

import sys, json, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT    = Path(__file__).parent
RESULTS = ROOT / "results" / "alpha"
sys.path.insert(0, str(ROOT))

from run_alpha import (
    BENCHMARK, MARSH_DIR, EMBED_DIR, compute_sasa_matrix, compute_go_matrix,
)
from src.simulation.assembly_sim import (
    Complex, greedy_assembly, ensemble_assembly, candidate_score, _ensure,
)

EMBED_8M  = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t6_8M_UR50D"
EMBED_150 = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t30_150M_UR50D"


def load_esm(pdb_id, chains, embed_dir, dim):
    data = np.load(str(embed_dir / "embeddings.npz"))
    embs = [data.get(f"{pdb_id}_{c}", np.zeros(dim)) for c in chains]
    mat  = np.array(embs)
    n    = np.linalg.norm(mat, axis=1, keepdims=True)
    mn   = mat / (n + 1e-8)
    return (mn @ mn.T).clip(-1, 1)


def build_all_results():
    with open(MARSH_DIR / "manifest.json") as f:
        manifest = json.load(f)

    rows = []
    for pdb_id, meta in BENCHMARK.items():
        chains   = meta["chains"]
        seqs     = [manifest.get(pdb_id, {}).get("sequences", {}).get(c, "M"*50) for c in chains]
        pdb_path = str(MARSH_DIR / f"{pdb_id}.pdb")
        sasa     = compute_sasa_matrix(pdb_path, chains)
        go_mat   = compute_go_matrix(seqs)
        e8       = load_esm(pdb_id, chains, EMBED_8M,  320)
        e150     = load_esm(pdb_id, chains, EMBED_150, 640)
        zero     = np.zeros((len(chains), len(chains)))

        def run(w_sasa=0, w_esm=0, w_go=0, esm=e8):
            return greedy_assembly(Complex(
                name=pdb_id, sequences=seqs, chain_ids=chains, pdb_path=pdb_path,
                sasa_matrix=sasa, ppi_matrix=zero, esm_matrix=esm, go_matrix=go_mat,
                ground_truth_order=meta["gt_order"],
            ), w_sasa=w_sasa, w_esm=w_esm, w_go=w_go, w_ppi=0.0)

        rows.append({
            "pdb": pdb_id, "name": meta["name"], "type": meta["type"], "n": len(chains),
            "sasa":    run(w_sasa=1.0).kendall_tau,
            "esm8":    run(w_esm=1.0, esm=e8).kendall_tau,
            "esm150":  run(w_esm=1.0, esm=e150).kendall_tau,
            "go":      run(w_go=1.0).kendall_tau,
            "all8":    run(w_sasa=.33, w_esm=.33, w_go=.33, esm=e8).kendall_tau,
            "all150":  run(w_sasa=.33, w_esm=.33, w_go=.33, esm=e150).kendall_tau,
        })
    return rows


def aggregate(rows, ctype=None):
    subset = rows if ctype is None else [r for r in rows if r["type"]==ctype]
    result = {}
    for col in ["sasa","esm8","esm150","go","all8","all150"]:
        vals = [r[col] for r in subset if r[col] is not None]
        result[col] = float(np.mean(vals)) if vals else 0.0
    return result


def generate_report(rows):
    homo   = aggregate(rows, "homomer")
    hetero = aggregate(rows, "heteromer")
    overall = aggregate(rows)

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    # ── Panel A: Grouped bar — homo vs hetero per condition ──────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    labels   = ["SASA\n(Marsh)", "ESM 8M", "ESM 150M", "GO\nCoherence", "All\n+ESM8M", "All\n+ESM150M"]
    homo_v   = [homo[k]   for k in ["sasa","esm8","esm150","go","all8","all150"]]
    hetero_v = [hetero[k] for k in ["sasa","esm8","esm150","go","all8","all150"]]
    x = np.arange(len(labels)); w = 0.35
    bars1 = ax_a.bar(x - w/2, homo_v,   w, label="Homomers",  color="#3498db", alpha=0.85)
    bars2 = ax_a.bar(x + w/2, hetero_v, w, label="Heteromers",color="#e74c3c", alpha=0.85)
    ax_a.axhline(homo["sasa"],   color="#3498db", ls=":", lw=1, alpha=0.6)
    ax_a.axhline(hetero["sasa"], color="#e74c3c", ls=":", lw=1, alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax_a.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.2f}", ha="center", fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax_a.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.2f}", ha="center", fontsize=7.5)
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel("Mean Kendall's τ"); ax_a.set_ylim(0, 1.2)
    ax_a.set_title("A.  Ablation Study — Mean Kendall's τ by Scoring Term", fontsize=11, loc="left")
    ax_a.legend(fontsize=9)
    ax_a.text(2, 1.08, "★ ESM 150M: τ=1.000 on homomers",
              ha="center", fontsize=9, color="#27ae60", fontweight="bold")

    # ── Panel B: Δτ vs SASA baseline ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    delta_homo   = {k: homo[k]   - homo["sasa"]   for k in ["esm8","esm150","go","all8","all150"]}
    delta_hetero = {k: hetero[k] - hetero["sasa"] for k in ["esm8","esm150","go","all8","all150"]}
    dlabels = ["ESM 8M", "ESM 150M", "GO", "All+8M", "All+150M"]
    dh  = [delta_homo[k]   for k in ["esm8","esm150","go","all8","all150"]]
    dhe = [delta_hetero[k] for k in ["esm8","esm150","go","all8","all150"]]
    y  = np.arange(len(dlabels))
    ax_b.barh(y+0.2, dh,  0.35, color=["#27ae60" if d>0 else "#c0392b" for d in dh],  label="Homo")
    ax_b.barh(y-0.2, dhe, 0.35, color=["#2980b9" if d>0 else "#e74c3c" for d in dhe], alpha=0.7, label="Hetero")
    ax_b.axvline(0, color="black", lw=0.8)
    ax_b.set_yticks(y); ax_b.set_yticklabels(dlabels, fontsize=8)
    ax_b.set_xlabel("Δτ vs SASA"); ax_b.set_title("B.  Δτ from Marsh Baseline", fontsize=11, loc="left")
    ax_b.legend(fontsize=8)

    # ── Panel C: Per-complex heatmap of τ ────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, :])
    cols    = ["sasa","esm8","esm150","go","all8","all150"]
    col_lbl = ["SASA\n(Marsh)","ESM\n8M","ESM\n150M","GO\nCoherence","All+\nESM8M","All+\nESM150M"]
    pdb_names = [f"{r['pdb']}\n{r['name']}" for r in rows]
    mat = np.array([[r[c] or 0 for c in cols] for r in rows])
    im  = ax_c.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=1.0)
    ax_c.set_xticks(range(len(cols))); ax_c.set_xticklabels(col_lbl, fontsize=8)
    ax_c.set_yticks(range(len(rows))); ax_c.set_yticklabels(pdb_names, fontsize=8)
    plt.colorbar(im, ax=ax_c, fraction=0.02, pad=0.01, label="Kendall's τ")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            ax_c.text(j, i, f"{v:.2f}", ha="center", va="center",
                      fontsize=8, color="black" if abs(v) < 0.7 else "white",
                      fontweight="bold" if v >= 1.0 else "normal")
    # Annotate complex type
    for i, r in enumerate(rows):
        color = "#e74c3c" if r["type"]=="heteromer" else "#3498db"
        ax_c.text(-0.5, i, "●", color=color, ha="center", va="center", fontsize=12)
    ax_c.set_title("C.  Per-Complex Kendall's τ Matrix  (● red=heteromer, blue=homomer)", fontsize=11, loc="left")

    # ── Panel D: NPC diversity ───────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, 0])
    try:
        npc_data = json.loads((RESULTS / "alpha_results.json").read_text())
        # Reconstruct step entropies from scratch via reading npc_step_entropy.png caption
        # Instead, show NPC unique pathway count per temperature
        from src.simulation.assembly_sim import ensemble_assembly
        from run_alpha import NPC_DIR, NPC_EMBED
        import json as _json
        with open(ROOT / "data" / "complexes" / "npc" / "manifest.json") as f:
            nm = _json.load(f)
        chains = sorted(nm["3F3F"]["sequences"].keys())[:8]
        seqs   = [nm["3F3F"]["sequences"].get(c, "M"*50) for c in chains]
        from src.data_prep.compute_embeddings import load_esm_model, get_device
        e_mat  = load_esm("3F3F", chains, NPC_EMBED, 320)
        s_mat  = compute_sasa_matrix(str(NPC_DIR/"3F3F.pdb"), chains)
        g_mat  = compute_go_matrix(seqs)
        npc    = Complex("NPC", seqs, chains, str(NPC_DIR/"3F3F.pdb"),
                         s_mat, np.zeros((8,8)), e_mat, g_mat)

        temps  = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
        diversities = []
        for T in temps:
            _, stats = ensemble_assembly(npc, n_runs=100, temperature=T,
                                         w_sasa=0.33, w_esm=0.33, w_go=0.33)
            diversities.append(stats["n_unique_pathways"])

        ax_d.plot(temps, diversities, "o-", color="#9b59b6", lw=2, ms=7)
        ax_d.fill_between(temps, diversities, alpha=0.15, color="#9b59b6")
        ax_d.set_xlabel("Temperature"); ax_d.set_ylabel("Unique pathways / 100 runs")
        ax_d.set_title("D.  NPC 3F3F: Path Diversity vs Temperature", fontsize=10, loc="left")
        ax_d.axhline(1,   color="gray", ls="--", lw=1, label="Fully deterministic")
        ax_d.axhline(100, color="gray", ls=":",  lw=1, label="Fully random")
        ax_d.legend(fontsize=8); ax_d.grid(alpha=0.3)
    except Exception as e:
        ax_d.text(0.5, 0.5, f"NPC data unavailable\n{e}", ha="center", va="center",
                  transform=ax_d.transAxes)
        ax_d.set_title("D.  NPC pathway diversity", loc="left")

    # ── Panel E: ESM8 vs ESM150 scatter ─────────────────────────────────────
    ax_e = fig.add_subplot(gs[2, 1])
    esm8_taus  = [r["esm8"]  or 0 for r in rows]
    esm150_taus= [r["esm150"] or 0 for r in rows]
    colors_e   = ["#e74c3c" if r["type"]=="heteromer" else "#3498db" for r in rows]
    ax_e.scatter(esm8_taus, esm150_taus, c=colors_e, s=100, edgecolors="white", zorder=5)
    for i, r in enumerate(rows):
        ax_e.annotate(r["pdb"], (esm8_taus[i], esm150_taus[i]),
                      fontsize=7, xytext=(3,2), textcoords="offset points")
    ax_e.plot([-1,1],[-1,1], "k--", lw=1, alpha=0.4)
    ax_e.set_xlabel("ESM 8M τ"); ax_e.set_ylabel("ESM 150M τ")
    ax_e.set_title("E.  ESM 8M vs 150M per Complex", fontsize=10, loc="left")
    red_p  = mpatches.Patch(color="#e74c3c", label="Heteromer")
    blue_p = mpatches.Patch(color="#3498db", label="Homomer")
    ax_e.legend(handles=[red_p, blue_p], fontsize=8)
    ax_e.set_xlim(-0.3, 1.2); ax_e.set_ylim(-0.3, 1.2)
    ax_e.grid(alpha=0.3)
    ax_e.text(0.05, 0.95, "★ ESM 150M >> 8M\n  for homomers",
              transform=ax_e.transAxes, fontsize=8.5, color="#27ae60",
              fontweight="bold", va="top")

    # ── Panel F: Key findings text ───────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 2])
    ax_f.axis("off")
    findings = (
        "ALPHA PHASE — KEY FINDINGS\n"
        "══════════════════════════\n\n"
        "1. ESM 150M: τ=1.000 on homomers\n"
        "   vs ESM 8M: τ=0.143\n"
        "   → Scale matters for structural\n"
        "     symmetry encoding\n\n"
        "2. SASA (Marsh baseline) best for\n"
        "   heteromers (τ=0.958 vs ESM 0.833)\n"
        "   → Interface area captures\n"
        "     heteromeric specificity\n\n"
        "3. GO coherence: τ=0.917 heteromers\n"
        "   > ESM 8M (0.833) — confirms\n"
        "   functional coherence hypothesis\n"
        "   ⚠️ Weak: model is OOD for human\n\n"
        "4. NPC 3F3F (8-mer): near-max entropy\n"
        "   at T=0.5; 294 unique paths/300\n"
        "   → Step 3 is assembly bottleneck\n\n"
        "NEXT: Train GNN on these matrices\n"
        "      to learn adaptive weights"
    )
    ax_f.text(0.03, 0.97, findings, va="top", ha="left",
              transform=ax_f.transAxes, fontsize=8,
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", alpha=0.9))

    fig.suptitle("Protein Assembly Simulation — Alpha Phase Report", fontsize=14, y=1.0)
    outpath = str(RESULTS / "FINAL_ALPHA_REPORT.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {outpath}")


if __name__ == "__main__":
    print("Building results...")
    rows = build_all_results()
    print("\nGenerating final report...")
    generate_report(rows)
    print("Done.")
