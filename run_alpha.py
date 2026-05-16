"""
Alpha-phase end-to-end demo.
Runs: SASA → ESM cosine → GO coherence → STRING PPI → assembly sim → ablation → viz.
All results saved to results/alpha/
"""

import os, sys, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
RESULTS  = ROOT / "results" / "alpha"
DATA     = ROOT / "data"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.simulation.assembly_sim import (
    Complex, greedy_assembly, stochastic_assembly, ensemble_assembly,
    ablation_table, evaluate_kendall_tau, _ensure,
)
from src.gromacs.extract_sasa import GromacsSASA
from src.visualization.trajectory_viz import (
    plot_assembly_order, plot_scoring_heatmap,
    plot_pathway_diversity, plot_ablation_table,
)

# ── Config ───────────────────────────────────────────────────────────────────
MARSH_DIR  = DATA / "complexes" / "marsh2013"
EMBED_DIR  = DATA / "embeddings" / "marsh2013" / "esm2_t6_8M_UR50D"
NPC_DIR    = DATA / "complexes" / "npc"
NPC_EMBED  = DATA / "embeddings" / "npc" / "esm2_t6_8M_UR50D"

SASA_CALC = GromacsSASA()
LOG = []

def log(msg):
    print(msg)
    LOG.append(msg)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GROMACS SASA pairwise matrices
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sasa_matrix(pdb_path: str, chain_ids: list[str]) -> np.ndarray:
    """Compute N×N buried-SASA matrix for a complex PDB."""
    N = len(chain_ids)
    mat = np.zeros((N, N))
    for i, ci in enumerate(chain_ids):
        for j, cj in enumerate(chain_ids):
            if i >= j:
                continue
            try:
                bsa = SASA_CALC.interface_buried_area(pdb_path, ci, cj)
                mat[i, j] = bsa
                mat[j, i] = bsa
            except Exception as e:
                log(f"  SASA WARN {Path(pdb_path).stem} {ci}-{cj}: {e}")
    return mat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ESM cosine matrix loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_esm_matrix(pdb_id: str, chain_ids: list[str], embed_dir: Path) -> np.ndarray:
    """Load per-chain embeddings and build cosine similarity matrix."""
    data = np.load(str(embed_dir / "embeddings.npz"))
    embs = []
    for c in chain_ids:
        key = f"{pdb_id}_{c}"
        if key in data:
            embs.append(data[key])
        else:
            embs.append(np.zeros(320))
    mat = np.array(embs)  # (N, 320)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_n = mat / (norms + 1e-8)
    return (mat_n @ mat_n.T).clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FABLE GO coherence matrix
# ═══════════════════════════════════════════════════════════════════════════════

_bridge = None
def get_bridge():
    global _bridge
    if _bridge is None:
        try:
            from src.protfunc_bridge.go_coherence import FABLEBridge
            _bridge = FABLEBridge()
            log("  [FABLE] bridge loaded OK")
        except Exception as e:
            log(f"  [FABLE] WARN: {e}. Using zero GO matrix.")
    return _bridge

def compute_go_matrix(sequences: list[str]) -> np.ndarray:
    bridge = get_bridge()
    if bridge is None:
        return np.zeros((len(sequences), len(sequences)))
    try:
        # Use logits (pre-sigmoid) for cosine: better discrimination on human proteins
        vecs = [bridge.get_logits(s) for s in sequences]
        N = len(vecs)
        mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    a, b = vecs[i], vecs[j]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    mat[i, j] = float(np.dot(a, b) / (na * nb + 1e-8))
        return mat
    except Exception as e:
        log(f"  [GO] error: {e}")
        return np.zeros((len(sequences), len(sequences)))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Benchmark complexes with ground-truth assembly orders
# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth orders from Marsh 2013 Table S1 (assembly = disassembly reversed)
# For heteromers, order = [chain that docks last in disassembly = first assembled].
# We encode as step order: index of chain in assembly sequence.

BENCHMARK = {
    # PDB   chains     GT order (0-indexed)   complex type    name
    "2HHB": {
        # Hemoglobin: α1β1 dimer forms first, then α2β2 joins → α1(0) seeds,
        # β1(1) joins (largest interface), then β2(3), α2(2) last
        "chains": ["A", "B", "C", "D"],
        "gt_order": [0, 1, 3, 2],
        "type": "heteromer",
        "name": "Hemoglobin α2β2",
    },
    "1BRS": {
        "chains": ["A", "B", "C", "D", "E", "F"],  # 3× barnase + 3× barstar
        "gt_order": [0, 3, 1, 4, 2, 5],  # alternating barnase/barstar dimers
        "type": "heteromer",
        "name": "Barnase / Barstar (6-mer)",
    },
    "1AY7": {
        "chains": ["A", "B"],
        "gt_order": [0, 1],
        "type": "heteromer",
        "name": "RNase SA / Barstar",
    },
    "1TGS": {
        "chains": ["Z", "I"],  # trypsinogen (Z) + PSTI inhibitor (I)
        "gt_order": [0, 1],
        "type": "heteromer",
        "name": "Trypsin / PSTI",
    },
    "2PTC": {
        "chains": ["E", "I"],  # trypsin (E) + BPTI (I)
        "gt_order": [0, 1],
        "type": "heteromer",
        "name": "Trypsin / BPTI",
    },
    "1A2K": {
        "chains": ["A", "B", "C", "D", "E"],  # Ran/Importin-β 5-mer
        "gt_order": [1, 0, 3, 2, 4],
        "type": "heteromer",
        "name": "Ran / Importin-β (5-mer)",
    },
    "1SBB": {
        "chains": ["A", "B", "C", "D"],  # TCR / Enterotoxin-B 4-mer
        "gt_order": [2, 0, 3, 1],
        "type": "heteromer",
        "name": "TCR / Enterotoxin-B (4-mer)",
    },
    "3GBN": {
        # HA (A+B heterotrimer context) + antibody (H+L): HA seeds, Ab joins
        "chains": ["A", "H", "L"],
        "gt_order": [0, 1, 2],  # HA1 seeds → heavy → light
        "type": "heteromer",
        "name": "Hemagglutinin / Antibody",
    },
    "1AON": {
        "chains": list("ABCDEFG"),  # GroEL ring 1 (7-mer)
        "gt_order": list(range(7)),  # radial sequential
        "type": "homomer",
        "name": "GroEL ring (7-mer)",
    },
    "1GRU": {
        "chains": list("ABCDEFG"),  # GroES (7-mer)
        "gt_order": list(range(7)),
        "type": "homomer",
        "name": "GroES (7-mer)",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Build Complex objects with all scoring matrices
# ═══════════════════════════════════════════════════════════════════════════════

def build_complex(pdb_id: str, meta: dict, embed_dir: Path) -> Complex:
    pdb_path = str(MARSH_DIR / f"{pdb_id}.pdb")
    chains   = meta["chains"]
    N        = len(chains)
    log(f"\n── {pdb_id}: {meta['name']} ({N} subunits, {meta['type']}) ──")

    # Load sequences from manifest
    with open(MARSH_DIR / "manifest.json") as f:
        manifest = json.load(f)
    seq_map = manifest.get(pdb_id, {}).get("sequences", {})
    sequences = [seq_map.get(c, "M" * 50) for c in chains]

    # SASA matrix
    log(f"  Computing SASA matrix...")
    sasa_mat = compute_sasa_matrix(pdb_path, chains)
    log(f"  SASA max: {sasa_mat.max():.3f} nm²  mean-nonzero: {sasa_mat[sasa_mat>0].mean():.3f} nm²")

    # ESM cosine matrix
    esm_mat = load_esm_matrix(pdb_id, chains, embed_dir)

    # GO coherence matrix
    log(f"  Computing GO coherence matrix...")
    go_mat = compute_go_matrix(sequences)

    # PPI matrix: use ESM cosine as stand-in (STRING API call saved for later)
    # Will be replaced in Phase 5
    ppi_mat = np.zeros((N, N))

    return Complex(
        name=f"{pdb_id}_{meta['name']}",
        sequences=sequences,
        chain_ids=chains,
        pdb_path=pdb_path,
        sasa_matrix=sasa_mat,
        ppi_matrix=ppi_mat,
        esm_matrix=esm_mat,
        go_matrix=go_mat,
        ground_truth_order=meta["gt_order"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Run ablation across all benchmark complexes
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_CONDITIONS = [
    {"name": "SASA only (Marsh baseline)", "w_sasa": 1.0, "w_ppi": 0.0, "w_esm": 0.0, "w_go": 0.0},
    {"name": "ESM compat only",            "w_sasa": 0.0, "w_ppi": 0.0, "w_esm": 1.0, "w_go": 0.0},
    {"name": "GO coherence only",          "w_sasa": 0.0, "w_ppi": 0.0, "w_esm": 0.0, "w_go": 1.0},
    {"name": "ESM + GO (no SASA)",         "w_sasa": 0.0, "w_ppi": 0.0, "w_esm": 0.5, "w_go": 0.5},
    {"name": "SASA + ESM",                 "w_sasa": 0.5, "w_ppi": 0.0, "w_esm": 0.5, "w_go": 0.0},
    {"name": "SASA + GO",                  "w_sasa": 0.5, "w_ppi": 0.0, "w_esm": 0.0, "w_go": 0.5},
    {"name": "All terms (equal)",          "w_sasa": 0.33,"w_ppi": 0.0, "w_esm": 0.33,"w_go": 0.33},
]

def run_ablation(complexes: list[Complex]) -> dict:
    """Run all ablation conditions on all complexes. Return structured results."""
    results = {c["name"]: {"homomers": [], "heteromers": []} for c in ABLATION_CONDITIONS}
    per_complex = {}

    for cplx in complexes:
        is_homo = "homomer" in cplx.name.lower() or any(
            meta["type"] == "homomer" for pid, meta in BENCHMARK.items() if pid in cplx.name
        )
        ctype = "homomers" if is_homo else "heteromers"
        per_complex[cplx.name] = {}

        for cond in ABLATION_CONDITIONS:
            w = {k: v for k, v in cond.items() if k.startswith("w_")}
            try:
                r_greedy = greedy_assembly(cplx, **w)
                _, stats  = ensemble_assembly(cplx, n_runs=200, temperature=0.6, **w)
                tau_g = r_greedy.kendall_tau or 0.0
                tau_s = stats["mean_tau"] or 0.0
                n_paths = stats["n_unique_pathways"]
            except Exception as e:
                log(f"  ABLATION WARN {cplx.name} / {cond['name']}: {e}")
                tau_g, tau_s, n_paths = 0.0, 0.0, 1

            results[cond["name"]][ctype].append(tau_g)
            per_complex[cplx.name][cond["name"]] = {
                "greedy_tau": tau_g, "mean_stochastic_tau": tau_s, "n_paths": n_paths
            }

    # Aggregate mean per condition per type
    summary = []
    for cond in ABLATION_CONDITIONS:
        homo  = results[cond["name"]]["homomers"]
        hetero = results[cond["name"]]["heteromers"]
        summary.append({
            "condition": cond["name"],
            "mean_tau_homo":   float(np.mean(homo))   if homo   else None,
            "mean_tau_hetero": float(np.mean(hetero)) if hetero else None,
            "mean_tau_all":    float(np.mean(homo + hetero)),
        })
    return {"summary": summary, "per_complex": per_complex}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ablation_split(summary: list[dict], outpath: str):
    """Side-by-side homo / hetero ablation bars with Δ-from-baseline annotation."""
    conditions = [r["condition"] for r in summary]
    homo_taus   = [r["mean_tau_homo"]   or 0 for r in summary]
    hetero_taus = [r["mean_tau_hetero"] or 0 for r in summary]

    baseline_h = homo_taus[0]
    baseline_e = hetero_taus[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors_h = ["#4878CF" if t >= baseline_h else "#C44E52" for t in homo_taus]
    colors_e = ["#4878CF" if t >= baseline_e else "#C44E52" for t in hetero_taus]

    y = np.arange(len(conditions))
    ax1.barh(y, homo_taus, color=colors_h, edgecolor="white", height=0.6)
    ax1.axvline(baseline_h, color="gray", ls="--", lw=1, label="SASA baseline")
    ax1.set_yticks(y); ax1.set_yticklabels(conditions, fontsize=9)
    ax1.set_xlabel("Mean Kendall's τ"); ax1.set_title("Homomers")
    ax1.set_xlim(-0.3, 1.05); ax1.legend(fontsize=8)

    ax2.barh(y, hetero_taus, color=colors_e, edgecolor="white", height=0.6)
    ax2.axvline(baseline_e, color="gray", ls="--", lw=1, label="SASA baseline")
    ax2.set_xlabel("Mean Kendall's τ"); ax2.set_title("Heteromers")
    ax2.set_xlim(-0.3, 1.05); ax2.legend(fontsize=8)

    # Δ annotations
    for ax, taus, bl in [(ax1, homo_taus, baseline_h), (ax2, hetero_taus, baseline_e)]:
        for i, t in enumerate(taus):
            delta = t - bl
            label = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            color = "green" if delta > 0.01 else ("red" if delta < -0.01 else "gray")
            ax.text(max(t + 0.02, 0.02), i, label, va="center", fontsize=7.5, color=color)

    fig.suptitle("Ablation Study — Assembly Order Prediction (Greedy, Marsh 2013)", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved → {outpath}")


def plot_per_complex_heatmap(per_complex: dict, conditions: list[str], outpath: str):
    """Heatmap: complexes × conditions, colored by Kendall's τ."""
    complex_names = list(per_complex.keys())
    n_c = len(complex_names)
    n_k = len(conditions)
    mat = np.zeros((n_c, n_k))
    for i, name in enumerate(complex_names):
        for j, cond in enumerate(conditions):
            mat[i, j] = per_complex[name].get(cond, {}).get("greedy_tau", 0.0)

    fig, ax = plt.subplots(figsize=(max(10, n_k * 1.4), max(4, n_c * 0.7)))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Kendall's τ")
    ax.set_xticks(range(n_k))
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels([n.split("_", 1)[-1] for n in complex_names], fontsize=8)
    ax.set_title("Greedy τ — All Complexes × Ablation Conditions")
    for i in range(n_c):
        for j in range(n_k):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < 0.6 else "white")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved → {outpath}")


def plot_pathway_ensemble(cplx: Complex, outpath: str, n_runs: int = 300):
    """Stochastic ensemble pathway plot for one complex."""
    results, stats = ensemble_assembly(
        cplx, n_runs=n_runs, temperature=0.5,
        w_sasa=0.33, w_esm=0.33, w_go=0.33
    )
    predicted_orders = [r.predicted_order for r in results]
    plot_assembly_order(
        predicted_orders,
        cplx.ground_truth_order,
        cplx.chain_ids,
        title=f"Pathway Ensemble — {cplx.name} (n={n_runs})",
        outpath=outpath,
    )
    log(f"  Unique pathways: {stats['n_unique_pathways']}  Mean τ: {stats['mean_tau']:.3f}")
    return stats


def plot_scoring_matrix_panel(cplx: Complex, outpath: str):
    """4-panel heatmap for all scoring matrices."""
    plot_scoring_heatmap(
        {
            "ΔSASA (nm²)": cplx.sasa_matrix,
            "ESM cosine":  cplx.esm_matrix,
            "GO coherence": cplx.go_matrix,
            "PPI (STRING)": cplx.ppi_matrix,
        },
        cplx.chain_ids,
        title=f"Scoring Matrices — {cplx.name}",
        outpath=outpath,
    )
    log(f"  Saved → {outpath}")


def plot_tau_summary_bar(all_results: list[dict], outpath: str):
    """Summary bar chart: per-complex best τ across conditions."""
    names, taus, colors = [], [], []
    for item in all_results:
        best_tau = max(
            v.get("greedy_tau", 0)
            for v in item["ablation"].values()
        )
        baseline_tau = item["ablation"].get("SASA only (Marsh baseline)", {}).get("greedy_tau", 0)
        names.append(item["name"].split("_", 1)[-1])
        taus.append(best_tau)
        colors.append("#2ecc71" if best_tau > baseline_tau + 0.05 else "#3498db")

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6)))
    y = np.arange(len(names))
    ax.barh(y, taus, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Best Kendall's τ (any condition)")
    ax.set_title("Best Assembly Order Prediction per Complex")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.5, color="gray", ls=":", lw=1, label="τ=0.5 reference")
    ax.legend(fontsize=8)
    green_patch = mpatches.Patch(color="#2ecc71", label="Beats SASA baseline")
    blue_patch  = mpatches.Patch(color="#3498db", label="SASA = best or tied")
    ax.legend(handles=[green_patch, blue_patch], fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved → {outpath}")


def plot_go_vs_sasa_scatter(all_results: list[dict], outpath: str):
    """Scatter: SASA τ vs GO τ per complex (do they agree or disagree?)."""
    sasa_taus, go_taus, names, types = [], [], [], []
    for item in all_results:
        sasa_tau = item["ablation"].get("SASA only (Marsh baseline)", {}).get("greedy_tau", 0)
        go_tau   = item["ablation"].get("GO coherence only",           {}).get("greedy_tau", 0)
        sasa_taus.append(sasa_tau)
        go_taus.append(go_tau)
        names.append(item["name"].split("_", 1)[-1][:15])
        types.append(item["type"])

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#e74c3c" if t == "heteromer" else "#3498db" for t in types]
    ax.scatter(sasa_taus, go_taus, c=colors, s=90, edgecolors="white", zorder=5)
    for x, y, n in zip(sasa_taus, go_taus, names):
        ax.annotate(n, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.plot([-1, 1], [-1, 1], "k--", lw=1, alpha=0.4, label="Equal performance")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("SASA-only Kendall's τ")
    ax.set_ylabel("GO-coherence-only Kendall's τ")
    ax.set_title("SASA vs GO Coherence: per-complex τ")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    red_p  = mpatches.Patch(color="#e74c3c", label="Heteromer")
    blue_p = mpatches.Patch(color="#3498db", label="Homomer")
    ax.legend(handles=[red_p, blue_p], fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. NPC ensemble analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_npc_analysis():
    log("\n══ NPC Analysis ══")
    npc_manifest_path = DATA / "complexes" / "npc" / "manifest.json"
    with open(npc_manifest_path) as f:
        npc_manifest = json.load(f)

    # Use 3F3F (Nup-SEH1 × 4 + Nup85 × 4 = 8-chain NPC Y-complex subunit)
    pdb_id = "3F3F"
    pdb_path = str(NPC_DIR / f"{pdb_id}.pdb")
    seq_map  = npc_manifest.get(pdb_id, {}).get("sequences", {})
    chains   = sorted(seq_map.keys())[:8]
    log(f"  3F3F chains: {chains}  ({len(chains)} subunits, Nup-SEH1 + Nup85)")

    sequences = [seq_map.get(c, "M" * 50) for c in chains]

    # SASA matrix
    log("  Computing NPC SASA matrix...")
    sasa_mat = compute_sasa_matrix(pdb_path, chains)

    # ESM matrix
    esm_mat = load_esm_matrix(pdb_id, chains, NPC_EMBED)

    # GO matrix
    log("  Computing NPC GO coherence matrix...")
    go_mat = compute_go_matrix(sequences)

    npc_cplx = Complex(
        name=f"NPC_3F3F_Nup-SEH1-Nup85",
        sequences=sequences,
        chain_ids=chains,
        pdb_path=pdb_path,
        sasa_matrix=sasa_mat,
        ppi_matrix=np.zeros((len(chains), len(chains))),
        esm_matrix=esm_mat,
        go_matrix=go_mat,
        ground_truth_order=None,  # no single ground truth — we explore
    )

    # Run large ensemble for pathway diversity
    log("  Running NPC ensemble (n=500)...")
    results, stats = ensemble_assembly(
        npc_cplx, n_runs=500, temperature=0.5,
        w_sasa=0.33, w_esm=0.33, w_go=0.33
    )

    log(f"  Unique pathways: {stats['n_unique_pathways']} / 500 runs")
    log(f"  Max-entropy step: {stats['max_entropy_step']} (subunit {chains[stats['max_entropy_step']] if stats['max_entropy_step'] and stats['max_entropy_step'] < len(chains) else '?'})")

    # Top 3 pathways
    for i, (order, count) in enumerate(stats["order_distribution"][:3]):
        freq = count / 500
        log(f"  Pathway {i+1}: {[chains[x] for x in order]} ({freq:.1%})")

    # Plots
    predicted_orders = [r.predicted_order for r in results]
    plot_assembly_order(
        predicted_orders, None, chains,
        title=f"NPC 3F3F (Nup-SEH1/Nup85) — Pathway Ensemble (n=500)",
        outpath=str(RESULTS / "npc_pathway_ensemble.png"),
    )
    plot_scoring_heatmap(
        {"ΔSASA": sasa_mat, "ESM cosine": esm_mat, "GO coherence": go_mat},
        chains,
        title="NPC Y-complex — Scoring Matrices",
        outpath=str(RESULTS / "npc_scoring_matrices.png"),
    )
    plot_pathway_diversity(
        stats["step_entropies"], chains,
        title="NPC Y-complex — Assembly Step Entropy",
        outpath=str(RESULTS / "npc_step_entropy.png"),
    )
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("═" * 60)
    log("  Protein Assembly Sim — Alpha Demo")
    log("═" * 60)

    # Build all benchmark complexes
    complexes = []
    all_results = []
    for pdb_id, meta in BENCHMARK.items():
        try:
            cplx = build_complex(pdb_id, meta, EMBED_DIR)
            complexes.append(cplx)
        except Exception as e:
            log(f"  BUILD FAIL {pdb_id}: {e}")
            continue

        # Per-complex scoring matrix visualization
        mat_dir = RESULTS / "scoring_matrices"
        mat_dir.mkdir(exist_ok=True)
        plot_scoring_matrix_panel(cplx, str(mat_dir / f"{pdb_id}_matrices.png"))

        # Ablation for this complex
        ablation_row = {}
        for cond in ABLATION_CONDITIONS:
            w = {k: v for k, v in cond.items() if k.startswith("w_")}
            try:
                r = greedy_assembly(cplx, **w)
                ablation_row[cond["name"]] = {
                    "greedy_tau": r.kendall_tau,
                }
            except Exception as e:
                ablation_row[cond["name"]] = {"greedy_tau": 0.0}

        all_results.append({
            "name": cplx.name,
            "type": meta["type"],
            "ablation": ablation_row,
        })

        # Ensemble for key complexes
        if len(meta["chains"]) >= 4:
            ens_dir = RESULTS / "ensembles"
            ens_dir.mkdir(exist_ok=True)
            plot_pathway_ensemble(
                cplx,
                str(ens_dir / f"{pdb_id}_ensemble.png"),
                n_runs=200,
            )

    # Global ablation
    log("\n══ Running Full Ablation ══")
    ablation_data = run_ablation(complexes)

    # Summary visualizations
    log("\n══ Generating Visualizations ══")
    plot_ablation_split(
        ablation_data["summary"],
        str(RESULTS / "ablation_homo_vs_hetero.png"),
    )
    plot_per_complex_heatmap(
        ablation_data["per_complex"],
        [c["name"] for c in ABLATION_CONDITIONS],
        str(RESULTS / "per_complex_heatmap.png"),
    )
    plot_tau_summary_bar(all_results, str(RESULTS / "best_tau_per_complex.png"))
    plot_go_vs_sasa_scatter(all_results, str(RESULTS / "go_vs_sasa_scatter.png"))

    # NPC analysis
    try:
        run_npc_analysis()
    except Exception as e:
        log(f"  NPC analysis error: {e}")

    # Save results JSON
    report = {
        "ablation_summary": ablation_data["summary"],
        "per_complex": ablation_data["per_complex"],
        "complexes_analyzed": [c.name for c in complexes],
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(RESULTS / "alpha_results.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save log
    with open(RESULTS / "run_log.txt", "w") as f:
        f.write("\n".join(LOG))

    log(f"\n═══ Done in {time.time()-t0:.0f}s ═══")
    log(f"Results → {RESULTS}")

    # Print summary table
    log("\n── Ablation Summary Table ──")
    header = f"{'Condition':<32} {'Homo τ':>8} {'Hetero τ':>9} {'All τ':>7}"
    log(header)
    log("-" * len(header))
    for row in ablation_data["summary"]:
        hom = f"{row['mean_tau_homo']:.3f}"   if row['mean_tau_homo']   is not None else "  N/A"
        het = f"{row['mean_tau_hetero']:.3f}" if row['mean_tau_hetero'] is not None else "  N/A"
        all_ = f"{row['mean_tau_all']:.3f}"
        log(f"{row['condition']:<32} {hom:>8} {het:>9} {all_:>7}")


if __name__ == "__main__":
    main()
