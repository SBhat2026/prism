"""
GNN graph visualizations for the assembly simulation.
Generates:
1. Assembly graph snapshots (partial complex at each step)
2. Edge weight evolution during assembly
3. Score contribution breakdown per step
4. Pathway comparison: SASA vs ESM vs GO vs All
5. 150M ESM embedding PCA/UMAP for subunit similarity space
6. NPC pathway tree visualization
"""

import os, sys, json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
from pathlib import Path
from itertools import combinations

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results" / "alpha"
GNN_DIR = RESULTS / "gnn_viz"
GNN_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from run_alpha import (
    build_complex, BENCHMARK, EMBED_DIR, MARSH_DIR, load_esm_matrix,
    compute_sasa_matrix, compute_go_matrix, get_bridge, log,
)
from src.simulation.assembly_sim import (
    greedy_assembly, ensemble_assembly, candidate_score, _ensure,
)

CMAP_EDGE  = plt.cm.YlOrRd
CMAP_NODE  = plt.cm.Blues
NODE_SCORE_CMAP = plt.cm.RdYlGn


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Assembly graph snapshots
# ═══════════════════════════════════════════════════════════════════════════════

def draw_assembly_step(
    G: nx.Graph,
    assembled: list[int],
    candidate: int,
    chain_ids: list[str],
    pos: dict,
    ax: plt.Axes,
    title: str,
    score: float,
):
    node_colors = []
    node_sizes  = []
    for node in G.nodes:
        if node in assembled:
            node_colors.append("#2ecc71")    # assembled: green
            node_sizes.append(800)
        elif node == candidate:
            node_colors.append("#e74c3c")    # candidate: red
            node_sizes.append(1000)
        else:
            node_colors.append("#bdc3c7")    # pending: gray
            node_sizes.append(500)

    # Edge widths proportional to SASA
    edges      = list(G.edges(data=True))
    widths     = [d.get("sasa", 0) * 3 + 0.5 for _, _, d in edges]
    edge_colors= ["#2c3e50" if (u in assembled and v in assembled)
                  else "#e67e22" if (u == candidate or v == candidate)
                  else "#ecf0f1"
                  for u, v, _ in edges]

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: chain_ids[i] for i in G.nodes},
                            font_size=9, font_weight="bold", font_color="white")

    ax.set_title(f"{title}\nScore={score:.3f}", fontsize=9, pad=4)
    ax.axis("off")


def plot_assembly_steps(cplx, outpath: str, condition: dict):
    """Show graph state at each assembly step."""
    N     = len(cplx.chain_ids)
    w     = {k: v for k, v in condition.items() if k.startswith("w_")}
    result = greedy_assembly(cplx, **w)
    order  = result.predicted_order

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    sasa = _ensure(cplx.sasa_matrix, N)
    for i in range(N):
        for j in range(i+1, N):
            if sasa[i, j] > 0:
                G.add_edge(i, j, sasa=sasa[i,j] / (sasa.max() + 1e-8))

    # Fixed spring layout
    try:
        pos = nx.spring_layout(G, seed=42, k=2.0)
    except Exception:
        pos = {i: (np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)) for i in range(N)}

    fig, axes = plt.subplots(1, N, figsize=(N * 2.8, 3.2))
    if N == 1:
        axes = [axes]

    for step_idx, ax in enumerate(axes):
        if step_idx == 0:
            assembled = [order[0]]
            candidate = order[0]
            score     = 0.0
        else:
            assembled = order[:step_idx]
            candidate = order[step_idx]
            score     = candidate_score(
                candidate, assembled,
                sasa_matrix=_ensure(cplx.sasa_matrix, N),
                ppi_matrix=_ensure(cplx.ppi_matrix, N),
                esm_matrix=_ensure(cplx.esm_matrix, N),
                go_matrix=_ensure(cplx.go_matrix, N),
                **w,
            )

        draw_assembly_step(
            G, assembled, candidate, cplx.chain_ids, pos, ax,
            f"Step {step_idx + 1}: Add {cplx.chain_ids[candidate]}", score
        )

    # Legend
    green_p = mpatches.Patch(color="#2ecc71", label="Assembled")
    red_p   = mpatches.Patch(color="#e74c3c", label="Being added")
    gray_p  = mpatches.Patch(color="#bdc3c7", label="Pending")
    fig.legend(handles=[green_p, red_p, gray_p], loc="lower center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    name  = condition.get("name", "")
    cname = cplx.name.split("_", 1)[-1]
    fig.suptitle(f"{cname} — Assembly Steps ({name})", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. Step diagnostic — all candidates + per-term scores at every step
# ═══════════════════════════════════════════════════════════════════════════════

def _term_scores(candidate: int, assembled: list[int], cplx, N: int) -> dict:
    """Return raw (un-weighted) per-term scores for one candidate."""
    sasa_mat = _ensure(cplx.sasa_matrix, N)
    esm_mat  = _ensure(cplx.esm_matrix,  N)
    go_mat   = _ensure(cplx.go_matrix,   N)
    ppi_mat  = _ensure(cplx.ppi_matrix,  N)
    if not assembled:
        return {"sasa": 0.0, "esm": 0.0, "go": 0.0, "ppi": 0.0}
    sasa_raw = sasa_mat[candidate, assembled].max()
    sasa_max = sasa_mat.max()
    return {
        "sasa": float(sasa_raw / (sasa_max + 1e-8)),
        "esm":  float(esm_mat[candidate, assembled].max()),
        "go":   float(go_mat[candidate, assembled].max()),
        "ppi":  float(ppi_mat[candidate, assembled].max()),
    }


def plot_step_diagnostic(cplx, outpath: str,
                         w_sasa=0.33, w_esm=0.33, w_go=0.33, w_ppi=0.0):
    """
    Diagnostic visualization: for every assembly step, show ALL remaining
    candidates with stacked per-term scores.

    Bar segments = weighted contributions (w * raw_score):
      Blue=SASA, Green=ESM, Red=GO.  Total bar length = combined score.

    Yellow outline = chosen (highest combined).
    Orange outline = SASA-preferred (when it differs from chosen) — the
      'term conflict' that reveals where multi-term scoring adds value.
    """
    N = len(cplx.chain_ids)
    result = greedy_assembly(cplx, w_sasa=w_sasa, w_esm=w_esm,
                             w_go=w_go, w_ppi=w_ppi)
    order = result.predicted_order

    steps_data = []
    for step_idx in range(1, N):
        assembled = order[:step_idx]
        chosen    = order[step_idx]
        remaining = [i for i in range(N) if i not in assembled]
        terms     = {c: _term_scores(c, assembled, cplx, N) for c in remaining}
        combined  = {c: (w_sasa * t["sasa"] + w_esm * t["esm"] +
                         w_go   * t["go"]   + w_ppi * t["ppi"])
                     for c, t in terms.items()}
        sasa_pref = max(remaining, key=lambda c: terms[c]["sasa"])
        steps_data.append({
            "step": step_idx, "assembled": assembled,
            "chosen": chosen, "remaining": remaining,
            "terms": terms, "combined": combined,
            "sasa_preferred": sasa_pref,
            "conflict": sasa_pref != chosen,
        })

    n_steps = len(steps_data)
    # Each step: left margin for labels + bar area
    fig_w = max(12, n_steps * 3.2)
    fig, axes = plt.subplots(1, n_steps, figsize=(fig_w, 5.8))
    if n_steps == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")

    SASA_CLR     = "#3498db"
    ESM_CLR      = "#2ecc71"
    GO_CLR       = "#e74c3c"
    CHOSEN_CLR   = "#f1c40f"
    SASA_CONF    = "#e67e22"
    IDLE_CLR     = "#7f8c8d"
    BG           = "#0d1117"
    PANEL_BG     = "#161b22"
    GRID_CLR     = "#21262d"

    for ax, sd in zip(axes, steps_data):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="#e6edf3", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(axis="x", color=GRID_CLR, alpha=0.6, linewidth=0.5)

        # Sort by combined descending
        rem = sorted(sd["remaining"], key=lambda c: sd["combined"][c], reverse=True)
        max_score = max(sd["combined"].values()) if sd["combined"] else 1.0

        for yi, cand in enumerate(rem):
            t   = sd["terms"][cand]
            comb= sd["combined"][cand]
            is_chosen    = cand == sd["chosen"]
            is_sasa_pref = sd["conflict"] and cand == sd["sasa_preferred"]

            # Weighted contribution segments
            segs = [
                (w_sasa * t["sasa"], SASA_CLR),
                (w_esm  * t["esm"],  ESM_CLR),
                (w_go   * t["go"],   GO_CLR),
            ]
            left = 0.0
            for seg_w, seg_clr in segs:
                if seg_w > 1e-6:
                    ax.barh(yi, seg_w, left=left, height=0.55,
                            color=seg_clr,
                            alpha=0.9 if (is_chosen or is_sasa_pref) else 0.55,
                            edgecolor="none")
                left += seg_w

            # Outline
            outline_clr = (CHOSEN_CLR if is_chosen else
                           SASA_CONF  if is_sasa_pref else None)
            if outline_clr:
                ax.barh(yi, comb, left=0, height=0.6,
                        color="none", edgecolor=outline_clr, linewidth=2.2)

            # Label (right side of bar, inside panel)
            chain_name = cplx.chain_ids[cand]
            suffix = (" ✓" if is_chosen else
                      " ← SASA" if is_sasa_pref else "")
            lbl_clr = (CHOSEN_CLR if is_chosen else
                       SASA_CONF  if is_sasa_pref else "#8b949e")
            ax.text(comb + 0.02, yi,
                    f"{chain_name}{suffix}  {comb:.3f}",
                    va="center", ha="left", fontsize=7.5, color=lbl_clr)

        ax.set_xlim(0, 1.15)
        ax.set_ylim(-0.8, len(rem) - 0.2)
        ax.set_yticks([])
        ax.set_xlabel("Weighted score  (SASA + ESM + GO)", fontsize=7,
                      color="#e6edf3")

        chosen_name  = cplx.chain_ids[sd["chosen"]]
        sasa_name    = cplx.chain_ids[sd["sasa_preferred"]]
        title_color  = SASA_CONF if sd["conflict"] else "#e6edf3"
        conflict_txt = (f"\n⚠ SASA prefers {sasa_name}" if sd["conflict"] else "")
        ax.set_title(
            f"Step {sd['step']+1}: +{chosen_name}"
            f"\nScore={sd['combined'][sd['chosen']]:.3f}{conflict_txt}",
            fontsize=8, color=title_color, pad=4,
        )

    # Legend
    patches = [
        mpatches.Patch(color=SASA_CLR,   label=f"SASA (w={w_sasa})"),
        mpatches.Patch(color=ESM_CLR,    label=f"ESM cos (w={w_esm})"),
        mpatches.Patch(color=GO_CLR,     label=f"GO (w={w_go})"),
        mpatches.Patch(color=CHOSEN_CLR, label="Chosen by multi-term"),
        mpatches.Patch(color=SASA_CONF,  label="SASA-only preference (conflict)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               labelcolor="#e6edf3", facecolor=PANEL_BG, edgecolor=GRID_CLR,
               bbox_to_anchor=(0.5, -0.04))

    title_name = cplx.name.split("_", 1)[-1] if "_" in cplx.name else cplx.name
    fig.suptitle(
        f"{title_name} — Step Diagnostic: All Candidates + Per-Term Scores",
        color="#e6edf3", fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Score contribution breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def plot_score_breakdown(cplx, outpath: str):
    """
    For each assembly step (greedy, all terms), show how each term
    contributes to the score of the chosen subunit.
    """
    N      = len(cplx.chain_ids)
    result = greedy_assembly(cplx, w_sasa=0.33, w_ppi=0.0, w_esm=0.33, w_go=0.33)
    order  = result.predicted_order

    steps    = []
    sasa_s   = []
    esm_s    = []
    go_s     = []

    sasa = _ensure(cplx.sasa_matrix, N)
    esm  = _ensure(cplx.esm_matrix,  N)
    go   = _ensure(cplx.go_matrix,   N)

    for step_idx in range(1, N):
        assembled = order[:step_idx]
        candidate = order[step_idx]

        s_sasa = sasa[candidate, assembled].max() / (sasa.max() + 1e-8)
        s_esm  = esm[candidate, assembled].max()
        s_go   = go[candidate, assembled].max()

        steps.append(f"→{cplx.chain_ids[candidate]}")
        sasa_s.append(s_sasa)
        esm_s.append(s_esm)
        go_s.append(s_go)

    x    = np.arange(len(steps))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(max(5, len(steps) * 1.1), 4))

    ax.bar(x - w, sasa_s, w, label="SASA (Marsh)", color="#3498db")
    ax.bar(x,     esm_s,  w, label="ESM cosine",   color="#2ecc71")
    ax.bar(x + w, go_s,   w, label="GO coherence", color="#e74c3c")

    ax.set_xticks(x); ax.set_xticklabels(steps)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Normalized score contribution")
    ax.set_title(f"{cplx.name.split('_',1)[-1]} — Score breakdown per step")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pathway comparison: 4 conditions side by side for 2HHB
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pathway_comparison(cplx, outpath: str):
    """Show greedy assembly order under 4 conditions."""
    conditions = [
        {"name": "SASA baseline", "w_sasa": 1.0, "w_esm": 0.0, "w_go": 0.0, "w_ppi": 0.0},
        {"name": "ESM only",      "w_sasa": 0.0, "w_esm": 1.0, "w_go": 0.0, "w_ppi": 0.0},
        {"name": "GO only",       "w_sasa": 0.0, "w_esm": 0.0, "w_go": 1.0, "w_ppi": 0.0},
        {"name": "All terms",     "w_sasa": 0.33,"w_esm": 0.33,"w_go": 0.33,"w_ppi": 0.0},
    ]
    N = len(cplx.chain_ids)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)

    for ax, cond in zip(axes, conditions):
        w = {k: v for k, v in cond.items() if k.startswith("w_")}
        result = greedy_assembly(cplx, **w)
        order  = result.predicted_order
        tau    = result.kendall_tau

        step_of = {sub: step for step, sub in enumerate(order)}
        chain_steps = [step_of[i] for i in range(N)]

        colors = ["#2ecc71" if s == 0 else "#3498db" for s in chain_steps]
        bars = ax.bar(range(N), chain_steps, color=colors, edgecolor="white")
        ax.set_xticks(range(N))
        ax.set_xticklabels(cplx.chain_ids)
        ax.set_title(f"{cond['name']}\nτ={tau:.2f}" if tau is not None else cond["name"])
        ax.set_ylabel("Assembly step")
        ax.set_ylim(-0.5, N + 0.5)

        # GT annotation
        if cplx.ground_truth_order:
            gt_steps = {sub: step for step, sub in enumerate(cplx.ground_truth_order)}
            for i in range(N):
                ax.plot(i, gt_steps[i], "k*", ms=10, zorder=10)

    fig.suptitle(
        f"Assembly Order — {cplx.name.split('_',1)[-1]}\n★ = ground truth",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ESM embedding PCA — subunit similarity space
# ═══════════════════════════════════════════════════════════════════════════════

def plot_esm_pca(outpath: str):
    """PCA of all subunit ESM embeddings, colored by complex."""
    from sklearn.decomposition import PCA

    # Load all embeddings from the .npz
    embed_file = EMBED_DIR / "embeddings.npz"
    data = np.load(str(embed_file))
    all_embs, all_names = [], []
    for key in sorted(data.files):
        all_embs.append(data[key])
        all_names.append(key)

    mat = np.array(all_embs)  # (n_chains, 320)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat)

    # Color by PDB ID
    pdb_ids = [n.split("_")[0] for n in all_names]
    unique_pdbs = sorted(set(pdb_ids))
    cmap_pca = plt.cm.tab20
    color_map = {p: cmap_pca(i / max(len(unique_pdbs)-1, 1)) for i, p in enumerate(unique_pdbs)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for pid in unique_pdbs:
        idxs = [i for i, p in enumerate(pdb_ids) if p == pid]
        xs   = coords[idxs, 0]
        ys   = coords[idxs, 1]
        ax.scatter(xs, ys, color=color_map[pid], s=80, alpha=0.8, label=pid, edgecolors="white", lw=0.5)
        # Label each point
        for i, idx in enumerate(idxs):
            chain = all_names[idx].split("_")[-1]
            ax.annotate(chain, (xs[i], ys[i]), fontsize=6.5,
                        xytext=(2, 2), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("ESM-2 8M Embedding PCA — All Benchmark Subunits")
    ax.legend(fontsize=7, ncol=2, loc="lower right",
              bbox_to_anchor=(1.0, 0.0))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ESM PCA: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NPC pathway tree
# ═══════════════════════════════════════════════════════════════════════════════

def plot_npc_pathway_tree(outpath: str, n_runs: int = 500):
    """
    Visualize the NPC assembly as a tree: shared prefixes → branching points.
    Shows the most common transitions at each step.
    """
    import json as _json

    # Load results
    with open(RESULTS / "alpha_results.json") as f:
        results = _json.load(f)

    # Rebuild 3F3F complex
    with open(ROOT / "data" / "complexes" / "npc" / "manifest.json") as f:
        npc_manifest = _json.load(f)
    from run_alpha import NPC_DIR, NPC_EMBED

    pdb_id   = "3F3F"
    pdb_path = str(NPC_DIR / f"{pdb_id}.pdb")
    seq_map  = npc_manifest.get(pdb_id, {}).get("sequences", {})
    chains   = sorted(seq_map.keys())[:8]
    sequences = [seq_map.get(c, "M" * 50) for c in chains]

    from src.simulation.assembly_sim import Complex
    sasa_mat = compute_sasa_matrix(pdb_path, chains)
    esm_mat  = load_esm_matrix(pdb_id, chains, NPC_EMBED)
    go_mat   = compute_go_matrix(sequences)

    npc_cplx = Complex(
        name="NPC_3F3F",
        sequences=sequences,
        chain_ids=chains,
        pdb_path=pdb_path,
        sasa_matrix=sasa_mat,
        ppi_matrix=np.zeros((len(chains), len(chains))),
        esm_matrix=esm_mat,
        go_matrix=go_mat,
        ground_truth_order=None,
    )

    run_results, stats = ensemble_assembly(
        npc_cplx, n_runs=n_runs, temperature=0.5,
        w_sasa=0.33, w_esm=0.33, w_go=0.33
    )

    # Build transition frequency matrix for first 3 steps
    N = len(chains)
    fig, axes = plt.subplots(1, N-1, figsize=((N-1)*3, 4))

    for step in range(N-1):
        ax = axes[step]
        # Count what goes at each position conditional on position 0 being 'A'
        transitions: dict[tuple, int] = {}
        for r in run_results:
            key = tuple(r.predicted_order[:step+2])
            transitions[key] = transitions.get(key, 0) + 1

        # Top 8 prefixes
        top = sorted(transitions.items(), key=lambda x: -x[1])[:8]
        total = sum(v for _, v in top)
        labels = [f"{'→'.join(chains[i] for i in k)}" for k, _ in top]
        counts = [v / n_runs for _, v in top]

        colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(counts)))
        ax.barh(range(len(counts)), counts, color=colors_bar, edgecolor="white")
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top pathways\nstep 1→{step+2}", fontsize=8)
        ax.axvline(1/len(transitions), color="gray", ls="--", lw=1, alpha=0.6)

    fig.suptitle(f"NPC 3F3F (Nup-SEH1/Nup85) Assembly Pathway Frequencies (n={n_runs})",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  NPC unique pathways: {stats['n_unique_pathways']}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Comprehensive summary figure
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary_figure(outpath: str):
    """
    4-panel summary: (1) ablation bars, (2) homo vs hetero τ,
    (3) step entropy NPC, (4) key findings text.
    """
    with open(RESULTS / "alpha_results.json") as f:
        data = json.load(f)

    summary = data["ablation_summary"]
    conditions  = [r["condition"] for r in summary]
    homo_taus   = [r.get("mean_tau_homo")   or 0 for r in summary]
    hetero_taus = [r.get("mean_tau_hetero") or 0 for r in summary]

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

    # Panel 1: ablation bars (homo + hetero)
    ax1 = fig.add_subplot(gs[0, :2])
    y = np.arange(len(conditions))
    h = 0.35
    b1 = ax1.barh(y + h/2, homo_taus,   h, label="Homomers",  color="#3498db", alpha=0.85)
    b2 = ax1.barh(y - h/2, hetero_taus, h, label="Heteromers",color="#e74c3c", alpha=0.85)
    ax1.axvline(homo_taus[0],   color="#3498db", ls="--", lw=1, alpha=0.5)
    ax1.axvline(hetero_taus[0], color="#e74c3c", ls="--", lw=1, alpha=0.5)
    ax1.set_yticks(y); ax1.set_yticklabels(conditions, fontsize=9)
    ax1.set_xlabel("Mean Kendall's τ"); ax1.set_title("Ablation Study — Assembly Order Kendall's τ")
    ax1.legend(fontsize=9); ax1.set_xlim(0, 1.1)

    # Panel 2: Δτ relative to SASA baseline
    ax2 = fig.add_subplot(gs[0, 2])
    delta_homo   = [t - homo_taus[0]   for t in homo_taus]
    delta_hetero = [t - hetero_taus[0] for t in hetero_taus]
    colors_d = ["#2ecc71" if d > 0.01 else "#e74c3c" if d < -0.01 else "#95a5a6"
                for d in delta_homo]
    ax2.barh(y + h/2, delta_homo,   h, color=colors_d, alpha=0.85, label="Homomers")
    colors_e = ["#27ae60" if d > 0.01 else "#c0392b" if d < -0.01 else "#95a5a6"
                for d in delta_hetero]
    ax2.barh(y - h/2, delta_hetero, h, color=colors_e, alpha=0.6, label="Heteromers")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels([""] * len(y))
    ax2.set_xlabel("Δτ vs SASA baseline")
    ax2.set_title("Δ from Marsh Baseline")
    ax2.legend(fontsize=8)

    # Panel 3: Per-complex best τ bar
    ax3 = fig.add_subplot(gs[1, :2])
    per_c = data["per_complex"]
    names_p  = [n.split("_", 1)[-1] for n in per_c]
    best_tau = []
    sasa_tau = []
    for name, conds in per_c.items():
        sasa_t = conds.get("SASA only (Marsh baseline)", {}).get("greedy_tau", 0) or 0
        best_t = max((v.get("greedy_tau") or 0) for v in conds.values())
        best_tau.append(best_t)
        sasa_tau.append(sasa_t)
    y2 = np.arange(len(names_p))
    ax3.barh(y2 + h/2, best_tau, h, color="#2ecc71", alpha=0.85, label="Best condition")
    ax3.barh(y2 - h/2, sasa_tau, h, color="#3498db", alpha=0.6,  label="SASA baseline")
    ax3.set_yticks(y2); ax3.set_yticklabels(names_p, fontsize=8)
    ax3.set_xlabel("Kendall's τ"); ax3.set_title("Per-Complex: Best vs SASA Baseline")
    ax3.legend(fontsize=9); ax3.set_xlim(-0.1, 1.1)
    ax3.axvline(0, color="black", lw=0.5)

    # Panel 4: Key findings
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    findings = [
        "KEY FINDINGS (Alpha Phase)",
        "",
        "1. SASA (Marsh baseline): τ=0.667 homo,\n   0.958 hetero — strong for heteromers.",
        "",
        "2. ESM cosine: τ=1.000 homo, 0.833 hetero\n   — perfect for symmetric homomers.",
        "",
        "3. GO coherence: τ=1.000 homo, 0.917 hetero\n   — edges ESM on heteromers (+0.084).",
        "",
        "4. NPC 3F3F (8-mer): 473/500 unique paths\n   — highly stochastic; step 3 is bottleneck.",
        "",
        "5. ProtFunc model is out-of-distribution for\n   human proteins (insect-trained).",
        "   GO term needs domain transfer to fully\n   test heteromer hypothesis.",
    ]
    ax4.text(0.02, 0.98, "\n".join(findings), va="top", ha="left",
             transform=ax4.transAxes, fontsize=8.5,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))

    fig.suptitle("Protein Assembly Simulation — Alpha Phase Results", fontsize=13)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import json as _json
    print("═" * 60)
    print("  GNN Graph Visualizations")
    print("═" * 60)

    # Build complexes for visualization
    key_complexes = ["2HHB", "1AON", "3GBN"]
    complexes_viz = {}
    for pdb_id in key_complexes:
        meta = BENCHMARK.get(pdb_id)
        if not meta:
            continue
        try:
            cplx = build_complex(pdb_id, meta, EMBED_DIR)
            complexes_viz[pdb_id] = cplx
        except Exception as e:
            print(f"  WARN {pdb_id}: {e}")

    # 1. Assembly step graphs for key complexes
    conditions_to_show = [
        {"name": "SASA baseline",  "w_sasa": 1.0, "w_esm": 0.0, "w_go": 0.0, "w_ppi": 0.0},
        {"name": "All terms",      "w_sasa": 0.33,"w_esm": 0.33,"w_go": 0.33,"w_ppi": 0.0},
    ]
    for pdb_id, cplx in complexes_viz.items():
        if len(cplx.chain_ids) <= 5:
            for cond in conditions_to_show:
                fname = GNN_DIR / f"{pdb_id}_steps_{cond['name'].replace(' ','_')}.png"
                print(f"  Generating assembly steps: {fname.name}")
                try:
                    plot_assembly_steps(cplx, str(fname), cond)
                except Exception as e:
                    print(f"  WARN: {e}")

    # 1b. Step diagnostic — all candidates + term conflicts (key diagnostic figure)
    for pdb_id, cplx in complexes_viz.items():
        fname = GNN_DIR / f"{pdb_id}_step_diagnostic.png"
        print(f"  Step diagnostic: {fname.name}")
        try:
            plot_step_diagnostic(cplx, str(fname), w_sasa=0.33, w_esm=0.33, w_go=0.33)
        except Exception as e:
            print(f"  WARN step diagnostic {pdb_id}: {e}")

    # 2. Score breakdown
    for pdb_id, cplx in complexes_viz.items():
        if len(cplx.chain_ids) >= 3:
            fname = GNN_DIR / f"{pdb_id}_score_breakdown.png"
            print(f"  Score breakdown: {fname.name}")
            try:
                plot_score_breakdown(cplx, str(fname))
            except Exception as e:
                print(f"  WARN: {e}")

    # 3. Pathway comparison (2HHB — most interesting heteromer)
    if "2HHB" in complexes_viz:
        fname = GNN_DIR / "2HHB_pathway_comparison.png"
        print(f"  Pathway comparison: {fname.name}")
        plot_pathway_comparison(complexes_viz["2HHB"], str(fname))

    # 4. ESM PCA
    fname = GNN_DIR / "esm_pca_all_subunits.png"
    print(f"  ESM PCA: {fname.name}")
    try:
        plot_esm_pca(str(fname))
    except Exception as e:
        print(f"  WARN ESM PCA: {e}")

    # 5. NPC pathway tree
    fname = GNN_DIR / "npc_pathway_tree.png"
    print(f"  NPC pathway tree: {fname.name}")
    try:
        stats = plot_npc_pathway_tree(str(fname), n_runs=300)
    except Exception as e:
        print(f"  WARN NPC tree: {e}")
        import traceback; traceback.print_exc()

    # 6. Summary figure
    fname = RESULTS / "summary_figure.png"
    print(f"  Summary figure: {fname.name}")
    plot_summary_figure(str(fname))

    print("\n═══ GNN Viz Done ═══")
    print(f"Output → {GNN_DIR}")
    files = list(GNN_DIR.glob("*.png")) + list(RESULTS.glob("*.png"))
    for f in sorted(set(files)):
        print(f"  {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
