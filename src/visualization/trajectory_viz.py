"""
Visualization: assembly trajectory plots, heatmaps, pathway diversity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_assembly_order(
    predicted_orders: list[list[int]],
    ground_truth: Optional[list[int]],
    chain_ids: list[str],
    title: str = "Assembly Trajectory",
    outpath: Optional[str] = None,
):
    """
    Sankey-style step chart showing when each subunit joins.
    Multiple predicted orders shown as overlapping lines.
    """
    N = len(chain_ids)
    fig, ax = plt.subplots(figsize=(max(6, N * 1.2), 4))

    # One line per predicted order
    for order in predicted_orders:
        step_of = {subunit: step for step, subunit in enumerate(order)}
        steps = [step_of[i] for i in range(N)]
        ax.plot(range(N), steps, "o-", alpha=0.2, color="steelblue", lw=1)

    # Mean predicted step for each subunit
    all_steps = np.array([
        [step for _, step in sorted(
            [(s, st) for st, s in enumerate(order)], key=lambda x: x[0]
        )]
        for order in predicted_orders
    ])
    mean_steps = all_steps.mean(axis=0)
    ax.plot(range(N), mean_steps, "o-", color="steelblue", lw=2.5,
            label="Mean predicted", zorder=5)

    # Ground truth
    if ground_truth:
        gt_steps = [ground_truth.index(i) for i in range(N)]
        ax.plot(range(N), gt_steps, "s--", color="firebrick", lw=2,
                label="Ground truth", zorder=6)

    ax.set_xticks(range(N))
    ax.set_xticklabels(chain_ids)
    ax.set_xlabel("Subunit (chain)")
    ax.set_ylabel("Assembly step")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150)
        print(f"Saved → {outpath}")
    else:
        plt.show()
    plt.close()


def plot_scoring_heatmap(
    matrices: dict[str, np.ndarray],
    chain_ids: list[str],
    title: str = "Pairwise Scoring Matrices",
    outpath: Optional[str] = None,
):
    """2×2 grid of SASA / ESM / GO / PPI heatmaps."""
    keys = list(matrices.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        mat = matrices[key]
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=mat.max())
        ax.set_xticks(range(len(chain_ids)))
        ax.set_yticks(range(len(chain_ids)))
        ax.set_xticklabels(chain_ids, rotation=45, ha="right")
        ax.set_yticklabels(chain_ids)
        ax.set_title(key)
        plt.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[len(keys):]:
        ax.set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_pathway_diversity(
    step_entropies: list[float],
    chain_ids: list[str],
    title: str = "Assembly Path Entropy per Step",
    outpath: Optional[str] = None,
):
    """Bar chart of Shannon entropy at each assembly step."""
    fig, ax = plt.subplots(figsize=(max(5, len(step_entropies) * 0.8), 4))
    steps = range(1, len(step_entropies) + 1)
    ax.bar(steps, step_entropies, color="mediumpurple", edgecolor="white")
    ax.set_xticks(list(steps))
    ax.set_xticklabels(chain_ids[1:], rotation=45, ha="right")
    ax.set_xlabel("Step (subunit added)")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(title)
    ax.axhline(np.log2(len(chain_ids)), color="gray", ls="--",
               label="Max entropy (random)")
    ax.legend()
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_ablation_table(
    rows: list[dict],
    outpath: Optional[str] = None,
):
    """Horizontal bar chart comparing Kendall τ across ablation conditions."""
    conditions = [r["condition"] for r in rows]
    greedy_taus = [r.get("greedy_tau") or 0 for r in rows]
    stoch_taus  = [r.get("mean_stochastic_tau") or 0 for r in rows]

    n = len(conditions)
    y = np.arange(n)
    h = 0.35

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.8)))
    ax.barh(y + h/2, greedy_taus, h, label="Greedy τ", color="steelblue")
    ax.barh(y - h/2, stoch_taus,  h, label="Mean stochastic τ", color="tomato")
    ax.set_yticks(y)
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Kendall's τ")
    ax.set_title("Ablation Study — Assembly Order Prediction")
    ax.axvline(0, color="black", lw=0.8)
    ax.legend()
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150)
    else:
        plt.show()
    plt.close()
