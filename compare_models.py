"""
Compare WeightLearner vs GNN training results side-by-side.
Reads results/training/training_log.json and results/gnn_training/training_log.json.
Outputs results/comparison.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent
WL_LOG  = ROOT / "results" / "training" / "training_log.json"
GNN_LOG = ROOT / "results" / "gnn_training" / "training_log.json"
OUT     = ROOT / "results" / "comparison.png"


def load_log(path):
    """Returns list of per-epoch dicts regardless of file format."""
    if not path.exists():
        return []
    raw = json.load(open(path))
    if isinstance(raw, list):
        return raw  # GNN format: [{epoch, loss, ...}, ...]
    # WL format: {loss:[...], tau:[...], ...}
    list_keys = {k: v for k, v in raw.items() if isinstance(v, list)}
    n = max(len(v) for v in list_keys.values())
    return [
        {k: v[i] if i < len(v) else None for k, v in list_keys.items()} | {"epoch": i + 1}
        for i in range(n)
    ]


def main():
    wl  = load_log(WL_LOG)
    gnn = load_log(GNN_LOG)

    if not wl and not gnn:
        print("No training logs found.")
        return

    clr = {"bg": "#0d1117", "panel": "#161b22", "grid": "#21262d",
           "fg": "#e6edf3", "wl": "#58a6ff", "gnn": "#3fb950",
           "loss": "#f85149", "prob": "#ffa657"}

    fig = plt.figure(figsize=(18, 12), facecolor=clr["bg"])
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    def _ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(clr["panel"])
        ax.tick_params(colors=clr["fg"], labelsize=8)
        ax.xaxis.label.set_color(clr["fg"]); ax.yaxis.label.set_color(clr["fg"])
        ax.title.set_color(clr["fg"])
        for sp in ax.spines.values(): sp.set_edgecolor(clr["grid"])
        ax.grid(color=clr["grid"], alpha=0.4, linewidth=0.5)
        return ax

    # ── Row 0: Loss comparison ───────────────────────────────────────────────
    ax = _ax(gs[0, :2])
    if wl:
        E = [d["epoch"] for d in wl]
        ax.plot(E, [d["loss"] for d in wl], color=clr["wl"], lw=1.5, label="WeightLearner loss")
    if gnn:
        E = [d["epoch"] for d in gnn]
        ax.plot(E, [d["loss"] for d in gnn], color=clr["gnn"], lw=1.5, label="GAT GNN loss")
    ax.set_title("A. Training Loss — WL vs GNN", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("CE Loss")
    ax.legend(fontsize=8); ax.set_ylim(bottom=0)

    # ── Row 0: GT prob comparison ────────────────────────────────────────────
    ax = _ax(gs[0, 2])
    if wl:
        pts = [(d["epoch"], d["gt_prob"]) for d in wl if d.get("gt_prob") is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=clr["wl"], lw=2, label="WL")
    if gnn:
        pts = [(d["epoch"], d["gt_prob"]) for d in gnn if d.get("gt_prob") is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=clr["gnn"], lw=2, label="GNN")
    ax.set_title("B. P(GT next step)", fontsize=10)
    ax.set_ylim(0, 1.0); ax.legend(fontsize=8)

    # ── Row 1: Kendall's τ ───────────────────────────────────────────────────
    ax = _ax(gs[1, :2])
    for log_data, color, label in [(wl, clr["wl"], "WeightLearner"), (gnn, clr["gnn"], "GAT GNN")]:
        pts = [(d["epoch"], d["mean_tau"]) for d in log_data if d.get("mean_tau") is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=color, lw=2, label=label)
    ax.set_title("C. Mean Kendall's τ (all complexes)", fontsize=10)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("τ")

    # ── Row 1: Homo vs Hetero τ (GNN) ───────────────────────────────────────
    ax = _ax(gs[1, 2])
    if gnn:
        homo   = [(d["epoch"], d["homo_tau"])   for d in gnn if d.get("homo_tau")   is not None]
        hetero = [(d["epoch"], d["hetero_tau"]) for d in gnn if d.get("hetero_tau") is not None]
        mean   = [(d["epoch"], d["mean_tau"])   for d in gnn if d.get("mean_tau")   is not None]
        if homo:   ax.plot([p[0] for p in homo],   [p[1] for p in homo],   color="#d2a679", lw=1.5, label="Homo")
        if hetero: ax.plot([p[0] for p in hetero], [p[1] for p in hetero], color="#bc8cff", lw=1.5, label="Hetero")
        if mean:   ax.plot([p[0] for p in mean],   [p[1] for p in mean],   color=clr["gnn"],  lw=2,   label="All")
    ax.set_title("D. GNN τ by type", fontsize=10)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

    # ── Row 2: Final per-complex τ bar chart ─────────────────────────────────
    ax = _ax(gs[2, :])
    COMPLEXES = ["2HHB","1BRS","1AY7","1TGS","2PTC","1A2K","1SBB","3GBN","1AON","1GRU"]
    x = np.arange(len(COMPLEXES))
    w = 0.35

    def final_tau_per_complex(log_data):
        # Find last entry with per-complex tau (stored in separate results.json if available)
        res_path = ROOT / "results" / ("training" if log_data is wl else "gnn_training") / "results.json"
        if res_path.exists():
            res = json.load(open(res_path))
            tau_data = res.get("loocv_results", res.get("per_complex_tau", {}))
            return [tau_data.get(c, {}).get("tau", tau_data.get(c, 0.0)) for c in COMPLEXES]
        # Fall back to last training log entry
        if log_data:
            last_tau = [d.get("mean_tau", 0.9333) for d in log_data if d.get("mean_tau")]
            return [last_tau[-1] if last_tau else 0.0] * len(COMPLEXES)
        return [0.0] * len(COMPLEXES)

    wl_taus  = final_tau_per_complex(wl)
    gnn_taus = final_tau_per_complex(gnn)

    bars1 = ax.bar(x - w/2, wl_taus,  w, label="WeightLearner", color=clr["wl"],  alpha=0.8)
    bars2 = ax.bar(x + w/2, gnn_taus, w, label="GAT GNN",       color=clr["gnn"], alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(COMPLEXES, rotation=30, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Kendall's τ")
    ax.axhline(1.0, color="white", ls="--", lw=0.8, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("E. Per-complex τ — WeightLearner vs GNN", fontsize=10)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=6, color=clr["fg"])
    for bar in bars2:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=6, color=clr["fg"])

    # ── Summary stats table ──────────────────────────────────────────────────
    wl_last  = wl[-1]  if wl  else {}
    gnn_last = gnn[-1] if gnn else {}
    fig.suptitle(
        f"Model Comparison  |  WL: τ={wl_last.get('mean_tau',0):.4f} p={wl_last.get('gt_prob',0):.4f}"
        f"  |  GNN: τ={gnn_last.get('mean_tau',0):.4f} p={gnn_last.get('gt_prob',0):.4f}",
        color=clr["fg"], fontsize=12, fontweight="bold", y=0.99
    )

    plt.savefig(OUT, dpi=130, bbox_inches="tight", facecolor=clr["bg"])
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
