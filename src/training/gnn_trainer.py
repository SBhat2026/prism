"""
GNN training loop for assembly scorer.
Uses the WeightLearner architecture (interpretable) with options for MLP and full GNN.

Outputs per epoch to results/training/epochs/:
  epoch_{n:04d}.png  — dashboard snapshot (loss + tau + weights + graph viz)

Also maintains:
  results/training/training_log.json  — all metrics history
  results/training/best_model.pt      — best τ checkpoint
"""

import os
import sys
import json
import time
import copy
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent.parent
TRAIN_DIR  = ROOT / "results" / "training"
EPOCH_DIR  = TRAIN_DIR / "epochs"
EPOCH_DIR.mkdir(parents=True, exist_ok=True)


# ── Dashboard rendering ───────────────────────────────────────────────────────

def render_epoch_dashboard(
    epoch: int,
    history: dict,
    model,
    dataset,
    best_epoch: int,
    outdir: Path = EPOCH_DIR,
):
    """
    Save one epoch snapshot PNG with:
      Panel A: Loss curve
      Panel B: Mean Kendall's τ (train)
      Panel C: Learned weight evolution (4 bars animated)
      Panel D: Per-complex τ bar chart this epoch
      Panel E: Complex graph colored by current scores (2HHB)
      Panel F: Weight magnitude over epochs (line)
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    epochs_so_far = list(range(1, len(history["loss"]) + 1))

    # ── A: Loss curve ────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(epochs_so_far, history["loss"], "b-", lw=1.5, label="Train loss")
    if "val_loss" in history and any(v is not None for v in history["val_loss"]):
        ax_a.plot(epochs_so_far,
                  [v if v is not None else float("nan") for v in history["val_loss"]],
                  "r--", lw=1.5, label="Val loss", alpha=0.8)
    ax_a.axvline(best_epoch, color="green", ls=":", lw=1.5, label=f"Best epoch ({best_epoch})")
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("CE Loss")
    ax_a.set_title("A.  Training Loss", loc="left", fontsize=10)
    ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)
    ax_a.set_ylim(bottom=0)

    # ── B: GT probability + τ ────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b2 = ax_b.twinx()

    if "gt_prob" in history and history["gt_prob"]:
        ax_b.plot(epochs_so_far[:len(history["gt_prob"])], history["gt_prob"],
                  "purple", lw=2, label="P(GT next step)", alpha=0.9)
    ax_b.axhline(1.0 / 3, color="purple", ls=":", lw=1, alpha=0.5, label="Random P (1/k)")
    ax_b.set_ylabel("P(correct next subunit)", color="purple")
    ax_b.tick_params(axis="y", labelcolor="purple")
    ax_b.set_ylim(0, 1.05)

    ax_b2.plot(epochs_so_far, history["mean_tau"], "g-", lw=1.5, label="Mean τ", alpha=0.7)
    if history.get("homo_tau"):
        ax_b2.plot(epochs_so_far, history["homo_tau"],   "b--", lw=1,  label="Homo τ",   alpha=0.6)
        ax_b2.plot(epochs_so_far, history["hetero_tau"], "r--", lw=1,  label="Hetero τ", alpha=0.6)
    ax_b2.axhline(history["baseline_tau"], color="gray", ls="--", lw=1,
                  label=f"SASA τ={history['baseline_tau']:.3f}")
    ax_b2.set_ylabel("Kendall's τ", color="green")
    ax_b2.tick_params(axis="y", labelcolor="green")
    ax_b2.set_ylim(-0.1, 1.15)
    ax_b2.axvline(best_epoch, color="green", ls=":", lw=1.5)

    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines1+lines2, labels1+labels2, fontsize=7, loc="lower right")
    ax_b.set_xlabel("Epoch")
    ax_b.set_title("B.  GT Probability + τ", loc="left", fontsize=10)
    ax_b.grid(alpha=0.3)

    # ── C: Current learned weights ───────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    w = model.named_weights()
    names  = ["SASA", "ESM", "GO", "PPI"]
    values = [w["w_sasa"], w["w_esm"], w["w_go"], w["w_ppi"]]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
    bars   = ax_c.bar(names, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                  f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax_c.set_ylabel("Weight (normalized)"); ax_c.set_ylim(0, 1.0)
    ax_c.set_title(f"C.  Learned Weights (epoch {epoch})", loc="left", fontsize=10)
    ax_c.grid(axis="y", alpha=0.3)

    # ── D: Per-complex τ this epoch ──────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    if history["per_complex_tau"]:
        last_tau = history["per_complex_tau"][-1]
        pdb_ids  = list(last_tau.keys())
        tau_vals = list(last_tau.values())
        type_colors = []
        for pid in pdb_ids:
            for cdata in dataset:
                if cdata.pdb_id == pid:
                    type_colors.append("#e74c3c" if cdata.complex_type=="heteromer" else "#3498db")
                    break
            else:
                type_colors.append("gray")
        bars_d = ax_d.bar(pdb_ids, tau_vals, color=type_colors, edgecolor="white")
        ax_d.axhline(0, color="black", lw=0.5)
        ax_d.set_ylabel("Kendall's τ"); ax_d.set_ylim(-1.1, 1.2)
        ax_d.set_xticklabels(pdb_ids, rotation=35, ha="right", fontsize=8)
        ax_d.set_title("D.  Per-Complex τ (current epoch)", loc="left", fontsize=10)
        for bar, val in zip(bars_d, tau_vals):
            ax_d.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                      f"{val:.2f}", ha="center", fontsize=7)
        ax_d.grid(axis="y", alpha=0.3)

    # ── E: Complex graph colored by current SASA×weight scores ───────────────
    ax_e = fig.add_subplot(gs[1, 1])
    _draw_scored_graph(ax_e, model, dataset, target_pdb="2HHB",
                       title="E.  2HHB: Current Score Coloring")

    # ── F: Weight evolution lines ─────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    if len(history["weight_history"]) > 1:
        wh = np.array(history["weight_history"])  # (epochs, 4)
        labels_w = ["w_sasa", "w_esm", "w_go", "w_ppi"]
        colors_w  = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
        for i, (lbl, col) in enumerate(zip(labels_w, colors_w)):
            ax_f.plot(epochs_so_far[:len(wh)], wh[:, i], color=col, lw=2, label=lbl)
        ax_f.set_xlabel("Epoch"); ax_f.set_ylabel("Weight value")
        ax_f.set_title("F.  Weight Evolution", loc="left", fontsize=10)
        ax_f.legend(fontsize=8); ax_f.grid(alpha=0.3)
        ax_f.set_ylim(0, 1.0)

    fig.suptitle(
        f"Assembly GNN Training  |  Epoch {epoch}  |  "
        f"Loss={history['loss'][-1]:.4f}  τ={history['mean_tau'][-1]:.3f}",
        fontsize=12,
    )
    out = outdir / f"epoch_{epoch:04d}.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close()
    return out


def _draw_scored_graph(ax, model, dataset, target_pdb: str = "2HHB", title: str = ""):
    """Draw complex graph with nodes colored by assembly score contributions."""
    cdata = next((d for d in dataset if d.pdb_id == target_pdb), None)
    if cdata is None or len(cdata.chain_ids) < 2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    N     = len(cdata.chain_ids)
    sasa  = cdata.sasa_matrix / (cdata.sasa_matrix.max() + 1e-8)
    w     = model.named_weights()

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    edge_weights = {}
    for i in range(N):
        for j in range(i+1, N):
            s = (w["w_sasa"] * sasa[i,j] +
                 w["w_esm"]  * cdata.esm_matrix[i,j] +
                 w["w_go"]   * cdata.go_matrix[i,j])
            if s > 0.01:
                G.add_edge(i, j, weight=s)
                edge_weights[(i,j)] = s

    # Node positions — circular for 4-node
    angles = [2 * np.pi * i / N for i in range(N)]
    pos    = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}

    # Compute score of each node if added as second (seed=gt_order[0])
    seed     = cdata.gt_order[0]
    scores   = []
    for i in range(N):
        if i == seed:
            scores.append(0.0)
            continue
        feat = np.array([
            sasa[i, seed], cdata.esm_matrix[i, seed],
            cdata.go_matrix[i, seed], 0.0,
            sasa[i, seed], cdata.esm_matrix[i, seed],
            cdata.go_matrix[i, seed], 0.0,
        ], dtype=np.float32)
        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scores.append(float(model(feat_t).item()))

    max_s = max(scores) if max(scores) > 0 else 1.0
    node_colors = plt.cm.RdYlGn([s / max_s for s in scores])

    # GT order annotation
    gt_labels = {i: f"{cdata.chain_ids[i]}\nstep{cdata.gt_order.index(i)+1}" for i in range(N)}

    edges     = list(G.edges())
    e_weights = [edge_weights.get((min(u,v), max(u,v)), 0) * 8 for u, v in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=e_weights,
                           edge_color="#2c3e50", alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=700, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=gt_labels,
                            font_size=7, font_color="black")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                norm=plt.Normalize(vmin=0, vmax=max_s))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="Score")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    model,
    dataset,
    n_epochs: int = 300,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    snapshot_every: int = 10,
    device: str = "cpu",
    baseline_tau: float = 0.900,
    val_split: Optional[list[str]] = None,
    verbose: bool = True,
):
    """
    Full training loop with epoch snapshots and JSON logging.
    val_split: list of pdb_ids to hold out for validation.
    """
    from src.training.dataset import build_examples, build_step_groups

    # Split train/val
    if val_split:
        train_data = [d for d in dataset if d.pdb_id not in val_split]
        val_data   = [d for d in dataset if d.pdb_id in val_split]
    else:
        train_data = dataset
        val_data   = []

    # Build step groups
    all_train_examples = []
    for cdata in train_data:
        all_train_examples.extend(build_examples(cdata))
    train_groups = build_step_groups(all_train_examples)

    val_groups = []
    if val_data:
        all_val_examples = []
        for cdata in val_data:
            all_val_examples.extend(build_examples(cdata))
        val_groups = build_step_groups(all_val_examples)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr/20)

    history = {
        "loss": [], "val_loss": [], "mean_tau": [],
        "homo_tau": [], "hetero_tau": [],
        "per_complex_tau": [], "weight_history": [],
        "baseline_tau": baseline_tau,
    }
    best_tau   = -1.0
    best_epoch = 1
    best_state = None

    log_path = TRAIN_DIR / "training_log.json"

    for epoch in range(1, n_epochs + 1):
        model.train()
        # Shuffle step groups each epoch
        idxs = np.random.permutation(len(train_groups))
        total_loss = 0.0

        for idx in idxs:
            group    = train_groups[idx]
            feats    = torch.tensor(
                np.stack([ex.features for ex in group]), dtype=torch.float32
            )
            scores   = model(feats)
            correct  = next(i for i, ex in enumerate(group) if ex.is_correct)
            loss     = F.cross_entropy(scores.unsqueeze(0), torch.tensor([correct]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_groups), 1)

        # Val loss
        val_loss = None
        if val_groups:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                for group in val_groups:
                    feats  = torch.tensor(np.stack([ex.features for ex in group]), dtype=torch.float32)
                    scores = model(feats)
                    correct = next(i for i, ex in enumerate(group) if ex.is_correct)
                    vl += F.cross_entropy(scores.unsqueeze(0), torch.tensor([correct])).item()
                val_loss = vl / len(val_groups)

        # Evaluate τ + mean GT probability
        model.eval()
        tau_dict = _eval_tau_all(model, dataset)
        all_taus   = list(tau_dict.values())
        homo_taus  = [v for pid, v in tau_dict.items()
                      if any(d.complex_type=="homomer" for d in dataset if d.pdb_id==pid)]
        hetero_taus= [v for pid, v in tau_dict.items()
                      if any(d.complex_type=="heteromer" for d in dataset if d.pdb_id==pid)]
        mean_tau   = float(np.mean(all_taus)) if all_taus else 0.0

        # Mean probability assigned to GT step (more sensitive than τ)
        gt_probs = []
        with torch.no_grad():
            for group in train_groups:
                feats  = torch.tensor(np.stack([ex.features for ex in group]), dtype=torch.float32)
                scores = model(feats)
                probs  = torch.softmax(scores, dim=0)
                correct = next(i for i, ex in enumerate(group) if ex.is_correct)
                gt_probs.append(float(probs[correct].item()))
        mean_gt_prob = float(np.mean(gt_probs))

        w = model.get_weights().tolist()

        history["loss"].append(float(avg_loss))
        history["val_loss"].append(float(val_loss) if val_loss else None)
        history["mean_tau"].append(float(mean_tau))
        history["homo_tau"].append(float(np.mean(homo_taus)) if homo_taus else 0.0)
        history["hetero_tau"].append(float(np.mean(hetero_taus)) if hetero_taus else 0.0)
        history["per_complex_tau"].append({k: float(v) for k, v in tau_dict.items()})
        history["weight_history"].append([float(x) for x in w])
        if "gt_prob" not in history:
            history["gt_prob"] = []
        history["gt_prob"].append(float(mean_gt_prob))

        if mean_tau > best_tau:
            best_tau   = mean_tau
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 10 == 0 or epoch == 1):
            w_named = model.named_weights()
            print(f"  ep{epoch:4d}  loss={avg_loss:.4f}  gt_prob={mean_gt_prob:.3f}  "
                  f"τ={mean_tau:.3f}  best_τ={best_tau:.3f}@ep{best_epoch}  "
                  f"sasa={w_named['w_sasa']:.2f} esm={w_named['w_esm']:.2f} "
                  f"go={w_named['w_go']:.2f} ppi={w_named['w_ppi']:.2f}")

        if epoch % snapshot_every == 0 or epoch == 1 or epoch == n_epochs:
            render_epoch_dashboard(epoch, history, model, dataset, best_epoch)

    # Save best
    if best_state:
        torch.save({"state_dict": best_state, "epoch": best_epoch,
                    "tau": best_tau, "weights": history["weight_history"][best_epoch-1]},
                   str(TRAIN_DIR / "best_model.pt"))

    # Save log
    with open(log_path, "w") as f:
        json.dump({k: v for k, v in history.items() if k != "per_complex_tau"},
                  f, indent=2)

    return history, best_epoch, best_tau


def _eval_tau_all(model, dataset) -> dict[str, float]:
    from src.simulation.assembly_sim import Complex, greedy_assembly
    results = {}
    for cdata in dataset:
        N = len(cdata.chain_ids)
        sasa = cdata.sasa_matrix / (cdata.sasa_matrix.max() + 1e-8)
        cplx = Complex(
            name=cdata.pdb_id, sequences=cdata.sequences, chain_ids=cdata.chain_ids,
            sasa_matrix=sasa, ppi_matrix=cdata.ppi_matrix,
            esm_matrix=cdata.esm_matrix, go_matrix=cdata.go_matrix,
            ground_truth_order=cdata.gt_order,
        )
        w = model.named_weights()
        r = greedy_assembly(cplx, w_sasa=w["w_sasa"], w_ppi=w["w_ppi"],
                            w_esm=w["w_esm"], w_go=w["w_go"])
        results[cdata.pdb_id] = r.kendall_tau if r.kendall_tau is not None else 0.0
    return results
