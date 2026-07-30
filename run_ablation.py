"""
run_ablation.py — Feature ablation answering PRISM_scaling_roadmap.md §5 Q2:

    "Do ESM-2 sequence embeddings even carry the signal for assembly order, or
     is the order information almost entirely in the structure/interface geometry?"

and, as a by-product, §5 Q1 (is the ring failure a symmetry-breaking artifact?)
via symmetry-aware scoring, plus a first read on Lever 1's chiral descriptors.

Method
------
Every condition trains the *same* AssemblyScorer with the *same* parameter count
on the *same* split with the *same* seed. Conditions differ only by a mask over
the feature channels (features are zeroed, never removed), so no condition gets
a capacity or optimisation advantage. Two untrained references anchor the scale:
the Marsh/Levy BSA teacher and a random-order baseline.

Splits (src/evaluation/splits.py)
  train / test_small : n ≤ 20
  test_large         : n ≥ 21, frozen out of training entirely → extrapolation

Metrics (src/evaluation/symmetry.py)
  tau        exact-order Kendall τ, greedy from a BSA-chosen seed (no GT leak)
  tau_gtseed same but seeded at the GT first subunit (historical PRISM convention)
  tau_sym    τ after optimal within-orbit relabelling — order-agreement mod symmetry
  tau_ring   τ mod symmetry *and* cyclic rotation/reflection of detected rings
  top1       teacher-forced next-subunit accuracy (seed-independent)
  top1_sym   as top1 but a symmetry-equivalent pick counts as correct
  gt_prob    mean softmax probability assigned to the true next subunit

Outputs → results/ablation_scaling/
  metrics.json  metrics.csv  per_complex.csv  ablation.png  REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.scorer_gnn import AssemblyScorer
from src.evaluation.symmetry import (
    detect_symmetry, tau_exact, tau_mod_symmetry, tau_mod_ring,
    orbit_id_map, size_bin, restrict_to_gt,
)
from src.evaluation.splits import make_size_splits, save_splits
from src.data_prep.bsa_teacher import bsa_greedy_order
from train_gnn import load_labeled_15k_dataset, precompute, DEVICE

OUT_DIR = ROOT / "results" / "ablation_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESM_DIM = 640
NODE_DIM = ESM_DIM + 5 + 1 + 9 + 3      # esm | plddt | sasa_diag | geom | chiral = 658
EDGE_DIM = 6                             # ppi | esm_cos | go_cos | sasa | contact | chiral

NODE_SLICES = {
    "esm":    (0, ESM_DIM),
    "plddt":  (ESM_DIM, ESM_DIM + 5),
    "sasa":   (ESM_DIM + 5, ESM_DIM + 6),
    "geom":   (ESM_DIM + 6, ESM_DIM + 15),
    "chiral": (ESM_DIM + 15, ESM_DIM + 18),
}
EDGE_CHANNELS = {"ppi": 0, "esm_cos": 1, "go_cos": 2, "sasa": 3, "contact": 4, "chiral": 5}


# ──────────────────────────────────────────────────────────────────────────────
# Conditions
# ──────────────────────────────────────────────────────────────────────────────

CONDITIONS: dict[str, dict] = {
    # Current deployed PRISM feature set (v4): everything except chirality.
    "v4_baseline": {
        "node": ["esm", "plddt", "sasa", "geom"],
        "edge": ["ppi", "esm_cos", "go_cos", "sasa", "contact"],
        "note": "current PRISM v4 features — the reference row",
    },
    # Lever 1 interim: v4 + handedness descriptors.
    "v4_plus_chiral": {
        "node": ["esm", "plddt", "sasa", "geom", "chiral"],
        "edge": ["ppi", "esm_cos", "go_cos", "sasa", "contact", "chiral"],
        "note": "Lever 1 interim — adds pseudoscalar chirality features",
    },
    # The headline question: kill every sequence-derived channel.
    "no_esm": {
        "node": ["plddt", "sasa", "geom", "chiral"],
        "edge": ["ppi", "sasa", "contact", "chiral"],
        "note": "structure/interface only — ESM node block + esm_cos + go_cos removed",
    },
    # The mirror image: keep only sequence.
    "esm_only": {
        "node": ["esm"],
        "edge": ["esm_cos"],
        "note": "sequence only — no interface, no geometry",
    },
    # Is any of it needed beyond raw buried area?
    "interface_only": {
        "node": ["sasa"],
        "edge": ["sasa", "contact"],
        "note": "buried-surface-area channels only",
    },
    # Control: same ESM statistics, chain identity destroyed.
    "esm_shuffled": {
        "node": ["esm", "plddt", "sasa", "geom"],
        "edge": ["ppi", "esm_cos", "go_cos", "sasa", "contact"],
        "shuffle_esm": True,
        "note": "control — ESM rows permuted across chains within each complex",
    },
    # Everything.
    "all_features": {
        "node": list(NODE_SLICES),
        "edge": list(EDGE_CHANNELS),
        "note": "all channels including chirality",
    },
}


def build_masks(cond: dict) -> tuple[torch.Tensor, torch.Tensor]:
    nm = torch.zeros(NODE_DIM)
    for blk in cond["node"]:
        a, b = NODE_SLICES[blk]
        nm[a:b] = 1.0
    em = torch.zeros(EDGE_DIM)
    for ch in cond["edge"]:
        em[EDGE_CHANNELS[ch]] = 1.0
    return nm.to(DEVICE), em.to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

class Bundle:
    """Everything needed to train/score one complex, precomputed once and shared."""
    __slots__ = ("cd", "nf", "ef", "adj", "ci", "sym", "n", "gt", "esm_perm", "bsa_seed")

    def __init__(self, cd, nf, ef, adj, ci, sym):
        self.cd, self.nf, self.ef, self.adj, self.ci, self.sym = cd, nf, ef, adj, ci, sym
        self.n = len(cd.chain_ids)
        self.gt = list(cd.gt_order)
        rng = np.random.default_rng(abs(hash(cd.pdb_id)) % (2**31))
        self.esm_perm = torch.tensor(rng.permutation(self.n), dtype=torch.long, device=DEVICE)
        w = np.array(cd.sasa_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(w, 0.0)
        self.bsa_seed = int(np.argmax(w.sum(1))) if w.size and w.max() > 0 else 0

    def view(self, nm, em, shuffle_esm: bool):
        nf = self.nf
        if shuffle_esm:
            nf = torch.cat([nf[self.esm_perm][:, :ESM_DIM], nf[:, ESM_DIM:]], dim=1)
        return nf * nm, self.ef * em


def build_bundles(args) -> list[Bundle]:
    def _valid(c) -> bool:
        n = len(c.chain_ids)
        gt = list(c.gt_order)
        return (len(gt) >= args.min_gt_len
                and n >= 3
                and c.esm_embeddings.shape[1] == ESM_DIM
                and c.sasa_matrix.shape == (n, n)
                and len(set(gt)) == len(gt)          # no repeated subunits
                and all(0 <= v < n for v in gt))     # indices must address real chains

    raw = load_labeled_15k_dataset(max_complexes=args.max_complexes)
    ds = [c for c in raw if _valid(c)]
    print(f"[data] {len(ds)}/{len(raw)} complexes pass validity + gt_order ≥ {args.min_gt_len}")

    bundles, t0 = [], time.time()
    for i, cd in enumerate(ds):
        try:
            nf, ef, adj, ci = precompute(cd, ESM_DIM, include_chiral=True)
        except Exception as e:
            print(f"  [skip] {cd.pdb_id}: {e}")
            continue
        if nf.shape[1] != NODE_DIM or ef.shape[-1] != EDGE_DIM:
            continue
        sym = detect_symmetry(cd.pdb_id, cd.chain_ids, cd.uniprot_ids,
                              cd.sequences, cd.pdb_path)
        bundles.append(Bundle(cd, nf, ef, adj, ci, sym))
        if (i + 1) % 200 == 0:
            print(f"  precomputed {i+1}/{len(ds)}  ({time.time()-t0:.0f}s)")
    print(f"[data] {len(bundles)} bundles ready in {time.time()-t0:.0f}s")

    # Channel health gate. A constant channel carries no information and cannot
    # affect any prediction, so ablating it is guaranteed to show "no effect" —
    # which reads as a modelling result when it is really a data outage. Two
    # such outages were live in this corpus (go_cos identically zero; ppi
    # divided by 1000 a second time on parse), so report it up front rather than
    # leave it to be inferred from a flat ablation row.
    report_channel_health(bundles)
    return bundles


def report_channel_health(bundles: list) -> dict:
    """Log per-edge-channel dynamic range pooled across the loaded corpus."""
    if not bundles:
        return {}
    from src.features.channel_health import channel_stats

    acc = {name: [] for name in EDGE_CHANNELS}
    for b in bundles:
        e = b.ef.detach().cpu().numpy()
        n = e.shape[0]
        if n < 2:
            continue
        off = ~np.eye(n, dtype=bool)
        for name, ch in EDGE_CHANNELS.items():
            if ch < e.shape[-1]:
                acc[name].append(e[:, :, ch][off])

    # channel_stats only ever looks at the off-diagonal; a non-square array is
    # passed through whole, which is what we want for an already-flattened pool.
    pooled = {name: (np.concatenate(v).reshape(-1, 1) if v else None)
              for name, v in acc.items()}
    stats = channel_stats(pooled, reference="sasa")

    print("[channel-health] edge channel dynamic range across the corpus")
    for s in stats.values():
        flag = "" if s.status == "ok" else f"   <-- {s.status.upper()}"
        print(f"    {s.name:<9} mean={s.mean:>8.4f}  std={s.std:>8.4f}  "
              f"vs sasa={s.range_ratio:>6.3f}{flag}")
    bad = [s.name for s in stats.values() if s.status != "ok"]
    if bad:
        print(f"[channel-health] {', '.join(bad)} cannot contribute — ablating "
              f"these measures the data outage, not the model.")
    return {k: v.to_dict() for k, v in stats.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Loss / eval
# ──────────────────────────────────────────────────────────────────────────────

def step_loss(model, b: Bundle, nf, ef, label_smooth: float = 0.1) -> torch.Tensor:
    h = model.embed(nf, b.adj, ef, b.ci)
    order, n = b.gt, b.n
    losses = []
    for step in range(1, len(order)):
        assembled = order[:step]
        correct = order[step]
        remaining = [i for i in range(n) if i not in assembled]
        if correct not in remaining or len(remaining) < 2:
            continue
        mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        mask[assembled] = True
        scores = model.score_candidates(h, ef, mask, remaining)
        k = scores.size(0)
        soft = torch.full((k,), label_smooth / k, device=DEVICE)
        soft[remaining.index(correct)] += 1.0 - label_smooth
        losses.append(-(soft * F.log_softmax(scores, 0)).sum())
    if not losses:
        return torch.zeros((), device=DEVICE)
    return torch.stack(losses).mean()


@torch.no_grad()
def greedy(model, b: Bundle, nf, ef, seed: int) -> list[int]:
    h = model.embed(nf, b.adj, ef, b.ci)
    order = [seed]
    remaining = [i for i in range(b.n) if i != seed]
    while remaining:
        mask = torch.zeros(b.n, dtype=torch.bool, device=DEVICE)
        mask[order] = True
        scores = model.score_candidates(h, ef, mask, remaining)
        pick = remaining[int(scores.argmax().item())]
        order.append(pick)
        remaining.remove(pick)
    return order


@torch.no_grad()
def eval_complex(model, b: Bundle, nf, ef) -> dict:
    """Teacher-forced top1/gt_prob + free-running τ under three symmetry levels."""
    model.eval()
    h = model.embed(nf, b.adj, ef, b.ci)
    order, n = b.gt, b.n
    oid = orbit_id_map(b.sym.orbits)

    hits = hits_sym = steps = 0
    probs = []
    for step in range(1, len(order)):
        assembled = order[:step]
        correct = order[step]
        remaining = [i for i in range(n) if i not in assembled]
        if correct not in remaining or len(remaining) < 2:
            continue
        mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        mask[assembled] = True
        scores = model.score_candidates(h, ef, mask, remaining)
        p = torch.softmax(scores, 0)
        pick = remaining[int(scores.argmax().item())]
        steps += 1
        hits += int(pick == correct)
        hits_sym += int(oid.get(pick, -1) == oid.get(correct, -2))
        probs.append(float(p[remaining.index(correct)]))

    pred_bsa = greedy(model, b, nf, ef, b.bsa_seed)
    pred_gt = greedy(model, b, nf, ef, order[0])

    return {
        "pdb_id": b.cd.pdb_id,
        "n": n,
        "gt_len": len(order),
        "n_orbits": len(b.sym.orbits),
        "point_group": b.sym.point_group,
        "bin": size_bin(n),
        "tau": tau_exact(pred_bsa, order),
        "tau_gtseed": tau_exact(pred_gt, order),
        "tau_sym": tau_mod_symmetry(pred_bsa, order, b.sym.orbits),
        "tau_ring": tau_mod_ring(pred_bsa, order, b.sym),
        "top1": hits / steps if steps else 0.0,
        "top1_sym": hits_sym / steps if steps else 0.0,
        "gt_prob": float(np.mean(probs)) if probs else 0.0,
        # distinct orbits spanned by the annotated order; ==1 ⇒ τ_sym is trivially 1.0
        "n_gt_orbits": len({oid.get(v, -1 - v) for v in order}),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = ["tau", "tau_gtseed", "tau_sym", "tau_ring", "top1", "top1_sym", "gt_prob"]
    if not rows:
        return {k: float("nan") for k in keys} | {"n_complexes": 0}
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["n_complexes"] = len(rows)
    out["mean_n"] = float(np.mean([r["n"] for r in rows]))
    # τ_sym is vacuous when the annotated order lives inside a single orbit
    nd = [r for r in rows if r.get("n_gt_orbits", 2) > 1]
    out["n_sym_nondegenerate"] = len(nd)
    out["tau_sym_nondeg"] = float(np.mean([r["tau_sym"] for r in nd])) if nd else float("nan")
    out["tau_nondeg"] = float(np.mean([r["tau"] for r in nd])) if nd else float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Untrained references
# ──────────────────────────────────────────────────────────────────────────────

def reference_rows(bundles: list[Bundle], kind: str, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for b in bundles:
        w = np.array(b.cd.sasa_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(w, 0.0)
        if kind == "bsa_teacher":
            pred = bsa_greedy_order(w)
            pred_gt = bsa_greedy_order(w, seed=b.gt[0])
        else:
            pred = list(range(b.n)); rng.shuffle(pred)
            pred_gt = [b.gt[0]] + [x for x in pred if x != b.gt[0]]

        oid = orbit_id_map(b.sym.orbits)
        hits = hits_sym = steps = 0
        probs = []
        for step in range(1, len(b.gt)):
            assembled = b.gt[:step]
            correct = b.gt[step]
            remaining = [i for i in range(b.n) if i not in assembled]
            if correct not in remaining or len(remaining) < 2:
                continue
            if kind == "bsa_teacher":
                gains = np.array([w[c, assembled].sum() for c in remaining])
                g = gains / (gains.max() + 1e-9) / 0.15
                p = np.exp(g - g.max()); p /= p.sum()
                pick = remaining[int(np.argmax(gains))]
            else:
                p = np.full(len(remaining), 1.0 / len(remaining))
                pick = rng.choice(remaining)
            steps += 1
            hits += int(pick == correct)
            hits_sym += int(oid.get(pick, -1) == oid.get(correct, -2))
            probs.append(float(p[remaining.index(correct)]))

        rows.append({
            "pdb_id": b.cd.pdb_id, "n": b.n, "gt_len": len(b.gt),
            "n_orbits": len(b.sym.orbits), "point_group": b.sym.point_group,
            "bin": size_bin(b.n),
            "tau": tau_exact(pred, b.gt), "tau_gtseed": tau_exact(pred_gt, b.gt),
            "tau_sym": tau_mod_symmetry(pred, b.gt, b.sym.orbits),
            "tau_ring": tau_mod_ring(pred, b.gt, b.sym),
            "top1": hits / steps if steps else 0.0,
            "top1_sym": hits_sym / steps if steps else 0.0,
            "gt_prob": float(np.mean(probs)) if probs else 0.0,
            "n_gt_orbits": len({oid.get(v, -1 - v) for v in b.gt}),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_condition(name, cond, train_b, val_b, args) -> tuple[AssemblyScorer, list]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    nm, em = build_masks(cond)
    shuffle_esm = bool(cond.get("shuffle_esm"))

    model = AssemblyScorer(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden=args.hidden,
        n_layers=args.layers, heads=4, dropout=0.1, anchor_mode="pooled",
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    order_idx = list(range(len(train_b)))
    best_score, best_state, bad, history = -1e9, None, 0, []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(order_idx)
        epoch_loss, nb = 0.0, 0
        for s in range(0, len(order_idx), args.batch):
            batch = [train_b[i] for i in order_idx[s:s + args.batch]]
            opt.zero_grad()
            losses = []
            for b in batch:
                nf, ef = b.view(nm, em, shuffle_esm)
                losses.append(step_loss(model, b, nf, ef))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss); nb += 1
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rows = [eval_complex(model, b, *b.view(nm, em, shuffle_esm)) for b in val_b]
            agg = aggregate(rows)
            score = agg["top1"] + agg["gt_prob"]
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1), **agg})
            flag = ""
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0; flag = " *"
            else:
                bad += args.eval_every
            print(f"  [{name}] ep{epoch:3d} loss={epoch_loss/max(nb,1):.4f} "
                  f"τ={agg['tau']:.3f} top1={agg['top1']:.3f} gtp={agg['gt_prob']:.3f}{flag}")
            if bad >= args.patience:
                print(f"  [{name}] early stop @ {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{name}] trained in {time.time()-t0:.0f}s")
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────────────

METRIC_KEYS = ["tau", "tau_gtseed", "tau_sym", "tau_sym_nondeg", "tau_ring",
               "top1", "top1_sym", "gt_prob"]


def export(results: dict, per_complex: list[dict], meta: dict, shard: str = ""):
    sfx = f"_{shard}" if shard else ""
    (OUT_DIR / f"metrics{sfx}.json").write_text(json.dumps(
        {"meta": meta, "results": results}, indent=2))

    with open(OUT_DIR / f"metrics{sfx}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "split", "n_complexes", "mean_n"] + METRIC_KEYS)
        for cond, splits in results.items():
            for split, agg in splits["splits"].items():
                w.writerow([cond, split, agg.get("n_complexes", 0),
                            round(agg.get("mean_n", float("nan")), 2)]
                           + [round(agg.get(k, float("nan")), 4) for k in METRIC_KEYS])
        w.writerow([])
        w.writerow(["condition", "size_bin", "n_complexes", "mean_n"] + METRIC_KEYS)
        for cond, splits in results.items():
            for bn, agg in splits["bins"].items():
                w.writerow([cond, bn, agg.get("n_complexes", 0),
                            round(agg.get("mean_n", float("nan")), 2)]
                           + [round(agg.get(k, float("nan")), 4) for k in METRIC_KEYS])

    if per_complex:
        cols = list(per_complex[0].keys())
        with open(OUT_DIR / f"per_complex{sfx}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(per_complex)


def plot(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    conds = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor="#0d1117")
    fg = "#e6edf3"

    # 1) small vs large τ
    ax = axes[0]
    x = np.arange(len(conds)); w = 0.38
    for off, split, c in ((-w/2, "test_small", "#58a6ff"), (w/2, "test_large", "#f85149")):
        vals = [results[k]["splits"].get(split, {}).get("top1", np.nan) for k in conds]
        ax.bar(x + off, vals, w, label=split, color=c)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=35, ha="right", color=fg, fontsize=9)
    ax.set_title("Top-1 next-subunit accuracy", color=fg)
    ax.legend(facecolor="#161b22", labelcolor=fg)

    # 2) symmetry-aware τ ladder on large split
    ax = axes[1]
    for i, key, c in ((0, "tau", "#f85149"), (1, "tau_sym", "#d29922"), (2, "tau_ring", "#3fb950")):
        vals = [results[k]["splits"].get("test_large", {}).get(key, np.nan) for k in conds]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=key, color=c)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=35, ha="right", color=fg, fontsize=9)
    ax.set_title("τ on held-out large complexes (n≥21), by symmetry level", color=fg)
    ax.legend(facecolor="#161b22", labelcolor=fg)

    # 3) degradation with size
    ax = axes[2]
    bins = ["2-4", "5-10", "11-20", "21-40", "41+"]
    for k in conds:
        y = [results[k]["bins"].get(b, {}).get("top1", np.nan) for b in bins]
        ax.plot(bins, y, marker="o", label=k)
    ax.set_title("Top-1 vs complex size", color=fg)
    ax.set_xlabel("subunits", color=fg)
    ax.legend(fontsize=8, facecolor="#161b22", labelcolor=fg)

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors=fg)
        for s in ax.spines.values():
            s.set_color("#21262d")
        ax.grid(alpha=0.2, color="#21262d")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation.png", dpi=140, facecolor="#0d1117")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_gt_len", type=int, default=4)
    ap.add_argument("--max_complexes", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_train_eval", type=int, default=150)
    ap.add_argument("--large_threshold", type=int, default=21)
    ap.add_argument("--conditions", type=str, default="")
    ap.add_argument("--shard", type=str, default="", help="suffix for output files; lets shards run in parallel")
    ap.add_argument("--skip_refs", action="store_true")
    args = ap.parse_args()

    bundles = build_bundles(args)
    by_id = {b.cd.pdb_id: b for b in bundles}

    splits = make_size_splits([b.cd for b in bundles], large_threshold=args.large_threshold,
                              min_gt_len=args.min_gt_len)
    save_splits(splits)
    train_b = [by_id[p] for p in splits["train"] if p in by_id]
    small_b = [by_id[p] for p in splits["test_small"] if p in by_id]
    large_b = [by_id[p] for p in splits["test_large"] if p in by_id]
    if args.max_train:
        train_b = train_b[:args.max_train]
    print(f"[split] train={len(train_b)} test_small={len(small_b)} test_large={len(large_b)}")
    print(f"[split] large-split sizes: {sorted(b.n for b in large_b)}")

    selected = [c.strip() for c in args.conditions.split(",") if c.strip()] or list(CONDITIONS)

    results: dict = {}
    per_complex: list[dict] = []

    # Untrained references first — they set the floor and the heuristic bar.
    for ref in (() if args.skip_refs else ("random", "bsa_teacher")):
        splits_res, bins = {}, defaultdict(list)
        for split_name, bs in (("test_small", small_b), ("test_large", large_b)):
            rows = reference_rows(bs, ref, seed=args.seed)
            splits_res[split_name] = aggregate(rows)
            for r in rows:
                bins[r["bin"]].append(r)
                per_complex.append({"condition": ref, "split": split_name, **r})
        results[ref] = {
            "note": "untrained reference",
            "splits": splits_res,
            "bins": {b: aggregate(v) for b, v in sorted(bins.items())},
            "history": [],
        }
        print(f"[ref] {ref}: small top1={splits_res['test_small']['top1']:.3f} "
              f"large top1={splits_res['test_large']['top1']:.3f}")

    for name in selected:
        cond = CONDITIONS[name]
        print(f"\n=== {name} — {cond['note']} ===")
        model, history = train_condition(name, cond, train_b, small_b, args)
        nm, em = build_masks(cond)
        sh = bool(cond.get("shuffle_esm"))

        splits_res, bins = {}, defaultdict(list)
        for split_name, bs in (("train", train_b[:args.max_train_eval]),
                               ("test_small", small_b), ("test_large", large_b)):
            rows = [eval_complex(model, b, *b.view(nm, em, sh)) for b in bs]
            splits_res[split_name] = aggregate(rows)
            if split_name != "train":
                for r in rows:
                    bins[r["bin"]].append(r)
                    per_complex.append({"condition": name, "split": split_name, **r})
        results[name] = {
            "note": cond["note"],
            "splits": splits_res,
            "bins": {b: aggregate(v) for b, v in sorted(bins.items())},
            "history": history,
        }
        s, l = splits_res["test_small"], splits_res["test_large"]
        print(f"  → small: τ={s['tau']:.3f} top1={s['top1']:.3f} gtp={s['gt_prob']:.3f}"
              f"  | large: τ={l['tau']:.3f} τ_sym={l['tau_sym']:.3f} top1={l['top1']:.3f}")

    meta = {
        "device": DEVICE,
        "node_dim": NODE_DIM, "edge_dim": EDGE_DIM,
        "n_train": len(train_b), "n_test_small": len(small_b), "n_test_large": len(large_b),
        "args": vars(args),
        "large_split_sym_degenerate": int(sum(
            1 for r in per_complex
            if r["split"] == "test_large" and r["condition"] == "bsa_teacher"
            and r.get("n_gt_orbits", 2) <= 1)),
        "shard": args.shard,
    }
    export(results, per_complex, meta, args.shard)
    if not args.shard:
        plot(results)

    print(f"\n=== SUMMARY (test_large, n≥{args.large_threshold}) ===")
    print(f"{'condition':<18}" + "".join(f"{k:>16}" for k in METRIC_KEYS))
    for k, v in results.items():
        a = v["splits"]["test_large"]
        print(f"{k:<18}" + "".join(f"{a.get(m, float('nan')):>16.4f}" for m in METRIC_KEYS))
    print(f"\nexported → {OUT_DIR}")


if __name__ == "__main__":
    main()
