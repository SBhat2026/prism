"""
audit_channels.py — is the SASA/ESM/GO/PPI readout a display bug, or is the
model genuinely ignoring everything except SASA?

Answers both halves separately, because they have different causes:

  A. DATA HEALTH   — how much information does each channel actually carry in
                     the corpus? A channel that is constant cannot be used by
                     any model, no matter how it is architected.

  B. MODEL RELIANCE — occlusion attribution on a trained checkpoint. Each edge
                     channel / node block is neutralised in turn (replaced by
                     its own per-complex mean, so scale and shape are preserved
                     but variation is removed) and we measure how far the
                     model's candidate distribution moves. A channel the model
                     truly uses moves it; a channel it ignores does not.

Occlusion is teacher-forced along the annotated order, so the numbers are
seed-independent and directly comparable to `top1` in run_ablation.py.

Output → results/diagnostics/channel_audit.json  (+ a printed summary)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.scorer_gnn import AssemblyScorer
from train_gnn import load_labeled_15k_dataset, precompute, DEVICE

OUT_DIR = ROOT / "results" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ESM_DIM = 640

# v4 production layout (include_chiral=False): node 655, edge 5
NODE_SLICES_V4 = {
    "esm":   (0, ESM_DIM),
    "plddt": (ESM_DIM, ESM_DIM + 5),
    "sasa":  (ESM_DIM + 5, ESM_DIM + 6),
    "geom":  (ESM_DIM + 6, ESM_DIM + 15),
}
EDGE_CHANNELS_V4 = {"ppi": 0, "esm_cos": 1, "go_cos": 2, "sasa": 3, "contact": 4}


# ──────────────────────────────────────────────────────────────────────────────
# A. Data health
# ──────────────────────────────────────────────────────────────────────────────

def channel_health(complexes, edge_channels: dict[str, int]) -> dict:
    """Per-channel dynamic range across the corpus.

    `dead_frac`     fraction of complexes where the channel is constant off-diagonal
                    (std < 1e-9) — the model sees no contrast at all.
    `zero_frac`     fraction where it is identically zero.
    `mean_std`      mean within-complex off-diagonal std — the usable dynamic range.
    `range_ratio`   mean_std relative to the SASA channel, i.e. how loud this
                    channel is compared with the one channel known to work.
    """
    acc = {k: {"std": [], "mean": [], "dead": 0, "zero": 0, "const_val": set()}
           for k in edge_channels}
    n = 0
    for cd in complexes:
        N = len(cd.chain_ids)
        if N < 3:
            continue
        # Read the source matrices directly rather than going through
        # precompute() — the contact/geom passes are slow and irrelevant here.
        sasa = cd.sasa_matrix / (cd.sasa_matrix.max() + 1e-8)
        # contact_density is not carried on ComplexData — precompute() derives it
        # from the full-complex PDB and caches it. Read that cache directly.
        contact = None
        cpath = ROOT / "data" / "feature_cache" / f"{cd.pdb_id}_contact.npy"
        if cpath.exists():
            try:
                c = np.load(str(cpath))
                contact = c if np.shape(c) == (N, N) else None
            except Exception:
                contact = None
        raw = {"ppi": cd.ppi_matrix, "esm_cos": cd.esm_matrix,
               "go_cos": cd.go_matrix, "sasa": sasa, "contact": contact}
        if any(raw[k] is None for k in ("ppi", "esm_cos", "go_cos")):
            continue
        n += 1
        off = ~np.eye(N, dtype=bool)
        for name in edge_channels:
            m = raw.get(name)
            if m is None or np.shape(m) != (N, N):
                acc[name]["std"].append(0.0)
                acc[name]["mean"].append(0.0)
                acc[name]["dead"] += 1
                acc[name]["zero"] += 1
                continue
            v = np.asarray(m, dtype=np.float64)[off]
            s, m = float(v.std()), float(v.mean())
            acc[name]["std"].append(s)
            acc[name]["mean"].append(m)
            if s < 1e-9:
                acc[name]["dead"] += 1
                acc[name]["const_val"].add(round(m, 6))
            if s < 1e-9 and abs(m) < 1e-9:
                acc[name]["zero"] += 1

    sasa_std = float(np.mean(acc["sasa"]["std"])) if n else 1.0
    out = {"n_complexes": n, "channels": {}}
    for name in edge_channels:
        a = acc[name]
        ms = float(np.mean(a["std"])) if a["std"] else 0.0
        cv = sorted(a["const_val"])[:5]
        out["channels"][name] = {
            "mean_value": float(np.mean(a["mean"])) if a["mean"] else 0.0,
            "mean_std": ms,
            "dead_frac": a["dead"] / n if n else 0.0,
            "zero_frac": a["zero"] / n if n else 0.0,
            "range_ratio_vs_sasa": ms / (sasa_std + 1e-12),
            "constant_values_seen": cv,
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# B. Model reliance (occlusion)
# ──────────────────────────────────────────────────────────────────────────────

def _teacher_forced(model, nf, ef, adj, ci, gt, N):
    """Return (top1_hits, n_steps, list of candidate prob vectors)."""
    with torch.no_grad():
        h = model.embed(nf, adj, ef, ci)
    assembled: list[int] = []
    hits, steps, dists = 0, 0, []
    remaining = list(range(N))
    for t, true_next in enumerate(gt):
        if true_next not in remaining or len(remaining) < 2:
            break
        am = torch.zeros(N, dtype=torch.bool, device=nf.device)
        for i in assembled:
            am[i] = True
        with torch.no_grad():
            sc = model.score_candidates(h, ef, am, remaining)
            p = F.softmax(sc, dim=0)
        if remaining[int(torch.argmax(sc))] == true_next:
            hits += 1
        steps += 1
        dists.append((list(remaining), p.detach().cpu().numpy()))
        assembled.append(true_next)
        remaining.remove(true_next)
    return hits, steps, dists


def _occlude_edge(ef: torch.Tensor, ci: int) -> torch.Tensor:
    """Replace one edge channel by its off-diagonal mean (kills variation, keeps scale)."""
    out = ef.clone()
    N = ef.shape[0]
    off = ~torch.eye(N, dtype=torch.bool, device=ef.device)
    out[:, :, ci] = torch.where(off, ef[:, :, ci][off].mean(), ef[:, :, ci])
    return out


def _occlude_node(nf: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Replace a node feature block by its column-wise mean across chains."""
    out = nf.clone()
    out[:, lo:hi] = nf[:, lo:hi].mean(dim=0, keepdim=True)
    return out


def occlusion(model, bundles, node_slices, edge_channels) -> dict:
    base_hits = base_steps = 0
    per = {k: {"hits": 0, "jsd": []} for k in
           list(edge_channels) + [f"node:{k}" for k in node_slices]
           + ["ALL:edges", "ALL:random_edges"]}

    for nf, ef, adj, ci, gt, N in bundles:
        h0, s0, d0 = _teacher_forced(model, nf, ef, adj, ci, gt, N)
        base_hits += h0
        base_steps += s0
        if s0 == 0:
            continue

        def _run(nf2, ef2, key):
            h1, s1, d1 = _teacher_forced(model, nf2, ef2, adj, ci, gt, N)
            per[key]["hits"] += h1
            # Jensen-Shannon divergence between the two candidate distributions
            for (r0, p0), (r1, p1) in zip(d0, d1):
                if len(p0) != len(p1):
                    continue
                m = 0.5 * (p0 + p1)
                kl = lambda a, b: float(np.sum(a * np.log((a + 1e-12) / (b + 1e-12))))
                per[key]["jsd"].append(0.5 * kl(p0, m) + 0.5 * kl(p1, m))

        for name, cidx in edge_channels.items():
            _run(nf, _occlude_edge(ef, cidx), name)
        for name, (lo, hi) in node_slices.items():
            _run(_occlude_node(nf, lo, hi), ef, f"node:{name}")

        # ── Controls ──────────────────────────────────────────────────────────
        # ALL:edges neutralises every edge channel at once. If the individual
        # per-channel deltas are ~0 but this one is also ~0, the model is
        # edge-blind. If this one moves while the individuals do not, the
        # channels are redundant rather than unused. And if this is exactly 0
        # too, that is the signature to check for a no-op in the occlusion
        # itself — which ALL:random guards against: it replaces the edge tensor
        # with noise and MUST move the output of any edge-sensitive model.
        ef_flat = ef.clone()
        for cidx in edge_channels.values():
            ef_flat = _occlude_edge(ef_flat, cidx)
        _run(nf, ef_flat, "ALL:edges")

        g = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.rand(ef.shape, generator=g).to(ef.device)
        _run(nf, noise, "ALL:random_edges")

    base = base_hits / max(base_steps, 1)
    out = {"baseline_top1": base, "n_steps": base_steps, "channels": {}}
    for k, v in per.items():
        occ = v["hits"] / max(base_steps, 1)
        out["channels"][k] = {
            "top1_when_occluded": occ,
            "delta_top1": base - occ,
            "mean_jsd": float(np.mean(v["jsd"])) if v["jsd"] else 0.0,
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Trained-model occlusion (the honest version)
#
# The archived checkpoints are undertrained — their W_attn edge weights are still
# at initialisation magnitude, so "the model ignores this channel" would be
# unfalsifiable. This trains a fresh model with the current 658/6 layout on the
# same split run_ablation.py uses, then occludes all six edge channels.
# ──────────────────────────────────────────────────────────────────────────────

def _train_mode(args):
    from run_ablation import (
        CONDITIONS, build_bundles, train_condition, build_masks,
        NODE_SLICES, EDGE_CHANNELS,
    )
    from src.evaluation.splits import make_size_splits

    bundles = build_bundles(args)
    by_id = {b.cd.pdb_id: b for b in bundles}
    splits = make_size_splits([b.cd for b in bundles], args.large_threshold,
                              0.2, args.min_gt_len)
    train_b = [by_id[p] for p in splits["train"] if p in by_id]
    small_b = [by_id[p] for p in splits["test_small"] if p in by_id]
    print(f"[split] train={len(train_b)} small={len(small_b)}")

    cond = CONDITIONS["all_features"]
    model, _ = train_condition("all_features", cond, train_b, small_b, args)
    nm, em = build_masks(cond)

    eval_b = small_b[: args.n_occlude]
    packed = [(b.nf * nm, b.ef * em, b.adj, b.ci, b.gt, b.n) for b in eval_b]
    print(f"\n[model] occluding over {len(packed)} held-out complexes")

    occ = occlusion(model, packed, NODE_SLICES, EDGE_CHANNELS)
    print(f"\n=== MODEL RELIANCE — freshly trained all_features model ===")
    print(f"baseline top-1 = {occ['baseline_top1']:.4f} over {occ['n_steps']} steps\n")
    print(f"{'occluded':<16}{'top-1':>10}{'Δtop-1':>10}{'mean JSD':>12}")
    for k, v in sorted(occ["channels"].items(), key=lambda kv: -kv[1]["mean_jsd"]):
        print(f"{k:<16}{v['top1_when_occluded']:>10.4f}"
              f"{v['delta_top1']:>+10.4f}{v['mean_jsd']:>12.6f}")

    p = OUT_DIR / "channel_audit_trained.json"
    p.write_text(json.dumps({"occlusion_trained": occ}, indent=2))
    print(f"\nwrote → {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/gnn_training/best_v4.pt")
    ap.add_argument("--max_complexes", type=int, default=1500)
    ap.add_argument("--n_occlude", type=int, default=120)
    ap.add_argument("--min_gt_len", type=int, default=4)
    ap.add_argument("--train", action="store_true",
                    help="train a fresh 658/6 model on the real split instead of "
                         "loading --ckpt; the stored checkpoints are undertrained "
                         "and their edge weights sit at initialisation.")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--large_threshold", type=int, default=21)
    ap.add_argument("--max_train_eval", type=int, default=150)
    args = ap.parse_args()

    if args.train:
        return _train_mode(args)

    raw = load_labeled_15k_dataset(max_complexes=args.max_complexes)

    def _valid(c):
        n = len(c.chain_ids)
        gt = list(c.gt_order)
        return (len(gt) >= args.min_gt_len and n >= 3
                and c.esm_embeddings.shape[1] == ESM_DIM
                and c.sasa_matrix.shape == (n, n)
                and len(set(gt)) == len(gt) and all(0 <= v < n for v in gt))

    ds = [c for c in raw if _valid(c)]
    print(f"[data] {len(ds)}/{len(raw)} usable")

    print("\n=== A. DATA HEALTH ===")
    health = channel_health(ds, EDGE_CHANNELS_V4)
    print(f"{'channel':<10}{'mean':>10}{'std':>10}{'vs sasa':>10}"
          f"{'dead':>9}{'all-zero':>10}")
    for k, v in health["channels"].items():
        print(f"{k:<10}{v['mean_value']:>10.4f}{v['mean_std']:>10.4f}"
              f"{v['range_ratio_vs_sasa']:>10.3f}{v['dead_frac']:>9.1%}{v['zero_frac']:>10.1%}")

    result = {"data_health": health}

    ckpt_path = ROOT / args.ckpt
    if ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        node_dim = ck.get("node_dim", 655)
        model = AssemblyScorer(node_dim=node_dim, edge_dim=5, hidden=128,
                               n_layers=3, heads=4, dropout=0.0)
        model.load_state_dict(ck["model_state_dict"])
        model.eval().to(DEVICE)
        print(f"\n[model] {args.ckpt} node_dim={node_dim} epoch={ck.get('epoch')}")

        bundles = []
        for cd in ds:
            if len(bundles) >= args.n_occlude:
                break
            try:
                nf, ef, adj, ci = precompute(cd, ESM_DIM, include_chiral=False)
            except Exception:
                continue
            if nf.shape[1] != node_dim or ef.shape[-1] != 5:
                continue
            bundles.append((nf.to(DEVICE), ef.to(DEVICE), adj.to(DEVICE),
                            ci.to(DEVICE) if ci is not None else None,
                            list(cd.gt_order), len(cd.chain_ids)))
        print(f"[model] occluding over {len(bundles)} complexes")

        print("\n=== B. MODEL RELIANCE (occlusion) ===")
        occ = occlusion(model, bundles, NODE_SLICES_V4, EDGE_CHANNELS_V4)
        print(f"baseline top-1 = {occ['baseline_top1']:.4f} over {occ['n_steps']} steps\n")
        print(f"{'occluded':<16}{'top-1':>10}{'Δtop-1':>10}{'mean JSD':>12}")
        for k, v in sorted(occ["channels"].items(),
                           key=lambda kv: -kv[1]["mean_jsd"]):
            print(f"{k:<16}{v['top1_when_occluded']:>10.4f}"
                  f"{v['delta_top1']:>+10.4f}{v['mean_jsd']:>12.6f}")
        result["occlusion"] = occ
    else:
        print(f"[model] checkpoint {ckpt_path} missing — skipping occlusion")

    (OUT_DIR / "channel_audit.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote → {OUT_DIR / 'channel_audit.json'}")


if __name__ == "__main__":
    main()
