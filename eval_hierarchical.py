"""
eval_hierarchical.py — does Lever 2 (hierarchical assembly) actually pay off?

Trains one AssemblyScorer on the small-complex split (n ≤ 20) exactly as
run_ablation.py does, then evaluates it on the frozen large split (n ≥ 21) two
ways with the *same weights*:

  flat          — one greedy pass over all n subunits
  hierarchical  — module order on the coarsened interface graph, then
                  within-module order on each induced subgraph

The scorer is shape-agnostic, so this isolates the inference-time effect of
capping the effective n: no retraining, no extra parameters.

Also reports the effective n actually seen at each level, and the wall-clock
cost, since Lever 2 is claimed to be both an accuracy *and* an efficiency win.

Output → results/ablation_scaling/hierarchical.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from run_ablation import (
    CONDITIONS, build_masks, build_bundles, train_condition, greedy,
    aggregate, OUT_DIR,
)
from src.evaluation.symmetry import tau_exact, tau_mod_symmetry, tau_mod_ring, size_bin
from src.evaluation.splits import make_size_splits
from src.models.hierarchical import detect_modules, hierarchical_order


def score(pred, b) -> dict:
    return {
        "tau": tau_exact(pred, b.gt),
        "tau_sym": tau_mod_symmetry(pred, b.gt, b.sym.orbits),
        "tau_ring": tau_mod_ring(pred, b.gt, b.sym),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="v4_plus_chiral")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_gt_len", type=int, default=4)
    ap.add_argument("--max_complexes", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--large_threshold", type=int, default=21)
    ap.add_argument("--target_size", type=int, default=8)
    args = ap.parse_args()

    bundles = build_bundles(args)
    by_id = {b.cd.pdb_id: b for b in bundles}
    splits = make_size_splits([b.cd for b in bundles], args.large_threshold, 0.2, args.min_gt_len)
    train_b = [by_id[p] for p in splits["train"] if p in by_id]
    small_b = [by_id[p] for p in splits["test_small"] if p in by_id]
    large_b = [by_id[p] for p in splits["test_large"] if p in by_id]
    print(f"[split] train={len(train_b)} small={len(small_b)} large={len(large_b)}")

    cond = CONDITIONS[args.condition]
    model, _ = train_condition(args.condition, cond, train_b, small_b, args)
    nm, em = build_masks(cond)
    sh = bool(cond.get("shuffle_esm"))

    flat_rows, hier_rows, info_rows = [], [], []
    t_flat = t_hier = 0.0

    for b in large_b:
        nf, ef = b.view(nm, em, sh)

        t0 = time.time()
        pred_flat = greedy(model, b, nf, ef, b.bsa_seed)
        t_flat += time.time() - t0

        mods = detect_modules(b.cd.sasa_matrix, target_size=args.target_size)
        t0 = time.time()
        pred_hier, info = hierarchical_order(
            model, nf, ef, b.cd.sasa_matrix, b.ci,
            target_size=args.target_size, modules=mods,
        )
        t_hier += time.time() - t0

        flat_rows.append({"pdb_id": b.cd.pdb_id, "n": b.n, "bin": size_bin(b.n),
                          "n_gt_orbits": len({i for i in range(len(b.sym.orbits))
                                              if any(v in b.sym.orbits[i] for v in b.gt)}),
                          **score(pred_flat, b)})
        hier_rows.append({"pdb_id": b.cd.pdb_id, "n": b.n, "bin": size_bin(b.n),
                          "n_gt_orbits": flat_rows[-1]["n_gt_orbits"],
                          **score(pred_hier, b)})
        info_rows.append({"pdb_id": b.cd.pdb_id, "n": b.n, **info})

    def agg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in ("tau", "tau_sym", "tau_ring")} | {
            "n_complexes": len(rows)}

    out = {
        "condition": args.condition,
        "n_large": len(large_b),
        "flat": agg(flat_rows),
        "hierarchical": agg(hier_rows),
        "mean_n": float(np.mean([r["n"] for r in info_rows])),
        "mean_effective_n": float(np.mean([r["effective_n"] for r in info_rows])),
        "mean_modules": float(np.mean([r["n_modules"] for r in info_rows])),
        "flat_seconds": t_flat,
        "hierarchical_seconds": t_hier,
        "per_complex": {"flat": flat_rows, "hierarchical": hier_rows, "modules": info_rows},
    }
    (OUT_DIR / "hierarchical.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== LEVER 2 — flat vs hierarchical on {len(large_b)} held-out large complexes ===")
    print(f"{'mode':<16}{'tau':>10}{'tau_sym':>10}{'tau_ring':>10}{'seconds':>10}")
    print(f"{'flat':<16}{out['flat']['tau']:>10.4f}{out['flat']['tau_sym']:>10.4f}"
          f"{out['flat']['tau_ring']:>10.4f}{t_flat:>10.2f}")
    print(f"{'hierarchical':<16}{out['hierarchical']['tau']:>10.4f}{out['hierarchical']['tau_sym']:>10.4f}"
          f"{out['hierarchical']['tau_ring']:>10.4f}{t_hier:>10.2f}")
    print(f"mean n={out['mean_n']:.1f} → effective n={out['mean_effective_n']:.1f} "
          f"across {out['mean_modules']:.1f} modules")
    print(f"wrote → {OUT_DIR / 'hierarchical.json'}")


if __name__ == "__main__":
    main()
