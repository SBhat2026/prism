"""
diagnose_scaling.py — Step 1 of PRISM_scaling_roadmap.md §4: "Diagnose first."

Four questions, answered from data rather than argument. None of this trains
anything, so it runs in seconds and can be re-run after every change.

  D1  Symmetry census
      How symmetric is the corpus, how far does asymmetric-unit reduction
      actually collapse n, and how much of the ≥21-chain split has an annotated
      order that lives inside a single orbit (where exact-order τ is meaningless)?

  D2  Chirality blind spot — direct proof
      Mirror every complex's coordinates and re-derive features. If the current
      v4 channels are bit-identical before and after, the network provably
      cannot distinguish an assembly from its mirror image, and no amount of
      data fixes it. The new pseudoscalar descriptors must flip sign.

  D3  BSA teacher quality vs n  (Lever 4 feasibility)
      The Marsh/Levy rule can label any structure. How good is that label, and
      does it decay with size? This bounds what distillation can buy.

  D4  Hierarchical reduction  (Lever 2 feasibility)
      Community detection on the interface graph: how small does the effective
      n get, and is the modular decomposition consistent with the annotated
      assembly order (do modules assemble contiguously)?

Output → results/diagnostics/diagnostics.json + a printed summary.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.evaluation.symmetry import (
    detect_symmetry, load_ca_coords, chain_centroids, size_bin, orbit_id_map,
)
from src.features.chiral_descriptors import compute_chiral_features, _oriented_axis
from src.data_prep.bsa_teacher import bsa_greedy_order, agreement_with_gt, bsa_order_confidence
from src.models.hierarchical import detect_modules
from train_gnn import load_labeled_15k_dataset

OUT = ROOT / "results" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def _valid(c) -> bool:
    n = len(c.chain_ids)
    gt = list(c.gt_order)
    return (len(gt) >= 4 and n >= 3 and c.sasa_matrix.shape == (n, n)
            and len(set(gt)) == len(gt) and all(0 <= v < n for v in gt))


# ──────────────────────────────────────────────────────────────────────────────
# D2 helpers — invariant vs pseudoscalar features under reflection
# ──────────────────────────────────────────────────────────────────────────────

def mirror_invariance_probe(pdb_path: str, chain_ids: list[str]) -> dict | None:
    """
    Recompute the *distance-based* channels (what v4 actually uses) and the new
    chiral channels on the original and mirrored coordinates.
    """
    coords = load_ca_coords(pdb_path, chain_ids)
    if len(coords) < len(chain_ids):
        return None
    mirror = {c: v * np.array([1.0, 1.0, -1.0]) for c, v in coords.items()}

    def dist_matrix(cs):
        cents = np.stack([cs[c].mean(axis=0) for c in chain_ids])
        return np.linalg.norm(cents[:, None] - cents[None, :], axis=-1)

    def chiral_edge(cs):
        cents = np.stack([cs[c].mean(axis=0) for c in chain_ids])
        axes = np.stack([_oriented_axis(cs[c]) for c in chain_ids])
        n = len(chain_ids)
        e = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                u = cents[j] - cents[i]
                u = u / (np.linalg.norm(u) + 1e-12)
                e[i, j] = float(np.dot(u, np.cross(axes[i], axes[j])))
                e[j, i] = -e[i, j]
        return e

    d0, d1 = dist_matrix(coords), dist_matrix(mirror)
    c0, c1 = chiral_edge(coords), chiral_edge(mirror)
    return {
        "invariant_max_abs_diff": float(np.abs(d0 - d1).max()),
        "chiral_mean_abs": float(np.abs(c0).mean()),
        "chiral_max_abs_sum": float(np.abs(c0 + c1).max()),   # ≈0 ⇒ exact sign flip
    }


# ──────────────────────────────────────────────────────────────────────────────

def main():
    raw = load_labeled_15k_dataset()
    ds = [c for c in raw if _valid(c)]
    print(f"[diag] {len(ds)}/{len(raw)} usable complexes (gt_order ≥ 4, indices valid)\n")

    d1_rows, d3_rows, d4_rows = [], [], []
    d2_rows = []
    pg_counter, bin_counter = Counter(), Counter()

    for cd in ds:
        n = len(cd.chain_ids)
        b = size_bin(n)
        bin_counter[b] += 1
        sym = detect_symmetry(cd.pdb_id, cd.chain_ids, cd.uniprot_ids,
                              cd.sequences, cd.pdb_path)
        pg_counter[sym.point_group] += 1
        oid = orbit_id_map(sym.orbits)
        gt_orbits = len({oid.get(v, -1 - v) for v in cd.gt_order})

        d1_rows.append({
            "pdb_id": cd.pdb_id, "n": n, "bin": b,
            "point_group": sym.point_group,
            "n_orbits": len(sym.orbits),
            "reduction": len(sym.orbits) / n,
            "has_ring": bool(sym.ring_orbits),
            "gt_orbits": gt_orbits,
            "gt_len": len(cd.gt_order),
            "sym_degenerate": int(gt_orbits <= 1),
        })

        # D3 — teacher agreement
        agr = agreement_with_gt(cd.sasa_matrix, cd.gt_order)
        conf = bsa_order_confidence(cd.sasa_matrix, cd.gt_order)
        d3_rows.append({"pdb_id": cd.pdb_id, "n": n, "bin": b,
                        "tau": agr["tau"], "top1": agr["top1"],
                        "mean_margin": float(np.mean(conf)) if conf else 0.0,
                        "tied_step_frac": float(np.mean([c < 0.02 for c in conf])) if conf else 0.0})

        # D4 — hierarchical reduction
        mods = detect_modules(cd.sasa_matrix, target_size=8)
        eff = max(len(mods), max(len(m) for m in mods))
        mod_of = {i: k for k, m in enumerate(mods) for i in m}
        seq = [mod_of[v] for v in cd.gt_order]
        # contiguity: fraction of consecutive GT steps that stay inside a module
        contig = float(np.mean([seq[i] == seq[i - 1] for i in range(1, len(seq))])) if len(seq) > 1 else 0.0
        # blocks: how many times the GT order switches module
        switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        d4_rows.append({"pdb_id": cd.pdb_id, "n": n, "bin": b,
                        "n_modules": len(mods), "effective_n": eff,
                        "reduction": eff / n, "within_module_frac": contig,
                        "module_switches": switches,
                        "min_switches": len(set(seq)) - 1})

    # D2 on a size-stratified sample (the probe is O(N²) with coords)
    sample = []
    for b in ["2-4", "5-10", "11-20", "21-40", "41+"]:
        cands = [c for c in ds if size_bin(len(c.chain_ids)) == b and c.pdb_path]
        sample += cands[:12]
    for cd in sample:
        r = mirror_invariance_probe(cd.pdb_path, cd.chain_ids)
        if r:
            d2_rows.append({"pdb_id": cd.pdb_id, "n": len(cd.chain_ids),
                            "bin": size_bin(len(cd.chain_ids)), **r})

    def by_bin(rows, key):
        out = {}
        for b in ["2-4", "5-10", "11-20", "21-40", "41+"]:
            vals = [r[key] for r in rows if r["bin"] == b]
            out[b] = {"n": len(vals), "mean": float(np.mean(vals)) if vals else float("nan")}
        return out

    large = [r for r in d1_rows if r["n"] >= 21]
    report = {
        "n_complexes": len(ds),
        "size_distribution": dict(bin_counter),
        "D1_symmetry": {
            "point_groups": dict(pg_counter.most_common()),
            "frac_with_detected_rotational_symmetry":
                float(np.mean([r["point_group"] != "C1" for r in d1_rows])),
            "asu_reduction_by_bin": by_bin(d1_rows, "reduction"),
            "large_split": {
                "n": len(large),
                "frac_symmetric": float(np.mean([r["point_group"] != "C1" for r in large])) if large else float("nan"),
                "frac_sym_degenerate_gt": float(np.mean([r["sym_degenerate"] for r in large])) if large else float("nan"),
                "mean_n": float(np.mean([r["n"] for r in large])) if large else float("nan"),
                "mean_effective_n_after_asu": float(np.mean([r["n_orbits"] for r in large])) if large else float("nan"),
            },
        },
        "D2_chirality": {
            "n_probed": len(d2_rows),
            "invariant_channels_max_abs_diff_under_mirror":
                float(np.max([r["invariant_max_abs_diff"] for r in d2_rows])) if d2_rows else float("nan"),
            "chiral_channel_mean_abs_signal":
                float(np.mean([r["chiral_mean_abs"] for r in d2_rows])) if d2_rows else float("nan"),
            "chiral_channel_max_abs_sum_under_mirror":
                float(np.max([r["chiral_max_abs_sum"] for r in d2_rows])) if d2_rows else float("nan"),
            "verdict": None,
        },
        "D3_bsa_teacher": {
            "overall_tau": float(np.nanmean([r["tau"] for r in d3_rows])),
            "overall_top1": float(np.nanmean([r["top1"] for r in d3_rows])),
            "tau_by_bin": by_bin(d3_rows, "tau"),
            "top1_by_bin": by_bin(d3_rows, "top1"),
            "tied_step_frac_by_bin": by_bin(d3_rows, "tied_step_frac"),
        },
        "D4_hierarchical": {
            "mean_effective_n_by_bin": by_bin(d4_rows, "effective_n"),
            "mean_reduction_by_bin": by_bin(d4_rows, "reduction"),
            "within_module_contiguity_by_bin": by_bin(d4_rows, "within_module_frac"),
            "large_split": {
                "mean_n": float(np.mean([r["n"] for r in d4_rows if r["n"] >= 21])),
                "mean_effective_n": float(np.mean([r["effective_n"] for r in d4_rows if r["n"] >= 21])),
                "mean_modules": float(np.mean([r["n_modules"] for r in d4_rows if r["n"] >= 21])),
            },
        },
    }

    inv = report["D2_chirality"]["invariant_channels_max_abs_diff_under_mirror"]
    flip = report["D2_chirality"]["chiral_channel_max_abs_sum_under_mirror"]
    report["D2_chirality"]["verdict"] = (
        f"invariant channels differ by at most {inv:.2e} Å under reflection "
        f"(= identical ⇒ architectural blind spot confirmed); chiral channels "
        f"sum to at most {flip:.2e} with their mirror (= exact sign flip ⇒ "
        f"handedness is now representable)."
    )

    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2))
    for name, rows in (("d1_symmetry", d1_rows), ("d2_chirality", d2_rows),
                       ("d3_teacher", d3_rows), ("d4_modules", d4_rows)):
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2))

    # ── printed summary ───────────────────────────────────────────────────────
    print("=== D1  SYMMETRY CENSUS ===")
    print(f"  complexes: {len(ds)}   size dist: {dict(bin_counter)}")
    print(f"  detected rotational symmetry: {report['D1_symmetry']['frac_with_detected_rotational_symmetry']:.1%}")
    print(f"  top point groups: {list(pg_counter.most_common(8))}")
    ls = report["D1_symmetry"]["large_split"]
    print(f"  n≥21 split (N={ls['n']}): {ls['frac_symmetric']:.1%} symmetric, "
          f"mean n={ls['mean_n']:.1f} → mean ASU n={ls['mean_effective_n_after_asu']:.1f}")
    print(f"  n≥21 with a single-orbit annotated order (exact τ meaningless): {ls['frac_sym_degenerate_gt']:.1%}")

    print("\n=== D2  CHIRALITY BLIND SPOT ===")
    print(f"  {report['D2_chirality']['verdict']}")

    print("\n=== D3  BSA TEACHER (Lever 4) ===")
    print(f"  overall τ={report['D3_bsa_teacher']['overall_tau']:.3f}  "
          f"top1={report['D3_bsa_teacher']['overall_top1']:.3f}")
    for b, v in report["D3_bsa_teacher"]["tau_by_bin"].items():
        t1 = report["D3_bsa_teacher"]["top1_by_bin"][b]["mean"]
        tie = report["D3_bsa_teacher"]["tied_step_frac_by_bin"][b]["mean"]
        print(f"    {b:>6}  N={v['n']:<5} τ={v['mean']:.3f}  top1={t1:.3f}  tied_steps={tie:.1%}")

    print("\n=== D4  HIERARCHICAL REDUCTION (Lever 2) ===")
    for b, v in report["D4_hierarchical"]["mean_effective_n_by_bin"].items():
        red = report["D4_hierarchical"]["mean_reduction_by_bin"][b]["mean"]
        con = report["D4_hierarchical"]["within_module_contiguity_by_bin"][b]["mean"]
        print(f"    {b:>6}  N={v['n']:<5} effective_n={v['mean']:.1f} ({red:.0%} of n)  "
              f"GT stays within module {con:.1%} of steps")
    dl = report["D4_hierarchical"]["large_split"]
    print(f"  n≥21: mean n={dl['mean_n']:.1f} → effective_n={dl['mean_effective_n']:.1f} "
          f"across {dl['mean_modules']:.1f} modules")

    print(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()
