"""
diagnose_quaternary.py — Lever 7 diagnostics: chirality and binding patterns in
nonstandard quaternary structure (rings, helices, general homomers).

Same discipline as `diagnose_scaling.py`: measure before spending training
compute. Nothing here trains anything.

  Q1  Does the new handedness descriptor actually represent handedness?
      Mirror the coordinates and recompute. The screw *rise* must flip sign; the
      screw *angle* and the isologous overlap must not move at all. This is the
      test `chiral_descriptors.py` passes trivially (any triple product flips)
      but which says nothing about whether the flipped quantity is the physically
      meaningful one. Here the quantity is the rise of the operator that maps one
      copy onto the next — the thing that makes a helix left- or right-handed.

  Q2  Interface-type census (Monod; Levy/Teichmann).
      How much of the corpus is isologous (2-fold, self-terminating) vs
      heterologous (head-to-tail, ring/helix forming)? Heterologous fraction is
      the fraction of the corpus where "which copy comes next" is a real
      question with a geometric answer, rather than a labelling artifact.

  Q3  Orbit degeneracy — how much of the current training signal is noise?
      At every supervised step, count how many *interchangeable* chains the true
      next subunit is drawn from. Exact cross-entropy against one of k identical
      copies has an irreducible floor of log k nats that a perfect model still
      pays, and whose gradient is pure noise. Reported per size bin, alongside
      how much of it the contact-restricted (frontier) target set removes.

  Q4  Frontier dilution — is mean-pooling over the assembled core the size bug?
      The scorer feeds the score head `edge_feats[cand, assembled].mean(1)`.
      Rank the true next subunit by mean- vs max- vs sum-over-core buried area
      and compare top-1.

      This one came back NEGATIVE and the negative is worth keeping: max is not
      better than mean in any size bin, and mean and sum agree exactly — as they
      must, since the divisor is constant across candidates at a fixed step, so
      dilution moves the feature's scale and not its argmax. The `sum` column is
      therefore also a self-check on this script.

      The useful finding is the level, not the contrast: mean-pooled buried area
      alone reaches top-1 = 0.294 at n=21–40, matching the BSA teacher and
      beating every trained condition in the scaling ablation. The statistic is
      available and the trained model is losing it — which indicts the objective
      (Q3), not the pooling.

Output → results/diagnostics/quaternary.json + a printed summary.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.evaluation.symmetry import detect_symmetry, size_bin, orbit_id_map
from src.features.quaternary import (
    load_ca_with_resnums, pair_screw, isologous_score, classify_interface, mirror_probe,
)
from src.training.orbit_loss import degeneracy_stats
from train_gnn import load_labeled_15k_dataset

OUT = ROOT / "results" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

BINS = ["2-4", "5-10", "11-20", "21-40", "41+"]


def _valid(c) -> bool:
    n = len(c.chain_ids)
    gt = list(c.gt_order)
    return (len(gt) >= 4 and n >= 3 and c.sasa_matrix.shape == (n, n)
            and len(set(gt)) == len(gt) and all(0 <= v < n for v in gt))


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Q1 — mirror probe on the screw descriptors
# ──────────────────────────────────────────────────────────────────────────────

def q1_mirror(ds, limit: int = 120) -> dict:
    rows = []
    for c in ds:
        if not c.pdb_path:
            continue
        r = mirror_probe(c.pdb_id, c.chain_ids, c.pdb_path, c.uniprot_ids, c.sequences)
        if r.get("ok") and r["n_pairs"]:
            rows.append(r)
        if len(rows) >= limit:
            break
    if not rows:
        return {"n": 0}
    chiral = [r for r in rows if r["n_chiral_pairs"] > 0]
    return {
        "n_complexes": len(rows),
        "n_pairs": int(sum(r["n_pairs"] for r in rows)),
        "n_with_nonzero_rise": len(chiral),
        "mean_abs_rise": _mean([r["mean_abs_rise"] for r in rows]),
        # These three must all be ~0 for the descriptor to be correct.
        "rise_sign_flip_residual": max(r["rise_sign_flip_residual"] for r in rows),
        "theta_invariance_residual": max(r["theta_invariance_residual"] for r in rows),
        "iso_invariance_residual": max(r["iso_invariance_residual"] for r in rows),
        "rise_sign_flip_fraction": _mean([r["rise_sign_flip_fraction"] for r in chiral]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Q2 — interface-type census
# ──────────────────────────────────────────────────────────────────────────────

def q2_interface_types(ds, limit: int = 400) -> dict:
    per_bin: dict[str, Counter] = {b: Counter() for b in BINS}
    per_bin_iso: dict[str, list] = {b: [] for b in BINS}
    pg = Counter()
    seen = 0

    for c in ds:
        if not c.pdb_path:
            continue
        n = len(c.chain_ids)
        ca = load_ca_with_resnums(c.pdb_path, c.chain_ids)
        if len(ca) < 2:
            continue
        sym = detect_symmetry(c.pdb_id, c.chain_ids, c.uniprot_ids, c.sequences, c.pdb_path)
        oid = orbit_id_map(sym.orbits)
        w = np.array(c.sasa_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(w, 0.0)
        b = size_bin(n)
        pg[sym.point_group] += 1
        hit = False

        for orb in sym.orbits:
            if len(orb) < 2:
                continue
            for a_i in range(len(orb)):
                for b_i in range(a_i + 1, len(orb)):
                    i, j = orb[a_i], orb[b_i]
                    if w[i, j] <= 0:
                        continue
                    ci, cj = c.chain_ids[i], c.chain_ids[j]
                    if ci not in ca or cj not in ca:
                        continue
                    sc = pair_screw(ca[ci], ca[cj])
                    if sc is None:
                        continue
                    theta, rise, _ = sc
                    iso = isologous_score(ca[ci], ca[cj])
                    per_bin[b][classify_interface(iso, theta, rise)] += 1
                    per_bin_iso[b].append(iso)
                    hit = True
        if hit:
            seen += 1
        if seen >= limit:
            break

    out = {"n_complexes": seen, "point_groups": dict(pg.most_common(10)), "by_bin": {}}
    for b in BINS:
        tot = sum(per_bin[b].values())
        if not tot:
            continue
        out["by_bin"][b] = {
            "n_homomeric_contacts": tot,
            "mean_isologous_overlap": _mean(per_bin_iso[b]),
            **{k: per_bin[b][k] / tot for k in
               ("isologous", "heterologous", "helical", "translational", "mixed")},
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Q3 — orbit degeneracy / cross-entropy floor
# ──────────────────────────────────────────────────────────────────────────────

def q3_degeneracy(ds) -> dict:
    per_bin: dict[str, list] = defaultdict(list)
    for c in ds:
        n = len(c.chain_ids)
        sym = detect_symmetry(c.pdb_id, c.chain_ids, c.uniprot_ids, c.sequences, c.pdb_path)
        oid = orbit_id_map(sym.orbits)
        w = np.array(c.sasa_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(w, 0.0)
        st = degeneracy_stats(n, c.gt_order, oid, contact=w)
        if st.get("steps"):
            per_bin[size_bin(n)].append(st)

    out = {}
    for b in BINS:
        rows = per_bin.get(b, [])
        if not rows:
            continue
        out[b] = {
            "complexes": len(rows),
            "ambiguous_step_frac": _mean([r["ambiguous_frac"] for r in rows]),
            "mean_orbit_k": _mean([r["mean_orbit_k"] for r in rows]),
            "max_orbit_k": int(max(r["max_orbit_k"] for r in rows)),
            # nats of cross-entropy a *perfect* model still pays under the
            # exact-index objective, i.e. pure label noise in the gradient
            "ce_floor_nats": _mean([r["ce_floor"] for r in rows]),
            "ce_floor_after_frontier": _mean([r["ce_floor_frontier"] for r in rows]),
            "frontier_resolved_frac": _mean([r["frontier_resolved_frac"] for r in rows]),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Q4 — frontier dilution: mean vs max pooling over the assembled core
# ──────────────────────────────────────────────────────────────────────────────

def q4_pooling(ds) -> dict:
    per_bin: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for c in ds:
        n = len(c.chain_ids)
        w = np.array(c.sasa_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(w, 0.0)
        order = list(c.gt_order)
        b = size_bin(n)
        sym = detect_symmetry(c.pdb_id, c.chain_ids, c.uniprot_ids, c.sequences, c.pdb_path)
        oid = orbit_id_map(sym.orbits)

        hits_mean = hits_max = hits_sum = steps = 0
        hits_mean_sym = hits_max_sym = 0
        for step in range(1, len(order)):
            assembled = order[:step]
            correct = order[step]
            remaining = [i for i in range(n) if i not in set(assembled)]
            if correct not in remaining or len(remaining) < 2:
                continue
            sub = w[np.ix_(remaining, assembled)]
            picks = {
                "mean": remaining[int(np.argmax(sub.mean(1)))],
                "max": remaining[int(np.argmax(sub.max(1)))],
                "sum": remaining[int(np.argmax(sub.sum(1)))],
            }
            steps += 1
            hits_mean += int(picks["mean"] == correct)
            hits_max += int(picks["max"] == correct)
            hits_sum += int(picks["sum"] == correct)
            g = oid.get(correct, -2)
            hits_mean_sym += int(oid.get(picks["mean"], -1) == g)
            hits_max_sym += int(oid.get(picks["max"], -1) == g)

        if steps:
            per_bin[b]["mean"].append(hits_mean / steps)
            per_bin[b]["max"].append(hits_max / steps)
            per_bin[b]["sum"].append(hits_sum / steps)
            per_bin[b]["mean_sym"].append(hits_mean_sym / steps)
            per_bin[b]["max_sym"].append(hits_max_sym / steps)

    out = {}
    for b in BINS:
        d = per_bin.get(b)
        if not d:
            continue
        out[b] = {"complexes": len(d["mean"]), **{k: _mean(v) for k, v in d.items()}}
        out[b]["max_minus_mean"] = out[b]["max"] - out[b]["mean"]
    return out


# ──────────────────────────────────────────────────────────────────────────────

def main():
    raw = load_labeled_15k_dataset()
    ds = [c for c in raw if _valid(c)]
    print(f"[quat] {len(ds)}/{len(raw)} usable complexes\n")

    print("── Q1  mirror probe on the screw descriptors ──────────────────────")
    q1 = q1_mirror(ds)
    if q1.get("n_complexes"):
        print(f"  {q1['n_complexes']} complexes / {q1['n_pairs']} same-orbit pairs; "
              f"{q1['n_with_nonzero_rise']} have a nonzero rise")
        print(f"  mean |rise|                      {q1['mean_abs_rise']:.3f} Å")
        print(f"  rise sign-flip residual (max)    {q1['rise_sign_flip_residual']:.2e}  "
              f"(0 ⇒ handedness is representable)")
        print(f"  theta invariance residual (max)  {q1['theta_invariance_residual']:.2e}  "
              f"(0 ⇒ rotation angle is achiral, as it must be)")
        print(f"  isologous invariance residual    {q1['iso_invariance_residual']:.2e}")
        print(f"  fraction of chiral pairs flipping sign  "
              f"{q1['rise_sign_flip_fraction']:.3f}")
    else:
        print("  no complexes with coordinates")

    print("\n── Q2  interface-type census ──────────────────────────────────────")
    q2 = q2_interface_types(ds)
    print(f"  {q2['n_complexes']} complexes with a homomeric contact")
    print(f"  point groups: {q2['point_groups']}")
    print(f"  {'bin':<8}{'contacts':>10}{'isologous':>12}{'heterolog':>12}"
          f"{'helical':>10}{'mean J':>9}")
    for b in BINS:
        r = q2["by_bin"].get(b)
        if not r:
            continue
        print(f"  {b:<8}{r['n_homomeric_contacts']:>10}{r['isologous']:>12.3f}"
              f"{r['heterologous']:>12.3f}{r['helical']:>10.3f}"
              f"{r['mean_isologous_overlap']:>9.3f}")

    print("\n── Q3  orbit degeneracy in the training signal ────────────────────")
    q3 = q3_degeneracy(ds)
    print(f"  {'bin':<8}{'cplx':>6}{'ambig step':>12}{'mean k':>9}{'max k':>7}"
          f"{'CE floor':>10}{'+frontier':>11}")
    for b in BINS:
        r = q3.get(b)
        if not r:
            continue
        print(f"  {b:<8}{r['complexes']:>6}{r['ambiguous_step_frac']:>12.3f}"
              f"{r['mean_orbit_k']:>9.2f}{r['max_orbit_k']:>7}"
              f"{r['ce_floor_nats']:>10.3f}{r['ce_floor_after_frontier']:>11.3f}")

    print("\n── Q4  mean vs max pooling over the assembled core ────────────────")
    q4 = q4_pooling(ds)
    print(f"  {'bin':<8}{'cplx':>6}{'mean-pool':>11}{'max-pool':>10}{'sum-pool':>10}"
          f"{'Δ(max−mean)':>13}")
    for b in BINS:
        r = q4.get(b)
        if not r:
            continue
        print(f"  {b:<8}{r['complexes']:>6}{r['mean']:>11.3f}{r['max']:>10.3f}"
              f"{r['sum']:>10.3f}{r['max_minus_mean']:>13.3f}")

    payload = {"q1_mirror": q1, "q2_interface_types": q2,
               "q3_degeneracy": q3, "q4_pooling": q4}
    (OUT / "quaternary.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[quat] wrote {OUT / 'quaternary.json'}")


if __name__ == "__main__":
    main()
