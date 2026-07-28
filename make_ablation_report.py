"""
make_ablation_report.py — turn results/ablation_scaling/metrics.json (+ the
Step-1 diagnostics) into a single readable REPORT.md with the tables you'd
actually paste into a lab notebook.

Run after run_ablation.py. Safe to re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
ABL = ROOT / "results" / "ablation_scaling"
DIAG = ROOT / "results" / "diagnostics"

KEYS = [("top1", "top-1"), ("gt_prob", "P(GT)"), ("tau", "τ"),
        ("tau_sym", "τ_sym"), ("tau_ring", "τ_ring")]
BINS = ["2-4", "5-10", "11-20", "21-40", "41+"]


def f(v, nd=3):
    try:
        if v != v:      # NaN
            return "—"
        return f"{v:.{nd}f}"
    except Exception:
        return "—"


def table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


CONDITION_ORDER = [
    "random", "bsa_teacher", "interface_only", "esm_only", "esm_shuffled",
    "no_esm", "v4_baseline", "v4_plus_chiral", "all_features",
]


def load_metrics() -> tuple[dict, dict]:
    """Merge the parallel shards (metrics_a.json …) or fall back to metrics.json."""
    shards = sorted(ABL.glob("metrics_*.json"))
    if not shards:
        m = json.loads((ABL / "metrics.json").read_text())
        return m["meta"], m["results"]

    meta, res = {}, {}
    for p in shards:
        m = json.loads(p.read_text())
        meta = {**m["meta"], **meta} if meta else m["meta"]
        res.update(m["results"])
    ordered = {k: res[k] for k in CONDITION_ORDER if k in res}
    ordered.update({k: v for k, v in res.items() if k not in ordered})

    # Merge the shard-local per_complex CSVs into one file.
    parts = sorted(ABL.glob("per_complex_*.csv"))
    if parts:
        merged, header = [], None
        for p in parts:
            lines = p.read_text().splitlines()
            if not lines:
                continue
            header = header or lines[0]
            merged += lines[1:]
        (ABL / "per_complex.csv").write_text("\n".join([header] + merged) + "\n")
    return meta, ordered


def main():
    meta, res = load_metrics()
    diag = json.loads((DIAG / "diagnostics.json").read_text()) if (DIAG / "diagnostics.json").exists() else {}

    lines: list[str] = []
    A = lines.append

    A("# PRISM scaling — ablation + diagnostics report")
    A("")
    A(f"Generated from `results/ablation_scaling/metrics.json`. "
      f"Device `{meta['device']}`, node_dim {meta['node_dim']}, edge_dim {meta['edge_dim']}.")
    A(f"Split: **{meta['n_train']} train / {meta['n_test_small']} test_small (n≤20) / "
      f"{meta['n_test_large']} test_large (n≥{meta['args']['large_threshold']}, frozen out of training)**.")
    A(f"Every condition: identical architecture, parameter count, seed, epochs "
      f"({meta['args']['epochs']}) and split — only the feature mask differs.")
    A("")

    # ── headline table: test_large ────────────────────────────────────────────
    for split, title in (("test_small", "Interpolation — held-out small complexes (n ≤ 20)"),
                         ("test_large", "Extrapolation — held-out large complexes (n ≥ 21, never trained on)")):
        A(f"## {title}")
        A("")
        rows = []
        for name, r in res.items():
            a = r["splits"].get(split, {})
            rows.append([f"`{name}`"] + [f(a.get(k)) for k, _ in KEYS] + [r["note"]])
        A(table(rows, ["condition"] + [lbl for _, lbl in KEYS] + ["what it is"]))
        A("")

    # ── size degradation ──────────────────────────────────────────────────────
    A("## Degradation with complex size (top-1 next-subunit accuracy)")
    A("")
    rows = []
    for name, r in res.items():
        rows.append([f"`{name}`"] + [f(r["bins"].get(b, {}).get("top1")) for b in BINS])
    A(table(rows, ["condition"] + [f"n={b}" for b in BINS]))
    A("")
    A("_Counts per bin:_ " + ", ".join(
        f"{b}={next(iter(res.values()))['bins'].get(b, {}).get('n_complexes', 0)}" for b in BINS))
    A("")

    # ── symmetry ladder ───────────────────────────────────────────────────────
    A("## How much of the large-n error is symmetry bookkeeping?")
    A("")
    A("τ → τ_sym → τ_ring progressively quotients out (a) within-orbit chain "
      "relabelling and (b) ring rotation/reflection. The gap is error that "
      "exact-order scoring charges the model for but which carries no information.")
    A("")
    rows = []
    for name, r in res.items():
        a = r["splits"].get("test_large", {})
        gap = (a.get("tau_ring", float("nan")) - a.get("tau", float("nan")))
        rows.append([f"`{name}`", f(a.get("tau")), f(a.get("tau_sym")),
                     f(a.get("tau_sym_nondeg")), f(a.get("tau_ring")), f(gap)])
    A(table(rows, ["condition", "τ", "τ_sym", "τ_sym (non-degenerate only)", "τ_ring", "τ_ring − τ"]))
    A("")

    # ── diagnostics ───────────────────────────────────────────────────────────
    if diag:
        A("## Step-1 diagnostics (`diagnose_scaling.py`)")
        A("")
        d1 = diag["D1_symmetry"]; ls = d1["large_split"]
        A(f"**D1 — symmetry census.** {diag['n_complexes']} usable complexes. "
          f"{d1['frac_with_detected_rotational_symmetry']:.1%} have a detected rotational "
          f"point group overall, but **{ls['frac_symmetric']:.1%}** of the n≥21 split does. "
          f"Asymmetric-unit reduction collapses mean n **{ls['mean_n']:.1f} → "
          f"{ls['mean_effective_n_after_asu']:.1f}**. "
          f"{ls['frac_sym_degenerate_gt']:.1%} of the large split has an annotated order "
          f"confined to a single orbit, where exact-order τ is not even well defined.")
        A("")
        A(f"**D2 — chirality.** {diag['D2_chirality']['verdict']}")
        A("")
        d3 = diag["D3_bsa_teacher"]
        A(f"**D3 — BSA teacher (Lever 4).** Free-running order agreement with the "
          f"annotated labels is τ={d3['overall_tau']:.3f}, top-1={d3['overall_top1']:.3f}. "
          "By size:")
        A("")
        A(table([[b,
                  str(d3["tau_by_bin"][b]["n"]),
                  f(d3["tau_by_bin"][b]["mean"]),
                  f(d3["top1_by_bin"][b]["mean"]),
                  f"{d3['tied_step_frac_by_bin'][b]['mean']:.1%}"]
                 for b in BINS],
                ["n", "complexes", "τ", "top-1", "steps the rule can't break the tie"]))
        A("")
        d4 = diag["D4_hierarchical"]
        A(f"**D4 — hierarchical reduction (Lever 2).** Interface-graph community "
          f"detection takes the n≥21 split from mean n={d4['large_split']['mean_n']:.1f} to "
          f"effective n={d4['large_split']['mean_effective_n']:.1f} across "
          f"{d4['large_split']['mean_modules']:.1f} modules.")
        A("")
        A(table([[b,
                  str(d4["mean_effective_n_by_bin"][b]["n"]),
                  f(d4["mean_effective_n_by_bin"][b]["mean"], 1),
                  f"{d4['mean_reduction_by_bin'][b]['mean']:.0%}",
                  f"{d4['within_module_contiguity_by_bin'][b]['mean']:.1%}"]
                 for b in BINS],
                ["n", "complexes", "effective n", "fraction of n", "GT steps staying inside a module"]))
        A("")

    A("## Caveats")
    A("")
    A("- 97% of the labelled corpus is tier-2 (heuristic/curated, not experimental "
      "assembly order) and 64% carries only a prefix of the order; every number "
      "here is agreement with those labels, not with in-vivo pathways.")
    A("- τ is computed on the annotated subset of the order, with the prediction "
      "projected onto that subset preserving predicted relative order.")
    A("- Greedy assembly is seeded by the largest-total-buried-area subunit "
      "(identical across conditions, no GT leak). `tau_gtseed` in `metrics.csv` "
      "reports the historical GT-seeded convention for comparison.")
    A("- top-1 and P(GT) are teacher-forced along the annotated order and are "
      "therefore seed-independent — they are the cleanest ablation signal.")
    A("")

    (ABL / "REPORT.md").write_text("\n".join(lines))
    (ABL / "metrics.json").write_text(json.dumps({"meta": meta, "results": res}, indent=2))

    # Rebuild the combined CSV + figure from the merged results.
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from run_ablation import export as _export, plot as _plot
        _export(res, [], meta)
        _plot(res)
    except Exception as e:                                   # figure is optional
        print(f"[warn] could not regenerate csv/plot: {e}")

    print(f"wrote → {ABL / 'REPORT.md'}")


if __name__ == "__main__":
    main()
