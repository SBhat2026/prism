"""
Statistical significance testing between Marsh 2013 and Experimental GT v2 metrics.

Tests:
  - Paired Wilcoxon signed-rank on τ (if sets are paired; unpaired otherwise)
  - Mann-Whitney U test (unpaired two-sample) comparing the two sets
  - Bootstrap CI (1000 resamples) on mean τ for each set
  - Cohen's d effect size

Reads results from results/expanded_benchmark/results.json (run expanded_benchmark.py first).

Usage:
    python src/evaluation/significance_test.py
    python src/evaluation/significance_test.py --metric top1_accuracy
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "results" / "expanded_benchmark" / "results.json"
OUT_DIR      = ROOT / "results" / "expanded_benchmark"


def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap CI for the mean. Returns (mean, lo, hi)."""
    arr = np.array(values)
    means = [np.mean(arr[np.random.randint(len(arr), size=len(arr))]) for _ in range(n_resamples)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(arr.mean()), float(lo), float(hi)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d (pooled SD)."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def interpret_p(p: float) -> str:
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    if p < 0.10:    return "~"
    return "ns"


def run_tests(marsh_vals: list[float], v2_vals: list[float], metric: str):
    from scipy.stats import wilcoxon, mannwhitneyu

    print(f"\n{'='*60}")
    print(f"  Significance tests — metric: {metric}")
    print('='*60)

    # Bootstrap CI for each set
    m_mean, m_lo, m_hi = bootstrap_ci(marsh_vals)
    v_mean, v_lo, v_hi = bootstrap_ci(v2_vals)

    print(f"\n  Marsh 2013  (n={len(marsh_vals)}):")
    print(f"    mean = {m_mean:.4f}  95% CI [{m_lo:.4f}, {m_hi:.4f}]")
    print(f"\n  Exp GT v2   (n={len(v2_vals)}):")
    print(f"    mean = {v_mean:.4f}  95% CI [{v_lo:.4f}, {v_hi:.4f}]")
    print(f"\n  Δ (v2 − Marsh) = {v_mean - m_mean:+.4f}")

    # Cohen's d
    d = cohens_d(marsh_vals, v2_vals)
    d_interp = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
    print(f"\n  Cohen's d = {d:.3f} ({d_interp})")

    # Mann-Whitney U (unpaired)
    try:
        u_stat, p_mw = mannwhitneyu(marsh_vals, v2_vals, alternative="two-sided")
        print(f"\n  Mann-Whitney U = {u_stat:.1f},  p = {p_mw:.4f}  {interpret_p(p_mw)}")
    except Exception as e:
        print(f"\n  Mann-Whitney U: {e}")
        p_mw = None

    # Paired Wilcoxon (only if same length)
    if len(marsh_vals) == len(v2_vals):
        try:
            diffs = np.array(marsh_vals) - np.array(v2_vals)
            if np.all(diffs == 0):
                print("  Paired Wilcoxon: all differences are zero — cannot compute")
            else:
                w_stat, p_w = wilcoxon(marsh_vals, v2_vals, alternative="two-sided")
                print(f"  Paired Wilcoxon W = {w_stat:.1f},  p = {p_w:.4f}  {interpret_p(p_w)}")
        except Exception as e:
            print(f"  Paired Wilcoxon: {e}")
    else:
        print(f"  Paired Wilcoxon: skipped (n_marsh={len(marsh_vals)} ≠ n_v2={len(v2_vals)})")

    print(f"\n  Legend: *** p<.001  ** p<.01  * p<.05  ~ p<.10  ns p≥.10")
    print('='*60)

    return {
        "metric": metric,
        "marsh_n": len(marsh_vals),
        "v2_n": len(v2_vals),
        "marsh_mean": m_mean,
        "marsh_ci_lo": m_lo,
        "marsh_ci_hi": m_hi,
        "v2_mean": v_mean,
        "v2_ci_lo": v_lo,
        "v2_ci_hi": v_hi,
        "delta": float(v_mean - m_mean),
        "cohens_d": d,
        "mannwhitney_p": float(p_mw) if p_mw is not None else None,
    }


def plot_ci_bars(all_results: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [r["metric"] for r in all_results]
    x = np.arange(len(metrics))
    w = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(all_results):
        m_err = [[r["marsh_mean"] - r["marsh_ci_lo"]], [r["marsh_ci_hi"] - r["marsh_mean"]]]
        v_err = [[r["v2_mean"] - r["v2_ci_lo"]], [r["v2_ci_hi"] - r["v2_mean"]]]
        ax.bar(i - w/2, r["marsh_mean"], width=w, color="#4C72B0", alpha=0.8, label="Marsh 2013" if i == 0 else "")
        ax.errorbar(i - w/2, r["marsh_mean"], yerr=m_err, fmt="none", color="black", capsize=4)
        ax.bar(i + w/2, r["v2_mean"], width=w, color="#DD8452", alpha=0.8, label="Exp GT v2" if i == 0 else "")
        ax.errorbar(i + w/2, r["v2_mean"], yerr=v_err, fmt="none", color="black", capsize=4)

        # significance star
        p = r.get("mannwhitney_p")
        if p is not None:
            sig = interpret_p(p)
            ymax = max(r["marsh_ci_hi"], r["v2_ci_hi"]) + 0.02
            ax.annotate(sig, xy=(i, ymax), ha="center", fontsize=10, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels([r["metric"].replace("_", "\n") for r in all_results], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Marsh 2013 vs Experimental GT v2 — 95% Bootstrap CI", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "significance.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="all",
                        help="Metric to test: tau|top1_accuracy|mean_confidence|early_step_tau|all")
    parser.add_argument("--results", default=str(RESULTS_PATH))
    args = parser.parse_args()

    rpath = Path(args.results)
    if not rpath.exists():
        print(f"[error] {rpath} not found — run expanded_benchmark.py first")
        sys.exit(1)

    data = json.loads(rpath.read_text())
    marsh_rows = data.get("marsh", [])
    v2_rows    = data.get("v2", [])

    if not marsh_rows:
        print("[error] no Marsh 2013 results in file")
        sys.exit(1)
    if not v2_rows:
        print("[error] no experimental v2 results in file")
        sys.exit(1)

    print(f"[loaded] Marsh n={len(marsh_rows)}, ExpV2 n={len(v2_rows)}")

    if args.metric == "all":
        metrics_to_test = ["tau", "top1_accuracy", "mean_confidence", "early_step_tau"]
    else:
        metrics_to_test = [args.metric]

    np.random.seed(42)
    all_sig_results = []
    for metric in metrics_to_test:
        marsh_vals = [r[metric] for r in marsh_rows if metric in r]
        v2_vals    = [r[metric] for r in v2_rows    if metric in r]
        sig = run_tests(marsh_vals, v2_vals, metric)
        all_sig_results.append(sig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "significance_results.json"
    out_json.write_text(json.dumps(all_sig_results, indent=2))
    print(f"\n[saved] {out_json}")

    plot_ci_bars(all_sig_results)


if __name__ == "__main__":
    main()
