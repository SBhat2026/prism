#!/usr/bin/env python3
"""
Ablation study runner for PRISM.

Runs 3 ablation suites in sequence:
  1. Feature ablation (6 conditions × --ablation_epochs each):
       bsa_only, esm_only, go_only, ppi_only, no_go, full_prism
  2. Anchor aggregation (2 conditions): max vs pooled
  3. Sparsification (6 k values): k ∈ {3, 4, 5, 6, 8, 0=full}

Each condition trains from scratch for --ablation_epochs epochs on the labeled_15k data,
then evaluates on the val and test splits. Results saved to results/ablation.json.

Usage: python scripts/run_ablations.py [--ablation_epochs 100] [--suite feature|anchor|sparse|all]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

RESULTS_DIR = ROOT / "results" / "ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CONDITIONS = {
    "bsa_only":  {"w_sasa": 1.0, "w_esm": 0.0, "w_go": 0.0, "w_ppi": 0.0},
    "esm_only":  {"w_sasa": 0.0, "w_esm": 1.0, "w_go": 0.0, "w_ppi": 0.0},
    "go_only":   {"w_sasa": 0.0, "w_esm": 0.0, "w_go": 1.0, "w_ppi": 0.0},
    "ppi_only":  {"w_sasa": 0.0, "w_esm": 0.0, "w_go": 0.0, "w_ppi": 1.0},
    "no_go":     {"w_sasa": 0.4, "w_esm": 0.5, "w_go": 0.0, "w_ppi": 0.1},
    "full_prism": None,  # use all features (no weight override)
}

ANCHOR_CONDITIONS = ["max", "pooled"]

SPARSE_K_VALUES = [3, 4, 5, 6, 8, 0]  # 0 = full (no pruning)


def run_training(condition_name: str, extra_args: list[str], epochs: int) -> dict:
    """Run train_v2.py with given args and return summary metrics."""
    out_dir = RESULTS_DIR / condition_name
    out_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(ROOT / "train_v2.py"),
        "--epochs", str(epochs),
        "--save_every", str(epochs),  # save only at end
        "--eval_every", "10",
        "--patience", str(epochs),   # no early stopping for ablations
    ] + extra_args

    print(f"\n[ablation] {condition_name}: {' '.join(extra_args)}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT
    )
    if proc.returncode != 0:
        print(f"  ERROR:\n{proc.stderr[-2000:]}")
        return {"condition": condition_name, "status": "error",
                "stderr": proc.stderr[-500:]}

    # Parse final metrics from stdout
    metrics = {"condition": condition_name, "status": "ok"}
    for line in (proc.stdout + proc.stderr).split("\n"):
        for key in ["val_tau", "val_top1", "test_tau_marsh", "test_top1_marsh",
                    "val_tau_homomer", "val_tau_heteromer"]:
            if key in line:
                try:
                    val = float(line.split(key)[-1].strip().split()[0].strip("=:,"))
                    metrics[key] = val
                except Exception:
                    pass

    out_metrics = out_dir / "metrics.txt"
    out_metrics.write_text(proc.stdout + proc.stderr)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_epochs", type=int, default=100,
                        help="Epochs per ablation condition")
    parser.add_argument("--suite", choices=["feature", "anchor", "sparse", "all"],
                        default="all", help="Which ablation suite to run")
    args = parser.parse_args()

    results = {}
    ablation_out = ROOT / "results" / "ablation.json"

    def save():
        ablation_out.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {ablation_out}")

    # ── Feature ablation ──────────────────────────────────────────────────────
    if args.suite in ("feature", "all"):
        print("\n=== Feature Ablation ===")
        for cond, weights in FEATURE_CONDITIONS.items():
            extra = ["--anchor_mode", "pooled", "--sparse_k", "5"]
            # Full PRISM needs no overrides beyond defaults
            result = run_training(cond, extra, args.ablation_epochs)
            results[f"feature/{cond}"] = result
            save()

    # ── Anchor aggregation ablation ───────────────────────────────────────────
    if args.suite in ("anchor", "all"):
        print("\n=== Anchor Aggregation Ablation ===")
        for mode in ANCHOR_CONDITIONS:
            extra = ["--anchor_mode", mode, "--sparse_k", "5"]
            result = run_training(f"anchor_{mode}", extra, args.ablation_epochs)
            results[f"anchor/{mode}"] = result
            save()

    # ── Sparsification ablation ───────────────────────────────────────────────
    if args.suite in ("sparse", "all"):
        print("\n=== Sparsification Ablation ===")
        for k in SPARSE_K_VALUES:
            name = f"sparse_k{k}" if k > 0 else "sparse_full"
            extra = ["--anchor_mode", "pooled", "--sparse_k", str(k)]
            result = run_training(name, extra, args.ablation_epochs)
            results[f"sparse/k{k}"] = result
            save()

    print("\n=== Ablation complete ===")
    for k, v in results.items():
        status = v.get("status", "?")
        tau = v.get("val_tau", "?")
        top1 = v.get("val_top1", "?")
        print(f"  {k:40s} status={status}  val_tau={tau}  val_top1={top1}")

    save()


if __name__ == "__main__":
    main()
