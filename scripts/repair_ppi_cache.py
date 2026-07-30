"""
repair_ppi_cache.py — undo the STRING score unit bug in already-fetched caches.

`src/data_prep/string_api.py` used to divide the STRING network response by
1000. That response is already normalised to [0, 1] (only the *request*
parameter `required_score` uses the 0-1000 integer scale), so every cached PPI
score was written 1000× too small: a genuine 0.999 was stored as 0.000999.

Consequence: the PPI edge channel had a dynamic range ~1000× smaller than the
SASA channel it sits beside, so the network could not use it at all — it looked
like a dead channel in the feature audit.

This repairs the damage in place, without re-querying STRING:

  1. data/string/*.json          — rescale bugged scores by 1000
  2. data/labeled_15k/*/ppi.npy  — rebuild each complex's matrix from the
                                   repaired pair scores via meta.json uniprot_ids

Idempotent. Bugged values land in [1e-4, 1e-3] (STRING does not report edges
below ~0.15, so 0.15/1000 = 1.5e-4 is the floor); genuine values are ≥ 0.15.
The two ranges do not overlap, so the `< SCALE_THRESHOLD` test is safe to
re-run. Pass --dry-run to see what would change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STRING_CACHE = ROOT / "data" / "string"
LABELED = ROOT / "data" / "labeled_15k"

# Any genuine STRING score is >= ~0.15. Anything below this was divided by 1000.
SCALE_THRESHOLD = 0.01


def repair_string_cache(dry_run: bool = False) -> tuple[int, int, dict]:
    """Rescale bugged entries. Returns (files_touched, values_rescaled, pair_scores)."""
    files = sorted(STRING_CACHE.glob("*.json"))
    touched = rescaled = 0
    pair_scores: dict[tuple[str, str], float] = {}

    for p in files:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        out, changed = {}, False
        for k, v in data.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if 0.0 < val < SCALE_THRESHOLD:
                val = min(val * 1000.0, 1.0)
                rescaled += 1
                changed = True
            out[k] = val
            if ":::" in k:
                a, b = k.split(":::", 1)
                # keep the strongest evidence for a pair seen in several files
                pair_scores[(a, b)] = max(pair_scores.get((a, b), 0.0), val)

        if changed:
            touched += 1
            if not dry_run:
                p.write_text(json.dumps(out, indent=2))

    return touched, rescaled, pair_scores


def rebuild_ppi_matrices(pair_scores: dict, dry_run: bool = False) -> tuple[int, int]:
    """Rewrite data/labeled_15k/*/ppi.npy from the repaired pair scores."""
    dirs = sorted(d for d in LABELED.iterdir() if d.is_dir())
    written = nonzero = 0

    for d in dirs:
        meta_p = d / "meta.json"
        if not meta_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue

        uids = meta.get("uniprot_ids") or []
        n = len(meta.get("chain_ids", []))
        if not uids or len(uids) != n or n < 2:
            continue

        mat = np.zeros((n, n), dtype=np.float32)
        hit = False
        for i in range(n):
            for j in range(n):
                if i == j or not uids[i] or not uids[j]:
                    continue
                s = pair_scores.get((uids[i], uids[j]))
                if s is None:
                    s = pair_scores.get((uids[j], uids[i]))
                if s:
                    mat[i, j] = s
                    hit = True

        if not hit:
            continue
        written += 1
        nonzero += 1
        if not dry_run:
            np.save(str(d / "ppi.npy"), mat)

    return written, nonzero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    touched, rescaled, pairs = repair_string_cache(args.dry_run)
    print(f"[string] {touched} cache files rescaled, {rescaled} values fixed")
    print(f"[string] {len(pairs)} unique UniProt pairs with a score")
    if pairs:
        v = np.array(list(pairs.values()))
        print(f"[string] score range now [{v.min():.4f}, {v.max():.4f}], "
              f"mean {v.mean():.4f}")

    written, _ = rebuild_ppi_matrices(pairs, args.dry_run)
    print(f"[ppi] rebuilt {written} per-complex ppi.npy matrices")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
