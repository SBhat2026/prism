"""
splits.py — Size-stratified splits, including the held-out large-complex split.

Lever 6 of PRISM_scaling_roadmap.md: "freeze all ≥N-chain complexes out of
training to measure true extrapolation, not interpolation."

train / test_small are drawn from complexes with n ≤ large_threshold-1;
test_large contains every complex with n ≥ large_threshold and is *never*
touched by training. Any reported large-n number is therefore extrapolation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "splits"


def _stable_hash(pdb_id: str, salt: str = "prism") -> float:
    h = hashlib.sha1(f"{salt}:{pdb_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def make_size_splits(
    complexes: Sequence,
    large_threshold: int = 21,
    test_small_frac: float = 0.2,
    min_gt_len: int = 4,
    salt: str = "prism",
) -> dict[str, list[str]]:
    """
    complexes: objects with .pdb_id, .chain_ids, .gt_order.
    Returns {"train": [...], "test_small": [...], "test_large": [...]}.

    Hash-based assignment keeps the split identical across runs and machines
    without storing a random seed alongside every result.
    """
    train, test_small, test_large = [], [], []
    for c in complexes:
        n = len(c.chain_ids)
        if len(c.gt_order) < min_gt_len:
            continue
        if n >= large_threshold:
            test_large.append(c.pdb_id)
        elif _stable_hash(c.pdb_id, salt) < test_small_frac:
            test_small.append(c.pdb_id)
        else:
            train.append(c.pdb_id)
    return {"train": sorted(train),
            "test_small": sorted(test_small),
            "test_large": sorted(test_large)}


def save_splits(splits: dict, name: str = "size_splits") -> Path:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"{name}.json"
    path.write_text(json.dumps({k: v for k, v in splits.items()}, indent=2))
    return path


def load_splits(name: str = "size_splits") -> dict:
    path = SPLIT_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}
