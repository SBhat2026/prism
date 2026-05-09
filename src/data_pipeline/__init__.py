"""
PRISM data integration pipeline.

Modules:
  fetch_corum          — CORUM 5.2 download, filter, PDB structure discovery
  fetch_complex_portal — EBI Complex Portal bulk download and parse
  fetch_rcsb_heteromers — RCSB PDB search for high-quality heteromers
  compute_gt_labels    — Assign ground-truth assembly labels (3 tiers)
  run_pipeline         — Orchestrate the full pipeline
"""

from pathlib import Path

ROOT       = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
INTERIM_DIR= DATA_DIR / "interim"
CACHE_DIR  = DATA_DIR / "cache"
GT_DIR     = DATA_DIR / "gt"

for _d in [RAW_DIR, INTERIM_DIR, CACHE_DIR / "sifts", GT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
