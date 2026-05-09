"""
Pipeline orchestrator — run all four modules in order.

Usage:
  python -m src.data_pipeline.run_pipeline            # full run
  python -m src.data_pipeline.run_pipeline --force    # re-download everything
  python -m src.data_pipeline.run_pipeline --module corum
  python -m src.data_pipeline.run_pipeline --module complex_portal
  python -m src.data_pipeline.run_pipeline --module rcsb
  python -m src.data_pipeline.run_pipeline --module labels
  python -m src.data_pipeline.run_pipeline --dry-run   # just print plan

All outputs land in data/ with deterministic filenames.
Re-runs are idempotent unless --force is passed.
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_corum(force: bool):
    log.info("══ Module 1: CORUM 5.2 ══")
    from src.data_pipeline.fetch_corum import run
    t0 = time.time()
    filtered_df, pdb_map_path = run(force=force)
    log.info(f"  CORUM done in {time.time()-t0:.1f}s — "
             f"{len(filtered_df)} filtered complexes, PDB map → {pdb_map_path}")


def run_complex_portal(force: bool):
    log.info("══ Module 2: Complex Portal ══")
    from src.data_pipeline.fetch_complex_portal import run
    t0 = time.time()
    filtered_df, pdb_map_path = run(force=force)
    log.info(f"  Complex Portal done in {time.time()-t0:.1f}s — "
             f"{len(filtered_df)} filtered complexes")


def run_rcsb(force: bool):
    log.info("══ Module 3: RCSB Heteromers ══")
    from src.data_pipeline.fetch_rcsb_heteromers import run
    t0 = time.time()
    # max_candidates=None for full run; set to e.g. 2000 for a quick test
    df = run(force=force, max_candidates=None)
    log.info(f"  RCSB done in {time.time()-t0:.1f}s — {len(df)} candidates")


def run_labels(force: bool):
    log.info("══ Module 4: GT Label Assignment ══")
    from src.data_pipeline.compute_gt_labels import run
    t0 = time.time()
    df = run(force=force)
    log.info(f"  Label assignment done in {time.time()-t0:.1f}s — {len(df)} total complexes")


def print_plan():
    """Print expected outputs without running anything."""
    from src.data_pipeline import DATA_DIR, INTERIM_DIR, RAW_DIR, GT_DIR
    print("\nPipeline execution plan:")
    outputs = [
        ("Module 1", RAW_DIR / "corum_allComplexes.txt",          "CORUM 5.2 raw download"),
        ("Module 1", INTERIM_DIR / "corum_filtered.tsv",           "CORUM filtered complexes"),
        ("Module 1", INTERIM_DIR / "corum_pdb_map.tsv",            "CORUM PDB structure map"),
        ("Module 2", RAW_DIR / "complex_portal_human.tsv",         "Complex Portal human"),
        ("Module 2", INTERIM_DIR / "complex_portal_filtered.tsv",  "Complex Portal filtered"),
        ("Module 2", INTERIM_DIR / "complex_portal_pdb_map.tsv",   "Complex Portal PDB map"),
        ("Module 3", INTERIM_DIR / "rcsb_candidates.tsv",          "RCSB heteromer candidates"),
        ("Module 4", GT_DIR / "experimental_gt.json",              "Tier 1 experimental GT"),
        ("Module 4", DATA_DIR / "known_gt_complexes.json",         "Known GT exclusion set"),
        ("Module 4", INTERIM_DIR / "labeled_dataset.tsv",          "Full labeled dataset"),
    ]
    for module, path, desc in outputs:
        status = "✅ exists" if path.exists() else "⬜ pending"
        print(f"  {status}  [{module}] {desc}")
        print(f"           {path.relative_to(path.parent.parent.parent)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="PRISM data integration pipeline")
    parser.add_argument("--force",    action="store_true", help="Re-download and recompute everything")
    parser.add_argument("--module",   choices=["corum","complex_portal","rcsb","labels"],
                        help="Run only one module")
    parser.add_argument("--dry-run",  action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    print_plan()
    if args.dry_run:
        return

    t_start = time.time()

    if args.module == "corum" or args.module is None:
        run_corum(args.force)

    if args.module == "complex_portal" or args.module is None:
        run_complex_portal(args.force)

    if args.module == "rcsb" or args.module is None:
        run_rcsb(args.force)

    if args.module == "labels" or args.module is None:
        run_labels(args.force)

    log.info(f"Pipeline complete in {(time.time()-t_start)/60:.1f} min")
    print_plan()


if __name__ == "__main__":
    main()
