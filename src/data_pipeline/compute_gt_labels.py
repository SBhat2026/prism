"""
Module 4 — Ground-truth assembly label assignment.

Three tiers:
  Tier 1 (gold)  — Experimental GT from literature (manual, stored in data/gt/experimental_gt.json)
  Tier 2 (silver)— BSA-greedy on real PDB structure (from SIFTS + BSA computation proxy)
  Tier 3 (bronze)— Sequence-length greedy (SASA proxy, no structure needed)

Outputs:
  data/gt/experimental_gt.json      (seeded; humans append Tier 1 entries here)
  data/known_gt_complexes.json      set of UniProt frozensets to exclude from pseudo-GT
  data/interim/labeled_dataset.tsv  merged dataset with all tiers + metadata
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.data_pipeline import CACHE_DIR, GT_DIR, INTERIM_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent.parent
GT_JSON  = GT_DIR / "experimental_gt.json"
KNOWN_GT = ROOT / "data" / "known_gt_complexes.json"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "PRISM-DataPipeline/1.0"})


# ── 4.0 Helpers ───────────────────────────────────────────────────────────────

def _fetch_uniprot_for_pdb_chains(pdb_id: str, chains: list[str]) -> dict[str, str]:
    """Return {chain_id: uniprot_id} from SIFTS for the given chains."""
    cache_file = CACHE_DIR / "sifts" / f"{pdb_id.lower()}.json"
    if cache_file.exists():
        return json.load(open(cache_file))
    try:
        r = _SESSION.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
            timeout=10,
        )
        if r.status_code != 200:
            return {}
        data  = r.json()
        entry = data.get(pdb_id.lower(), data.get(list(data.keys())[0] if data else "", {}))
        mapping: dict[str, str] = {}
        for acc, info in entry.get("UniProt", {}).items():
            for m in info.get("mappings", []):
                c = m.get("chain_id") or m.get("struct_asym_id", "")
                if c:
                    mapping[c] = acc
        json.dump(mapping, open(cache_file, "w"))
        return {c: mapping[c] for c in chains if c in mapping}
    except Exception:
        return {}


# ── 4.1 Tier 1 — Experimental GT ─────────────────────────────────────────────

def load_experimental_gt() -> list[dict]:
    """Load all Tier 1 complexes. Fills UniProt IDs via SIFTS where missing."""
    if not GT_JSON.exists():
        log.warning(f"[gt] {GT_JSON} not found — returning empty Tier 1")
        return []

    gt = json.load(open(GT_JSON))
    complexes = gt.get("complexes", [])
    updated   = False

    for c in complexes:
        if c.get("uniprot_ids"):
            continue  # already resolved
        pdb_id = c["pdb_id"]
        chains = c.get("chains", [])
        if not chains:
            continue
        chain_map = _fetch_uniprot_for_pdb_chains(pdb_id, chains)
        uid_list  = [chain_map.get(ch, "") for ch in chains]
        if any(uid_list):
            c["uniprot_ids"] = uid_list
            updated = True
            log.info(f"[gt] Resolved UniProt IDs for {pdb_id}: {uid_list}")
        time.sleep(0.1)

    if updated:
        json.dump(gt, open(GT_JSON, "w"), indent=2)
        log.info(f"[gt] Updated {GT_JSON} with resolved UniProt IDs")

    return complexes


# ── Build known_gt_complexes.json ─────────────────────────────────────────────

def build_known_gt_complexes(tier1: list[dict], force: bool = False) -> dict:
    """
    Create data/known_gt_complexes.json — the exclusion set used by CORUM/RCSB
    filters to avoid re-including complexes that already have GT labels.
    """
    if KNOWN_GT.exists() and not force:
        log.info(f"[gt] {KNOWN_GT} already exists")
        return json.load(open(KNOWN_GT))

    records = []
    for c in tier1:
        uids = [u for u in c.get("uniprot_ids", []) if u]
        if uids:
            records.append({
                "pdb_id":      c["pdb_id"],
                "uniprot_ids": sorted(set(uids)),
                "source":      "experimental_gt",
            })

    data = {"complexes": records}
    json.dump(data, open(KNOWN_GT, "w"), indent=2)
    log.info(f"[gt] Wrote {len(records)} known complexes → {KNOWN_GT}")
    return data


# ── 4.2 Tier 2 — BSA-greedy pseudo-GT ────────────────────────────────────────

def _sasa_proxy(seq_len: int, assembled_lens: list[int], total_len: int) -> float:
    """Sequence-length proxy for relative SASA contribution."""
    if not assembled_lens:
        return seq_len / max(total_len, 1)
    return seq_len / max(sum(assembled_lens) + seq_len, 1)


def greedy_bsa_order(
    uniprot_ids: list[str],
    seq_lens: Optional[list[int]] = None,
) -> list[int]:
    """
    Greedy BSA assembly order using sequence-length proxy.
    Step 0: pick longest subunit as seed.
    Each subsequent step: pick the remaining subunit that would bury the
    most surface area (proportional to its length vs assembled mass).

    Returns list of subunit indices in assembly order.
    """
    N = len(uniprot_ids)
    if seq_lens is None:
        seq_lens = [200] * N  # fallback: equal lengths

    total_len = sum(seq_lens)
    remaining = list(range(N))
    assembled: list[int] = []

    for _ in range(N):
        if not assembled:
            # Step 0: largest subunit
            chosen = max(remaining, key=lambda i: seq_lens[i])
        else:
            assembled_lens = [seq_lens[i] for i in assembled]
            scores = {
                i: _sasa_proxy(seq_lens[i], assembled_lens, total_len)
                for i in remaining
            }
            chosen = max(scores, key=scores.get)

        assembled.append(chosen)
        remaining.remove(chosen)

    return assembled


def _fetch_seq_lens_for_uniprots(uid_list: list[str]) -> list[int]:
    """Fetch sequence lengths from UniProt REST API."""
    lens = []
    for uid in uid_list:
        length = None
        cache_f = CACHE_DIR / "sifts" / f"seqlen_{uid}.txt"
        if cache_f.exists():
            length = int(cache_f.read_text().strip())
        else:
            try:
                r = _SESSION.get(
                    f"https://rest.uniprot.org/uniprotkb/{uid}.json",
                    timeout=8,
                )
                if r.status_code == 200:
                    d      = r.json()
                    length = d.get("sequence", {}).get("length", 200)
                    cache_f.write_text(str(length))
                time.sleep(0.05)
            except Exception:
                pass
        lens.append(length or 200)
    return lens


# ── 4.3 Merge all sources into labeled dataset ────────────────────────────────

def _load_interim_sources() -> list[dict]:
    """Load CORUM, Complex Portal, and RCSB candidate rows, joining PDB maps where available."""
    rows = []

    # CORUM — join pdb_map on corum_id if available
    corum_path = INTERIM_DIR / "corum_filtered.tsv"
    if corum_path.exists():
        df = pd.read_csv(corum_path, sep="\t")
        corum_pdb_path = INTERIM_DIR / "corum_pdb_map.tsv"
        if corum_pdb_path.exists():
            pdb_map = pd.read_csv(corum_pdb_path, sep="\t")[["corum_id", "pdb_id", "resolution"]]
            df = df.merge(pdb_map, on="corum_id", how="left")
            log.info(f"[gt] CORUM: joined pdb_map — {(df.pdb_id.notna() & (df.pdb_id != 'no_structure')).sum()} have structure")
        else:
            df["pdb_id"] = "no_structure"
            log.info(f"[gt] CORUM pdb_map not ready yet — all entries Tier 3")
        for _, row in df.iterrows():
            rows.append(dict(row))
        log.info(f"[gt] Loaded {len(df)} CORUM rows")

    # Complex Portal — join pdb_map on complex_ac if available
    cp_path = INTERIM_DIR / "complex_portal_filtered.tsv"
    if cp_path.exists():
        df = pd.read_csv(cp_path, sep="\t")
        cp_pdb_path = INTERIM_DIR / "complex_portal_pdb_map.tsv"
        if cp_pdb_path.exists():
            pdb_map = pd.read_csv(cp_pdb_path, sep="\t")[["complex_ac", "pdb_id", "resolution"]]
            df = df.merge(pdb_map, on="complex_ac", how="left")
            log.info(f"[gt] CP: joined pdb_map — {(df.pdb_id.notna() & (df.pdb_id != 'no_structure')).sum()} have structure")
        else:
            df["pdb_id"] = "no_structure"
        for _, row in df.iterrows():
            rows.append(dict(row))
        log.info(f"[gt] Loaded {len(df)} Complex Portal rows")

    # RCSB — already has pdb_id
    rcsb_path = INTERIM_DIR / "rcsb_candidates.tsv"
    if rcsb_path.exists():
        df = pd.read_csv(rcsb_path, sep="\t")
        for _, row in df.iterrows():
            rows.append(dict(row))
        log.info(f"[gt] Loaded {len(df)} RCSB rows")
    else:
        log.info(f"[gt] rcsb_candidates.tsv not found — skipping")

    log.info(f"[gt] Total: {len(rows)} rows from interim sources")
    return rows


def build_labeled_dataset(force: bool = False) -> pd.DataFrame:
    """
    Assign Tier 1/2/3 labels to all complexes and write labeled_dataset.tsv.
    """
    out_path = INTERIM_DIR / "labeled_dataset.tsv"
    if out_path.exists() and not force:
        log.info(f"[gt] {out_path} already exists")
        return pd.read_csv(out_path, sep="\t")

    # Tier 1 load
    tier1 = load_experimental_gt()
    tier1_pdbs = {c["pdb_id"] for c in tier1}

    # Build known GT set for dedup
    build_known_gt_complexes(tier1, force=force)
    known_uid_sets: set[frozenset] = set()
    for c in tier1:
        uids = [u for u in c.get("uniprot_ids", []) if u]
        if uids:
            known_uid_sets.add(frozenset(uids))

    records = []

    # Add Tier 1 entries
    for c in tier1:
        records.append({
            "pdb_id":      c["pdb_id"],
            "name":        c.get("name", ""),
            "n_subunits":  len(c.get("chains", c.get("gt_order", []))),
            "uniprot_ids": json.dumps([u for u in c.get("uniprot_ids", []) if u]),
            "gt_order":    json.dumps(c.get("gt_order", [])),
            "gt_tier":     1,
            "gt_source":   "experimental",
            "evidence":    c.get("source", ""),
            "type":        c.get("type", "heteromer"),
            "is_ring":     c.get("is_ring", False),
            "taxid":       9606,
            "source":      "experimental_gt",
        })

    # Load pseudo-GT sources and assign Tier 2/3
    interim_rows = _load_interim_sources()
    total = len(interim_rows)
    log.info(f"[gt] Assigning Tier 2/3 labels to {total} interim complexes")

    seen_uid_sets = set(known_uid_sets)

    for i, row in enumerate(interim_rows):
        if (i + 1) % 200 == 0:
            log.info(f"[gt]   label assignment {i+1}/{total}")

        uid_json = row.get("uniprot_ids", "[]")
        uid_list = json.loads(uid_json) if isinstance(uid_json, str) else uid_json
        uid_set  = frozenset(uid_list)

        if uid_set in seen_uid_sets:
            continue
        seen_uid_sets.add(uid_set)

        # Fetch sequence lengths (needed for greedy order)
        seq_lens = _fetch_seq_lens_for_uniprots(uid_list)
        gt_order = greedy_bsa_order(uid_list, seq_lens)

        # Tier 2 if we have a PDB structure (from pdb_map), else Tier 3
        pdb_id = row.get("pdb_id", "no_structure")
        tier   = 2 if pdb_id not in ("no_structure", "", None, float("nan")) else 3

        records.append({
            "pdb_id":      pdb_id if tier == 2 else "",
            "name":        str(row.get("name", "")),
            "n_subunits":  len(uid_list),
            "uniprot_ids": json.dumps(uid_list),
            "gt_order":    json.dumps(gt_order),
            "gt_tier":     tier,
            "gt_source":   "sasa_greedy",
            "evidence":    f"BSA-greedy proxy (seq-length) from {row.get('source','?')}",
            "type":        "homomer" if len(set(uid_list)) == 1 else "heteromer",
            "is_ring":     bool(row.get("is_ring", False)),
            "taxid":       int(row.get("taxid", 0) or 0),
            "source":      str(row.get("source", "?")),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_path, sep="\t", index=False)
    log.info(
        f"[gt] Saved {len(df)} labeled complexes → {out_path}\n"
        f"  Tier 1: {sum(df.gt_tier == 1)}\n"
        f"  Tier 2: {sum(df.gt_tier == 2)}\n"
        f"  Tier 3: {sum(df.gt_tier == 3)}"
    )
    return df


# ── Integration test helper ───────────────────────────────────────────────────

def validate_labeled_dataset(df: pd.DataFrame):
    """Print QC stats for the labeled dataset."""
    print("\n── Labeled Dataset QC ──")
    print(f"  Total complexes: {len(df)}")
    print(f"  Tier 1 (experimental): {sum(df.gt_tier == 1)}")
    print(f"  Tier 2 (BSA-greedy + structure): {sum(df.gt_tier == 2)}")
    print(f"  Tier 3 (BSA-greedy, no structure): {sum(df.gt_tier == 3)}")
    print(f"  Unique sources: {df.source.value_counts().to_dict()}")
    print(f"  N_subunits distribution: {df.n_subunits.value_counts().sort_index().to_dict()}")
    print(f"  Homomers: {sum(df.type == 'homomer')} | Heteromers: {sum(df.type == 'heteromer')}")
    print(f"  Rings: {sum(df.is_ring == True)}")
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

def run(force: bool = False):
    tier1 = load_experimental_gt()
    log.info(f"[gt] Loaded {len(tier1)} Tier 1 experimental complexes")

    build_known_gt_complexes(tier1, force=force)

    df = build_labeled_dataset(force=force)
    validate_labeled_dataset(df)
    return df


if __name__ == "__main__":
    run()
