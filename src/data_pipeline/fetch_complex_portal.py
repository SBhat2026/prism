"""
Module 2 — EBI Complex Portal bulk download and parsing.

Outputs:
  data/raw/complex_portal_{species}.tsv         raw ComplexTab files
  data/interim/complex_portal_filtered.tsv      filtered complex metadata
  data/interim/complex_portal_pdb_map.tsv       PDB structure assignments
"""

import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.data_pipeline import CACHE_DIR, INTERIM_DIR, RAW_DIR
from src.data_pipeline.fetch_corum import (
    _fetch_entry_info,
    _fetch_sifts_chain_uniprot,
    _fetch_assembly_metadata,
    _fetch_pdb_ids_for_uniprot,
    _is_canonical_uniprot,
    _load_known_uniprot_sets,
    _score_assembly_method,
    find_best_pdb,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SPECIES_URLS = {
    "human": ("https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv",  9606),
    "mouse": ("https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/10090.tsv",10090),
    "yeast": ("https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/559292.tsv",559292),
    "ecoli": ("https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/83333.tsv", 83333),
    "rat":   ("https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/10116.tsv",10116),
    "fly":   ("https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/7227.tsv",  7227),
}

# Matches: uniprotkb:P12345(2), uniprotkb:P12345, or bare P12345(2) / P12345
_UNIPROT_STOICH_RE = re.compile(
    r"(?:uniprotkb:)?([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:\((\d+)\))?",
    re.IGNORECASE,
)
_PDB_XREF_RE = re.compile(r"pdb:([A-Z0-9]{4})", re.IGNORECASE)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "PRISM-DataPipeline/1.0"})


# ── 2.1 Bulk download ─────────────────────────────────────────────────────────

def download_species(species: str, url: str, force: bool = False) -> Optional[Path]:
    out = RAW_DIR / f"complex_portal_{species}.tsv"
    if out.exists() and not force:
        log.info(f"[cp] Already downloaded: {out}")
        return out
    try:
        log.info(f"[cp] Downloading {species} from {url}")
        r = _SESSION.get(url, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        log.info(f"[cp] Saved {len(r.content)//1024} KB → {out}")
        return out
    except Exception as e:
        log.warning(f"[cp] Could not download {species}: {e}")
        return None


# ── 2.2 Parse ComplexTab format ───────────────────────────────────────────────

def parse_stoichiometry(identifiers_str: str) -> tuple[list[str], dict[str, int]]:
    """
    Parse "uniprotkb:P12345(2)|uniprotkb:Q67890(1)|..."
    Returns (expanded_chains, stoichiometry_dict)
    expanded_chains includes duplicates per stoichiometry.
    """
    stoich: dict[str, int] = {}
    expanded: list[str] = []

    for part in str(identifiers_str).split("|"):
        part = part.strip()
        if not part:
            continue
        m = _UNIPROT_STOICH_RE.search(part)
        if not m:
            continue
        uid   = m.group(1).strip()
        count = int(m.group(2)) if m.group(2) else 1
        if not uid or not uid[0].isalpha():
            continue
        stoich[uid] = stoich.get(uid, 0) + count
        expanded.extend([uid] * count)

    return expanded, stoich


def parse_pdb_xrefs(xrefs_str: str) -> list[str]:
    return list({m.group(1).upper() for m in _PDB_XREF_RE.finditer(str(xrefs_str))})


def parse_species_file(path: Path, taxid: int) -> list[dict]:
    log.info(f"[cp] Parsing {path.name}")
    df = pd.read_csv(path, sep="\t", low_memory=False, comment=None, on_bad_lines="skip")
    df.columns = [c.strip().lstrip("#") for c in df.columns]

    # Locate required columns (names vary slightly between releases)
    def find_col(*patterns):
        for p in patterns:
            for c in df.columns:
                if p.lower() in c.lower():
                    return c
        return None

    ac_col       = find_col("complex_ac", "#complex_ac") or df.columns[0]
    name_col     = find_col("recommended_name") or df.columns[1]
    identifiers_col = find_col("identifiers", "molecules")
    confidence_col  = find_col("confidence")
    evidence_col    = find_col("experimental", "evidence")
    go_col          = find_col("go annotation", "go_annotation")
    xref_col        = find_col("cross reference", "xref")
    desc_col        = find_col("description")

    if identifiers_col is None:
        log.warning(f"[cp] No identifiers column found in {path.name}, columns: {list(df.columns)}")
        return []

    known_sets = _load_known_uniprot_sets()
    records    = []

    for _, row in df.iterrows():
        # Filter: exclude explicitly low-confidence entries
        conf = str(row.get(confidence_col, "")) if confidence_col else ""
        if "low" in conf.lower():
            continue
        # Accept if: direct experimental evidence OR physical interaction ECO code
        # ECO:0000353 = physical interaction evidence (manually curated, legitimate)
        # Dropping pure reconstruction-only entries (ECO:0005546/47) is too aggressive —
        # Complex Portal curates these from literature and they are high-quality.
        # Only skip if zero evidence signal at all.
        evid     = str(row.get(evidence_col, "")) if evidence_col else ""
        eco_col  = find_col("evidence code", "eco")
        eco_val  = str(row.get(eco_col, "")) if eco_col else ""
        has_evid = (evid.strip() not in ("", "nan", "None", "-")
                    or eco_val.strip() not in ("", "nan", "None", "-"))
        if not has_evid:
            continue

        expanded, stoich = parse_stoichiometry(row.get(identifiers_col, ""))
        unique_proteins  = [u for u in dict.fromkeys(expanded) if _is_canonical_uniprot(u)]

        if len(unique_proteins) < 2 or len(unique_proteins) > 12:
            continue
        if frozenset(unique_proteins) in known_sets:
            continue

        pdb_ids = parse_pdb_xrefs(row.get(xref_col, "") if xref_col else "")

        records.append({
            "complex_ac":   str(row.get(ac_col, "")),
            "name":         str(row.get(name_col, "")).strip(),
            "organism":     "",
            "taxid":        taxid,
            "n_subunits":   len(unique_proteins),
            "uniprot_ids":  json.dumps(unique_proteins),
            "stoichiometry":json.dumps(stoich),
            "go_ids":       json.dumps([]),
            "pubmed_ids":   json.dumps([]),
            "pdb_xrefs":    json.dumps(pdb_ids),
            "source":       "complex_portal",
        })

    log.info(f"[cp] {path.name}: {len(records)} complexes passed filters")
    return records


# ── 2.3 PDB cross-reference and structure discovery ───────────────────────────

def _verify_pdb_valid(pdb_id: str) -> bool:
    """Check RCSB to confirm PDB entry is still valid."""
    try:
        r = _SESSION.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
            timeout=8,
        )
        return r.status_code == 200
    except Exception:
        return False


def build_pdb_map_cp(filtered_df: pd.DataFrame) -> pd.DataFrame:
    rows  = []
    total = len(filtered_df)

    for i, (_, row) in enumerate(filtered_df.iterrows()):
        uid_list = json.loads(row["uniprot_ids"])
        taxid    = int(row.get("taxid", 9606))
        pdb_xrefs= json.loads(row.get("pdb_xrefs", "[]"))

        if (i + 1) % 50 == 0:
            log.info(f"[cp] PDB map progress {i+1}/{total}")

        result = None

        # Try pre-existing PDB cross-references first
        for pdb_id in pdb_xrefs:
            if not _verify_pdb_valid(pdb_id):
                continue
            chain_map = _fetch_sifts_chain_uniprot(pdb_id)
            covered   = set(uid_list) & set(chain_map.values())
            frac      = len(covered) / len(uid_list)
            if frac >= 0.6:
                assemblies = _fetch_assembly_metadata(pdb_id)
                asm_id, method, _ = _score_assembly_method(assemblies)
                info  = _fetch_entry_info(pdb_id)
                res   = float(info.get("resolution", 99.0) or 99.0)
                score = (frac * 2) + (1.5 if "author" in method.lower() else 0.5) + (1.0 if res < 3.0 else 0)
                if not result or score > result["score"]:
                    result = {
                        "pdb_id": pdb_id,
                        "assembly_id": asm_id,
                        "coverage_fraction": round(frac, 3),
                        "resolution": res,
                        "method": method,
                        "chain_to_uniprot": json.dumps(chain_map),
                        "score": round(score, 3),
                    }
            time.sleep(0.05)

        # Fall back to RCSB search if no xref gave ≥60% coverage
        if not result:
            result = find_best_pdb(uid_list, taxid)

        if result:
            rows.append({
                "complex_ac":       row.get("complex_ac", ""),
                "pdb_id":           result["pdb_id"],
                "assembly_id":      result["assembly_id"],
                "coverage_fraction":result["coverage_fraction"],
                "resolution":       result["resolution"],
                "method":           result["method"],
                "chain_to_uniprot": result.get("chain_to_uniprot", "{}"),
                "source":           "complex_portal",
            })
        else:
            rows.append({
                "complex_ac": row.get("complex_ac",""),
                "pdb_id": "no_structure",
                "assembly_id":"","coverage_fraction":0.0,
                "resolution":99.0,"method":"","chain_to_uniprot":"{}",
                "source":"complex_portal",
            })
        time.sleep(0.1)

    return pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(force: bool = False):
    filtered_path = INTERIM_DIR / "complex_portal_filtered.tsv"
    pdb_map_path  = INTERIM_DIR / "complex_portal_pdb_map.tsv"

    # Download all species
    all_records = []
    for species, (url, taxid) in SPECIES_URLS.items():
        raw_path = download_species(species, url, force=force)
        if raw_path is None:
            continue
        all_records.extend(parse_species_file(raw_path, taxid))

    # Deduplicate by frozenset of UniProt IDs
    seen: set[frozenset] = set()
    deduped = []
    for rec in all_records:
        uid_set = frozenset(json.loads(rec["uniprot_ids"]))
        if uid_set not in seen:
            seen.add(uid_set)
            deduped.append(rec)

    filtered_df = pd.DataFrame(deduped)
    log.info(f"[cp] {len(filtered_df)} unique complexes after dedup")

    if not (filtered_path.exists() and not force):
        filtered_df.to_csv(filtered_path, sep="\t", index=False)
        log.info(f"[cp] Saved → {filtered_path}")
    else:
        log.info(f"[cp] Filtered table already exists: {filtered_path}")
        filtered_df = pd.read_csv(filtered_path, sep="\t")

    # Build PDB map
    if pdb_map_path.exists() and not force:
        log.info(f"[cp] PDB map already exists: {pdb_map_path}")
    else:
        log.info(f"[cp] Discovering PDB structures for {len(filtered_df)} complexes…")
        pdb_df = build_pdb_map_cp(filtered_df)
        pdb_df.to_csv(pdb_map_path, sep="\t", index=False)
        log.info(f"[cp] Saved PDB map → {pdb_map_path}")

    return filtered_df, pdb_map_path


if __name__ == "__main__":
    run()
