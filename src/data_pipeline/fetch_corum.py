"""
Module 1 — CORUM 5.2 download, filter, and PDB structure discovery.

Outputs:
  data/raw/corum_allComplexes.txt       raw CORUM flat file
  data/interim/corum_filtered.tsv       filtered complex metadata
  data/interim/corum_pdb_map.tsv        PDB structure assignments
"""

import io
import json
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.data_pipeline import CACHE_DIR, INTERIM_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Primary: Zenodo mirror (MIPS site no longer serves direct ZIP links)
CORUM_URL  = "https://zenodo.org/api/records/17419058/files/corum_allComplexes.txt/content"
# Fallbacks: MIPS (may require auth now)
CORUM_URL2 = "https://mips.helmholtz-muenchen.de/corum/download/releases/current_files/corum_allComplexes.txt.zip"
CORUM_URL3 = "https://mips.helmholtz-muenchen.de/corum/download/releases/current_files/corum_allComplexes.json.zip"

ORGANISM_TAXID = {
    "Human":                    9606,
    "Mouse":                    10090,
    "Rat":                      10116,
    "Bovine":                   9913,
    "Saccharomyces cerevisiae": 559292,
    "Drosophila melanogaster":  7227,
}

UNIPROT_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "PRISM-DataPipeline/1.0"})


# ── 1.1 Download ──────────────────────────────────────────────────────────────

def download_corum(force: bool = False) -> Path:
    out = RAW_DIR / "corum_allComplexes.txt"
    if out.exists() and not force:
        log.info(f"[corum] Already downloaded: {out} ({out.stat().st_size // 1024} KB)")
        return out

    for url in [CORUM_URL, CORUM_URL2, CORUM_URL3]:
        try:
            log.info(f"[corum] Downloading from {url}")
            r = _SESSION.get(url, timeout=120, verify=False)
            r.raise_for_status()
            content = r.content
            # Plain text file (Zenodo direct content link)
            if content[:3] != b'PK\x03' and b'\t' in content[:2048]:
                out.write_bytes(content)
                log.info(f"[corum] Saved {len(content)//1024} KB (plain text) to {out}")
                return out
            # ZIP file
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                names = z.namelist()
                log.info(f"[corum] ZIP contents: {names}")
                txt_name = next((n for n in names if n.endswith(".txt")), None)
                if txt_name:
                    data = z.read(txt_name)
                    out.write_bytes(data)
                    log.info(f"[corum] Saved {len(data)//1024} KB to {out}")
                    return out
                json_name = next((n for n in names if n.endswith(".json")), None)
                if json_name:
                    return _convert_corum_json(z.read(json_name))
        except Exception as e:
            log.warning(f"[corum] Download failed from {url}: {e}")

    raise RuntimeError("[corum] Could not download CORUM from any URL")


def _convert_corum_json(raw_json: bytes) -> Path:
    """Convert CORUM JSON to tab-delimited format."""
    data = json.loads(raw_json)
    rows = data if isinstance(data, list) else data.get("complexes", [])
    records = []
    for row in rows:
        records.append({
            "ComplexID":   row.get("ComplexID", ""),
            "ComplexName": row.get("ComplexName", ""),
            "Organism":    row.get("Organism", ""),
            "subunits(UniProt IDs)": ";".join(row.get("subunitsUniProtIDs", [])),
            "GO ID":       ";".join(row.get("GOID", [])),
            "PubMed ID":   ";".join(str(p) for p in row.get("PubMedID", [])),
        })
    out = RAW_DIR / "corum_allComplexes.txt"
    pd.DataFrame(records).to_csv(out, sep="\t", index=False)
    log.info(f"[corum] Converted JSON → {out}")
    return out


# ── 1.2 Parse and filter ──────────────────────────────────────────────────────

def _load_known_uniprot_sets() -> set[frozenset]:
    """Load frozensets of UniProt IDs for all known GT complexes."""
    gt_path = Path(__file__).parent.parent.parent / "data" / "gt" / "experimental_gt.json"
    if not gt_path.exists():
        return set()
    gt = json.load(open(gt_path))
    sets = set()
    for c in gt.get("complexes", []):
        uids = [u for u in c.get("uniprot_ids", []) if u]
        if uids:
            sets.add(frozenset(uids))
    return sets


def _is_canonical_uniprot(uid: str) -> bool:
    """Return True if uid is a canonical UniProt accession (no isoform dash)."""
    if not uid or "-" in uid:
        return False
    return bool(UNIPROT_RE.match(uid.strip()))


def parse_and_filter(raw_path: Path) -> pd.DataFrame:
    log.info(f"[corum] Parsing {raw_path}")
    df = pd.read_csv(raw_path, sep="\t", low_memory=False, on_bad_lines="skip")

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    subunit_col = next((c for c in df.columns if "UniProt" in c and "subunit" in c.lower()), None)
    if subunit_col is None:
        subunit_col = next((c for c in df.columns if "subunit" in c.lower()), None)
    if subunit_col is None:
        raise ValueError(f"[corum] Cannot find subunit column in: {list(df.columns)}")
    log.info(f"[corum] Using subunit column: '{subunit_col}'")

    organism_col = next((c for c in df.columns if "organism" in c.lower()), "Organism")
    name_col     = next((c for c in df.columns if "name" in c.lower() and "complex" in c.lower()), "ComplexName")
    id_col       = next((c for c in df.columns if "id" in c.lower() and "complex" in c.lower()), "ComplexID")
    go_col       = next((c for c in df.columns if "go" in c.lower() and "id" in c.lower()), "GO ID")
    pubmed_col   = next((c for c in df.columns if "pubmed" in c.lower()), "PubMed ID")

    log.info(f"[corum] Loaded {len(df)} rows")

    # Filter 1 — organism scope
    valid_orgs = set(ORGANISM_TAXID.keys())
    df = df[df[organism_col].astype(str).str.strip().isin(valid_orgs)].copy()
    log.info(f"[corum] After organism filter: {len(df)} rows")

    # Parse UniProt IDs and apply Filter 2 (N), 3 (canonical)
    records = []
    known_sets = _load_known_uniprot_sets()

    for _, row in df.iterrows():
        raw_ids = str(row.get(subunit_col, ""))
        parts   = [p.strip() for p in raw_ids.replace(",", ";").split(";") if p.strip()]
        unique_ids = list(dict.fromkeys(parts))  # deduplicate, preserve order

        # Filter 3: all must be canonical
        if not all(_is_canonical_uniprot(u) for u in unique_ids):
            continue
        if len(unique_ids) < 2 or len(unique_ids) > 12:
            continue

        # Filter 4: exclude known GT complexes
        if frozenset(unique_ids) in known_sets:
            continue

        organism = str(row.get(organism_col, "")).strip()
        taxid    = ORGANISM_TAXID.get(organism, 9606)

        go_raw  = str(row.get(go_col, "")) if go_col in df.columns else ""
        go_ids  = [g.strip() for g in go_raw.split(";") if g.strip()]
        pm_raw  = str(row.get(pubmed_col, "")) if pubmed_col in df.columns else ""
        pm_ids  = [p.strip() for p in pm_raw.split(";") if p.strip()]

        records.append({
            "corum_id":    str(row.get(id_col, "")),
            "name":        str(row.get(name_col, "")).strip(),
            "organism":    organism,
            "taxid":       taxid,
            "n_subunits":  len(unique_ids),
            "uniprot_ids": json.dumps(unique_ids),
            "go_ids":      json.dumps(go_ids),
            "pubmed_ids":  json.dumps(pm_ids),
            "source":      "corum",
        })

    result = pd.DataFrame(records)
    log.info(f"[corum] After all filters: {len(result)} complexes")
    return result


# ── 1.3 / 1.4 PDB structure discovery ────────────────────────────────────────

def _fetch_pdb_ids_for_uniprot(uniprot_id: str, retries: int = 3) -> list[str]:
    """Query RCSB Search API for PDB entries containing a given UniProt ID."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    for attempt in range(retries):
        try:
            r = _SESSION.post(
                "https://search.rcsb.org/rcsbsearch/v2/query",
                json=query, timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            return [hit["identifier"] for hit in data.get("result_set", [])]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.debug(f"[corum] RCSB query failed for {uniprot_id}: {e}")
    return []


def _fetch_assembly_metadata(pdb_id: str) -> list[dict]:
    """Fetch PDBe assembly info for scoring."""
    cache_file = CACHE_DIR / "sifts" / f"assembly_{pdb_id.lower()}.json"
    if cache_file.exists():
        return json.load(open(cache_file))
    try:
        r = _SESSION.get(
            f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/assembly/{pdb_id.lower()}",
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json().get(pdb_id.lower(), [])
            json.dump(data, open(cache_file, "w"))
            return data
    except Exception:
        pass
    return []


def _fetch_entry_info(pdb_id: str) -> dict:
    """Fetch RCSB entry resolution and method."""
    cache_file = CACHE_DIR / "sifts" / f"entry_{pdb_id.lower()}.json"
    if cache_file.exists():
        return json.load(open(cache_file))
    try:
        r = _SESSION.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            raw_res = data.get("rcsb_entry_info", {}).get("resolution_combined", 99.0)
            if isinstance(raw_res, list):
                raw_res = raw_res[0] if raw_res else 99.0
            info = {
                "resolution": raw_res,
                "method":     data.get("exptl", [{}])[0].get("method", ""),
            }
            json.dump(info, open(cache_file, "w"))
            return info
    except Exception:
        pass
    return {"resolution": 99.0, "method": ""}


def _fetch_sifts_chain_uniprot(pdb_id: str) -> dict[str, str]:
    """Return {chain_id: uniprot_id} via SIFTS PDBe API."""
    cache_file = CACHE_DIR / "sifts" / f"{pdb_id.lower()}.json"
    if cache_file.exists():
        return json.load(open(cache_file))
    try:
        r = _SESSION.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
            timeout=10,
        )
        if r.status_code == 200:
            data  = r.json()
            entry = data.get(pdb_id.lower(), data.get(list(data.keys())[0], {})) if data else {}
            mapping: dict[str, str] = {}
            for uniprot_acc, info in entry.get("UniProt", {}).items():
                for m in info.get("mappings", []):
                    chain = m.get("chain_id") or m.get("struct_asym_id", "")
                    if chain:
                        mapping[chain] = uniprot_acc
            json.dump(mapping, open(cache_file, "w"))
            return mapping
    except Exception:
        pass
    json.dump({}, open(cache_file, "w"))
    return {}


def _score_assembly_method(assemblies: list[dict]) -> tuple[str, str, float]:
    """Return (assembly_id, method, method_score) for the best assembly."""
    best_id, best_method, best_score = "1", "", 0.0
    for a in assemblies:
        method = a.get("assembly_method", a.get("method", ""))
        score  = 1.5 if "author" in method.lower() else 0.5
        if score > best_score:
            best_score  = score
            best_id     = str(a.get("assembly_id", a.get("id", "1")))
            best_method = method
    return best_id, best_method, best_score


def find_best_pdb(uniprot_ids: list[str], taxid: int) -> Optional[dict]:
    """
    Find the best PDB entry that covers the maximum fraction of subunits.
    Returns None if no entry covers ≥ 60%.
    """
    # Collect PDB IDs hitting any subunit, count coverage per PDB
    pdb_hits: dict[str, set[str]] = {}
    for uid in uniprot_ids:
        pdbs = _fetch_pdb_ids_for_uniprot(uid)
        for pdb in pdbs:
            pdb_hits.setdefault(pdb, set()).add(uid)
        time.sleep(0.05)  # gentle rate limiting

    target = frozenset(uniprot_ids)
    best: dict = {}

    for pdb_id, covered in sorted(pdb_hits.items(), key=lambda x: -len(x[1])):
        frac = len(covered) / len(target)
        if frac < 0.6:
            continue

        info      = _fetch_entry_info(pdb_id)
        _raw_res  = info.get("resolution", 99.0)
        if isinstance(_raw_res, list):
            _raw_res = _raw_res[0] if _raw_res else 99.0
        res       = float(_raw_res or 99.0)
        assemblies= _fetch_assembly_metadata(pdb_id)
        asm_id, method, method_score = _score_assembly_method(assemblies)
        chain_map = _fetch_sifts_chain_uniprot(pdb_id)

        score = (frac * 2) + method_score + (1.0 if res < 3.0 else 0.0)

        if not best or score > best.get("score", 0):
            best = {
                "pdb_id":           pdb_id,
                "assembly_id":      asm_id,
                "coverage_fraction":round(frac, 3),
                "resolution":       res,
                "method":           method,
                "chain_to_uniprot": chain_map,
                "score":            round(score, 3),
            }

    return best if best else None


def build_pdb_map(filtered_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(filtered_df)
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        uid_list = json.loads(row["uniprot_ids"])
        if (i + 1) % 50 == 0:
            log.info(f"[corum] PDB discovery {i+1}/{total}")

        result = find_best_pdb(uid_list, int(row["taxid"]))
        if result:
            rows.append({
                "corum_id":         row["corum_id"],
                "pdb_id":           result["pdb_id"],
                "assembly_id":      result["assembly_id"],
                "coverage_fraction":result["coverage_fraction"],
                "resolution":       result["resolution"],
                "method":           result["method"],
                "chain_to_uniprot": json.dumps(result["chain_to_uniprot"]),
                "source":           "corum",
            })
        else:
            rows.append({
                "corum_id": row["corum_id"], "pdb_id": "no_structure",
                "assembly_id": "", "coverage_fraction": 0.0,
                "resolution": 99.0, "method": "", "chain_to_uniprot": "{}",
                "source": "corum",
            })
        time.sleep(0.1)

    return pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(force: bool = False):
    """Run Module 1 end-to-end. Idempotent — skips existing outputs."""
    filtered_path = INTERIM_DIR / "corum_filtered.tsv"
    pdb_map_path  = INTERIM_DIR / "corum_pdb_map.tsv"

    # Step 1: download
    raw_path = download_corum(force=force)

    # Step 2: parse and filter
    if filtered_path.exists() and not force:
        log.info(f"[corum] Filtered table already exists: {filtered_path}")
        filtered_df = pd.read_csv(filtered_path, sep="\t")
    else:
        filtered_df = parse_and_filter(raw_path)
        filtered_df.to_csv(filtered_path, sep="\t", index=False)
        log.info(f"[corum] Saved {len(filtered_df)} rows → {filtered_path}")

    # Step 3: PDB structure discovery
    if pdb_map_path.exists() and not force:
        log.info(f"[corum] PDB map already exists: {pdb_map_path}")
    else:
        log.info(f"[corum] Discovering PDB structures for {len(filtered_df)} complexes…")
        pdb_df = build_pdb_map(filtered_df)
        pdb_df.to_csv(pdb_map_path, sep="\t", index=False)
        log.info(f"[corum] Saved PDB map ({len(pdb_df)} rows) → {pdb_map_path}")

    return filtered_df, pdb_map_path


if __name__ == "__main__":
    run()
