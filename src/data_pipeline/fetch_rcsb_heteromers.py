"""
Module 3 — RCSB PDB search for high-quality heteromeric complexes.

Outputs:
  data/interim/rcsb_candidates.tsv   filtered, deduplicated heteromer candidates
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.data_pipeline import CACHE_DIR, INTERIM_DIR
from src.data_pipeline.fetch_corum import (
    _fetch_entry_info,
    _fetch_sifts_chain_uniprot,
    _fetch_assembly_metadata,
    _is_canonical_uniprot,
    _load_known_uniprot_sets,
    _score_assembly_method,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "PRISM-DataPipeline/1.0"})


# ── 3.1 RCSB Search query ────────────────────────────────────────────────────

SEARCH_QUERY = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_assembly_info.polymer_entity_count_protein",
                    "operator": "greater_or_equal",
                    "value": 2,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_assembly_info.polymer_entity_count_protein",
                    "operator": "less_or_equal",
                    "value": 12,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": 3.5,
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                    "operator": "exact_match",
                    "value": "Protein (only)",
                },
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.deposit_date",
                    "operator": "less",
                    "value": "2024-01-01",
                },
            },
        ],
    },
    "return_type": "entry",
    "request_options": {"return_all_hits": True},
}


def fetch_rcsb_entry_list(retries: int = 3) -> list[str]:
    """Execute the RCSB search query. Returns list of PDB IDs."""
    for attempt in range(retries):
        try:
            r = _SESSION.post(RCSB_SEARCH_URL, json=SEARCH_QUERY, timeout=60)
            r.raise_for_status()
            data   = r.json()
            result = [hit["identifier"] for hit in data.get("result_set", [])]
            log.info(f"[rcsb] Search returned {len(result)} PDB entries")
            return result
        except Exception as e:
            log.warning(f"[rcsb] Search attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return []


# ── 3.2 Deduplication against CORUM + Complex Portal ────────────────────────

def _load_existing_uniprot_sets() -> set[frozenset]:
    """Load all UniProt frozensets from CORUM, Complex Portal, and GT files."""
    sets = _load_known_uniprot_sets()  # GT complexes

    for tsv_path in [
        INTERIM_DIR / "corum_filtered.tsv",
        INTERIM_DIR / "complex_portal_filtered.tsv",
    ]:
        if not tsv_path.exists():
            continue
        df = pd.read_csv(tsv_path, sep="\t")
        for uid_json in df.get("uniprot_ids", []):
            try:
                sets.add(frozenset(json.loads(uid_json)))
            except Exception:
                pass
    return sets


def _process_pdb_entry(pdb_id: str, existing_sets: set[frozenset]) -> Optional[dict]:
    """
    Fetch SIFTS mapping for one PDB entry. Returns filtered record or None.
    Thread-safe — uses only local state and cached files.
    """
    chain_map = _fetch_sifts_chain_uniprot(pdb_id)
    if not chain_map:
        return None

    all_uniprots  = list(chain_map.values())
    unique_uniprots = list(dict.fromkeys(all_uniprots))

    if len(unique_uniprots) < 2 or len(unique_uniprots) > 12:
        return None
    if not all(_is_canonical_uniprot(u) for u in unique_uniprots):
        return None
    if frozenset(unique_uniprots) in existing_sets:
        return None

    return {
        "pdb_id":          pdb_id,
        "chain_to_uniprot":json.dumps(chain_map),
        "uniprot_ids":     json.dumps(unique_uniprots),
        "n_subunits":      len(all_uniprots),
        "n_unique":        len(unique_uniprots),
    }


def deduplicate_entries(
    pdb_ids: list[str],
    existing_sets: set[frozenset],
    max_workers: int = 8,
    batch_size: int = 100,
) -> list[dict]:
    """Parallel SIFTS fetch with deduplication. Returns list of candidate dicts."""
    candidates: list[dict] = []
    seen_sets: set[frozenset] = set(existing_sets)

    total = len(pdb_ids)
    log.info(f"[rcsb] Deduplicating {total} entries with {max_workers} workers")

    for batch_start in range(0, total, batch_size):
        batch = pdb_ids[batch_start: batch_start + batch_size]
        if (batch_start // batch_size) % 10 == 0:
            log.info(f"[rcsb]   batch {batch_start}/{total}, accepted so far: {len(candidates)}")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_pdb_entry, pdb, seen_sets): pdb for pdb in batch}
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception:
                    result = None
                if result is None:
                    continue
                uid_set = frozenset(json.loads(result["uniprot_ids"]))
                if uid_set not in seen_sets:
                    seen_sets.add(uid_set)
                    candidates.append(result)

        time.sleep(0.2)  # gentle batch-level rate limiting

    log.info(f"[rcsb] {len(candidates)} unique candidates after dedup")
    return candidates


# ── 3.3 Quality filters ───────────────────────────────────────────────────────

def _fetch_assembly_form(pdb_id: str) -> str:
    """Return assembly form string (e.g. 'heterodimer') from PDBe."""
    assemblies = _fetch_assembly_metadata(pdb_id)
    if not assemblies:
        return ""
    a = assemblies[0] if isinstance(assemblies, list) else assemblies
    return str(a.get("form", a.get("assembly_form", ""))).lower()


VALID_ASSEMBLY_FORMS = {
    "dimer", "trimer", "tetramer", "pentamer", "hexamer",
    "heptamer", "octamer", "nonamer", "decamer",
    "undecamer", "dodecamer", "heteromer", "heterodimer",
    "heterotrimer", "heterotetramer", "heteropentamer",
    "homomer", "homodimer", "homotrimer",
}


def apply_quality_filters(candidates: list[dict]) -> list[dict]:
    """
    Filter A — biological assembly confidence (PISA form).
    Filter B — homomer ring exclusion.
    Filter C — sequence redundancy (requires embeddings; skips if unavailable).
    """
    filtered = []
    total    = len(candidates)

    for i, c in enumerate(candidates):
        if (i + 1) % 200 == 0:
            log.info(f"[rcsb] Quality filter {i+1}/{total}")

        pdb_id     = c["pdb_id"]
        n_unique   = int(c["n_unique"])
        n_subunits = int(c["n_subunits"])

        # Filter A: biological assembly form
        form       = _fetch_assembly_form(pdb_id)
        assemblies = _fetch_assembly_metadata(pdb_id)
        preferred  = any(a.get("preferred", False) for a in (assemblies if isinstance(assemblies, list) else []))
        if form not in VALID_ASSEMBLY_FORMS and not preferred and form != "":
            continue

        # Filter B: large ring homomers
        is_ring    = n_unique == 1
        if is_ring and n_subunits > 5:
            continue  # skip 7-mer+ rings

        # Annotate taxid from RCSB entry info
        entry_info = _fetch_entry_info(pdb_id)
        _raw_res   = entry_info.get("resolution", 99.0)
        if isinstance(_raw_res, list):
            _raw_res = _raw_res[0] if _raw_res else 99.0
        res        = float(_raw_res or 99.0)

        assemblies_list = assemblies if isinstance(assemblies, list) else []
        _, method, _    = _score_assembly_method(assemblies_list)

        # Infer taxid from chain UniProt IDs via a simple lookup
        # (full taxid lookup would require additional API calls; use 0 as unknown)
        taxid = _infer_taxid(json.loads(c.get("chain_to_uniprot", "{}")))

        filtered.append({
            "pdb_id":          pdb_id,
            "assembly_id":     "1",
            "uniprot_ids":     c["uniprot_ids"],
            "chain_to_uniprot":c.get("chain_to_uniprot", "{}"),
            "n_subunits":      n_subunits,
            "n_unique":        n_unique,
            "is_ring":         is_ring,
            "is_homomer":      n_unique == 1,
            "resolution":      res,
            "method":          method,
            "taxid":           taxid,
            "source":          "rcsb_search",
        })
        time.sleep(0.05)

    log.info(f"[rcsb] {len(filtered)} passed quality filters (from {total})")
    return filtered


_TAXID_HINTS: dict[str, int] = {
    "P": 9606, "Q": 9606, "O": 9606,   # human dominant in P/Q/O space
}

def _infer_taxid(chain_to_uniprot: dict) -> int:
    """Heuristic: use the first UniProt prefix to guess taxid."""
    for _, uid in chain_to_uniprot.items():
        if uid and uid[0] in _TAXID_HINTS:
            return _TAXID_HINTS[uid[0]]
    return 0


# ── Filter C: sequence redundancy (best-effort) ───────────────────────────────

def _cluster_by_embedding_similarity(
    candidates: list[dict],
    threshold: float = 0.97,
) -> list[dict]:
    """
    Within clusters of candidates with cosine sim > threshold on mean UniProt
    embedding, keep only the highest-resolution representative.
    Skips clustering if embeddings cache is unavailable.
    """
    cache_pt = Path(__file__).parent.parent.parent / "embeddings_cache.pt"
    if not cache_pt.exists():
        log.info("[rcsb] embeddings_cache.pt not found — skipping redundancy filter")
        return candidates

    try:
        import torch
        import numpy as np

        emb_cache = torch.load(str(cache_pt), map_location="cpu")
        import hashlib

        def mean_emb(uid_list):
            vecs = []
            for uid in uid_list:
                # Use UniProt ID as a lookup key proxy (not seq hash, but
                # close enough for clustering — sequences are stable per UniProt)
                key = hashlib.md5(uid.encode()).hexdigest()
                if key in emb_cache:
                    vecs.append(emb_cache[key].numpy())
            if not vecs:
                return np.zeros(640)
            return np.mean(vecs, axis=0)

        embs = []
        for c in candidates:
            uid_list = json.loads(c["uniprot_ids"])
            embs.append(mean_emb(uid_list))

        embs_np = np.array(embs)
        norms   = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-8
        embs_n  = embs_np / norms

        used       = [False] * len(candidates)
        reps       = []
        resolutions= [c.get("resolution", 99.0) for c in candidates]

        for i in range(len(candidates)):
            if used[i]:
                continue
            cluster = [i]
            for j in range(i + 1, len(candidates)):
                if not used[j]:
                    sim = float(np.dot(embs_n[i], embs_n[j]))
                    if sim > threshold:
                        cluster.append(j)
            best_in_cluster = min(cluster, key=lambda k: resolutions[k])
            reps.append(candidates[best_in_cluster])
            for k in cluster:
                used[k] = True

        log.info(f"[rcsb] Redundancy filter: {len(candidates)} → {len(reps)} representatives")
        return reps

    except Exception as e:
        log.warning(f"[rcsb] Redundancy filter failed: {e} — keeping all candidates")
        return candidates


# ── Main entry point ──────────────────────────────────────────────────────────

def run(force: bool = False, max_candidates: Optional[int] = None):
    out_path = INTERIM_DIR / "rcsb_candidates.tsv"
    if out_path.exists() and not force:
        log.info(f"[rcsb] Already exists: {out_path} — skipping")
        return pd.read_csv(out_path, sep="\t")

    # Step 1: search
    pdb_ids = fetch_rcsb_entry_list()
    if max_candidates:
        pdb_ids = pdb_ids[:max_candidates]

    # Step 2: load existing sets and deduplicate
    existing_sets = _load_existing_uniprot_sets()
    candidates    = deduplicate_entries(pdb_ids, existing_sets)

    # Step 3: quality filters
    candidates = apply_quality_filters(candidates)

    # Filter C: sequence redundancy
    candidates = _cluster_by_embedding_similarity(candidates)

    df = pd.DataFrame(candidates)
    df.to_csv(out_path, sep="\t", index=False)
    log.info(f"[rcsb] Saved {len(df)} candidates → {out_path}")
    return df


if __name__ == "__main__":
    # Limit to 5000 PDB IDs for initial testing; remove limit for full run
    run(max_candidates=5000)
