"""
Three-tier GO Molecular Function annotation resolver.

Tier 1: UniProt REST API (experimental, most reliable)
Tier 2: SIFTS canonical UniProt ID → UniProt REST (fixes isoform-ID mismatches)
Tier 3: FABLE predicted logits via HTTP (augments, does not replace)

go_source values: "uniprot" | "sifts" | "fable" | "none"
"none" is a data quality failure — log and investigate.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import requests

log = logging.getLogger(__name__)

UNIPROT_TIMEOUT = 8   # seconds
SIFTS_TIMEOUT   = 8
FABLE_TIMEOUT   = 15


class GOResolver:
    """
    Stateful three-tier GO resolver with in-memory caching.
    Instantiate once and share across a server process.
    """

    def __init__(
        self,
        fable_url: str = "http://localhost:7860",
        go_cache: Optional[dict] = None,
        fable_cache: Optional[dict] = None,
        go_source_cache: Optional[dict] = None,
    ):
        self.fable_url = fable_url
        self._go_cache       = go_cache if go_cache is not None else {}
        self._fable_cache    = fable_cache if fable_cache is not None else {}
        self._go_src_cache   = go_source_cache if go_source_cache is not None else {}
        self._fable_available: Optional[bool] = None

    # ── Tier 1: UniProt REST ───────────────────────────────────────────────────

    def _fetch_uniprot(self, uniprot_id: str) -> list[dict]:
        if uniprot_id in self._go_cache:
            return self._go_cache[uniprot_id]
        try:
            r = requests.get(
                f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
                timeout=UNIPROT_TIMEOUT,
            )
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")
            data = r.json()
            terms = []
            for ref in data.get("dbReferences", []):
                if ref.get("type") == "GO":
                    props = {p["key"]: p["value"] for p in ref.get("properties", [])}
                    term = props.get("term", "")
                    if term.startswith("F:"):  # Molecular Function only
                        terms.append({"term": term, "source": "uniprot"})
            self._go_cache[uniprot_id] = terms
            return terms
        except Exception as e:
            log.debug(f"[go] UniProt fetch failed for {uniprot_id}: {e}")
            self._go_cache[uniprot_id] = []
            return []

    # ── Tier 2: SIFTS canonical UniProt ───────────────────────────────────────

    def _canonical_uniprot_sifts(self, pdb_id: str, uniprot_hint: str) -> str:
        try:
            r = requests.get(
                f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
                timeout=SIFTS_TIMEOUT,
            )
            if r.status_code != 200:
                return uniprot_hint
            data = r.json()
            entry = data.get(pdb_id.lower(), {}) or next(iter(data.values()), {})
            uniprot_section = entry.get("UniProt", {})
            hint_base = uniprot_hint.split("-")[0]
            for acc in uniprot_section:
                if acc.split("-")[0] == hint_base:
                    return acc
            if uniprot_section:
                return next(iter(uniprot_section))
        except Exception as e:
            log.debug(f"[go] SIFTS lookup failed for {pdb_id}/{uniprot_hint}: {e}")
        return uniprot_hint

    # ── Tier 3: FABLE predicted logits ────────────────────────────────────────

    def _probe_fable(self) -> bool:
        if self._fable_available is not None:
            return self._fable_available
        try:
            r = requests.get(f"{self.fable_url}/api/health", timeout=3)
            ok = r.status_code == 200 and r.json().get("model_loaded", False)
            self._fable_available = ok
            if ok:
                log.info(f"[go] FABLE reachable at {self.fable_url}")
            else:
                log.warning(f"[go] FABLE at {self.fable_url} responded but model not loaded")
            return ok
        except Exception:
            self._fable_available = False
            log.warning(f"[go] FABLE not reachable at {self.fable_url}")
            return False

    def _fetch_fable(self, sequence: str, uniprot_id: str = "") -> list[dict]:
        if not self._probe_fable():
            return []
        key = (hashlib.md5(sequence.encode()).hexdigest(), uniprot_id)
        if key in self._fable_cache:
            return self._fable_cache[key]
        try:
            payload = {"sequence": sequence, "uniprot_id": uniprot_id,
                       "taxon": "auto", "top_k": 20}
            r = requests.post(f"{self.fable_url}/api/go_terms",
                              json=payload, timeout=FABLE_TIMEOUT)
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")
            data = r.json()
            if "error" in data:
                raise ValueError(data["error"])
            terms = [
                {"term": f"F:{p['name']}", "source": "fable", "prob": p["prob"]}
                for p in data.get("predictions", [])
            ]
            self._fable_cache[key] = terms
            return terms
        except Exception as e:
            log.debug(f"[go] FABLE fetch failed: {e}")
            self._fable_cache[key] = []
            return []

    # ── Public interface ───────────────────────────────────────────────────────

    def resolve(
        self,
        sequence: str,
        uniprot_id: str,
        pdb_id: str = "",
    ) -> dict:
        """
        Resolve GO-MF annotations for a single chain.

        Returns:
            {
                "known":     list[dict]  — experimentally validated terms
                "predicted": list[dict]  — FABLE-predicted terms (deduplicated vs known)
                "all":       list[dict]  — known + predicted
                "go_source": str         — "uniprot" | "sifts" | "fable" | "none"
            }
        """
        # Tier 1
        known = self._fetch_uniprot(uniprot_id)
        go_source = "none"

        if not known and pdb_id:
            # Tier 2: try SIFTS canonical accession
            canonical = self._canonical_uniprot_sifts(pdb_id, uniprot_id)
            if canonical != uniprot_id:
                tier2 = self._fetch_uniprot(canonical)
                if tier2:
                    known = tier2
                    go_source = "sifts"
                    log.debug(f"[go] {uniprot_id} → canonical {canonical} via SIFTS "
                              f"({len(known)} terms)")

        if known and go_source == "none":
            go_source = "uniprot"

        # Tier 3: FABLE (augments, does not replace)
        predicted = self._fetch_fable(sequence, uniprot_id)
        if not known and predicted:
            go_source = "fable"

        if go_source == "none":
            log.warning(f"[go] No GO annotation found for {uniprot_id} (pdb={pdb_id})")

        self._go_src_cache[uniprot_id] = go_source

        known_terms = {t["term"] for t in known}
        deduped_predicted = [t for t in predicted if t["term"] not in known_terms]

        return {
            "known":     known,
            "predicted": deduped_predicted,
            "all":       known + deduped_predicted,
            "go_source": go_source,
        }
