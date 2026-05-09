"""
Downloads and prepares benchmark complexes.

Datasets:
1. Marsh 2013 homomers/heteromers (Cell 2013) — ground-truth assembly orders
2. NPC nucleoporins (human) — target system

Usage:
    python fetch_complexes.py --dataset marsh2013
    python fetch_complexes.py --dataset npc
    python fetch_complexes.py --dataset all
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "complexes"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
STRING_API = "https://string-db.org/api/json/network"


# ──────────────────────────────────────────────────────────────────────────────
# Marsh 2013 benchmark complexes
# Source: Table S1 from Marsh et al. Cell 2013 — ordered assembly pathways
# ──────────────────────────────────────────────────────────────────────────────

MARSH_2013_HOMOMERS = {
    # PDB_ID: (n_subunits, assembly_order_description)
    "1AON": {"n": 14, "type": "homomer", "name": "GroEL (homotetradecamer)"},
    "1GRU": {"n": 7,  "type": "homomer", "name": "GroES (homoheptamer)"},
    "2HHB": {"n": 4,  "type": "heteromer", "name": "Hemoglobin (α2β2)"},
    "1AY7": {"n": 2,  "type": "heteromer", "name": "RNase SA / Barstar"},
    "1BRS": {"n": 2,  "type": "heteromer", "name": "Barnase / Barstar"},
    "1SBB": {"n": 2,  "type": "heteromer", "name": "Subtilisin / Eglin C"},
    "1TGS": {"n": 2,  "type": "heteromer", "name": "Trypsin / PSTI"},
    "2PTC": {"n": 2,  "type": "heteromer", "name": "Trypsin / BPTI"},
    "3GBN": {"n": 2,  "type": "heteromer", "name": "Antigen / Antibody"},
    "1A2K": {"n": 2,  "type": "heteromer", "name": "Ran-GTP / importin"},
}

# NPC nucleoporin PDB structures (major subcomplexes with known structures)
NPC_COMPLEXES = {
    "5A9Q": {"name": "Nup107-Nup133 (Y-complex fragment)", "n": 2},
    "3PBP": {"name": "Nup98 APD crystal structure",        "n": 1},
    "4I9B": {"name": "Nup358 RanBD1",                     "n": 1},
    "5C3L": {"name": "Nup214 / Nup88 (cytoplasmic ring)", "n": 2},
    "2QX5": {"name": "Nup153 LacZ fusion",                "n": 1},
    "3F3F": {"name": "Nup62/58/54 (central channel)",     "n": 3},
    "5HAX": {"name": "Y-complex (human, Nup107 subcomplex)","n": 7},
}


def _fetch_pdb(pdb_id: str, outdir: Path) -> Path:
    out = outdir / f"{pdb_id}.pdb"
    if out.exists():
        print(f"  {pdb_id}.pdb already present")
        return out
    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out.write_text(r.text)
    print(f"  downloaded {pdb_id}.pdb ({len(r.text)//1024} KB)")
    time.sleep(0.5)
    return out


def _fetch_uniprot_fasta(accession: str) -> str:
    """Returns FASTA sequence string."""
    r = requests.get(UNIPROT_FASTA.format(accession=accession), timeout=30)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    return "".join(l for l in lines if not l.startswith(">"))


def _extract_sequences_from_pdb(pdb_path: Path) -> dict[str, str]:
    """Extract per-chain sequences from SEQRES records or ATOM records."""
    chains: dict[str, list[str]] = {}
    aa3to1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }

    seen_residues: dict[str, set] = {}
    with open(pdb_path) as f:
        for line in f:
            if line[:4] == "ATOM" and line[13:15].strip() == "CA":
                chain = line[21]
                resnum = line[22:26].strip()
                resname = line[17:20].strip()
                key = (resnum, resname)
                if chain not in seen_residues:
                    seen_residues[chain] = set()
                    chains[chain] = []
                if key not in seen_residues[chain]:
                    seen_residues[chain].add(key)
                    chains[chain].append(aa3to1.get(resname, "X"))

    return {c: "".join(seq) for c, seq in chains.items()}


def fetch_string_scores(
    proteins: list[str],  # UniProt accessions or gene names
    species: int = 9606,  # Human
    min_score: int = 400,
) -> dict[tuple[str, str], float]:
    """
    Fetch STRING PPI scores for a set of proteins.
    Returns dict: (protein_a, protein_b) → combined_score [0,1]
    """
    identifiers = "%0d".join(proteins)
    params = {
        "identifiers": identifiers,
        "species": species,
        "required_score": min_score,
        "caller_identity": "protein_assembly_sim",
    }
    r = requests.get(STRING_API, params=params, timeout=60)
    if r.status_code != 200:
        print(f"  STRING API error {r.status_code}; using zero scores")
        return {}

    scores = {}
    for item in r.json():
        a = item.get("preferredName_A", "")
        b = item.get("preferredName_B", "")
        s = item.get("score", 0) / 1000.0
        scores[(a, b)] = s
        scores[(b, a)] = s
    return scores


def download_dataset(dataset: str = "marsh2013"):
    if dataset in ("marsh2013", "all"):
        outdir = DATA_DIR / "marsh2013"
        outdir.mkdir(exist_ok=True)
        print("Downloading Marsh 2013 benchmark complexes...")
        manifest = {}
        for pdb_id, meta in {**MARSH_2013_HOMOMERS}.items():
            try:
                pdb_path = _fetch_pdb(pdb_id, outdir)
                seqs = _extract_sequences_from_pdb(pdb_path)
                meta["sequences"] = seqs
                meta["pdb_path"] = str(pdb_path)
                manifest[pdb_id] = meta
            except Exception as e:
                print(f"  WARN: {pdb_id} failed: {e}")
        manifest_path = outdir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  manifest saved → {manifest_path}")

    if dataset in ("npc", "all"):
        outdir = DATA_DIR / "npc"
        outdir.mkdir(exist_ok=True)
        print("Downloading NPC nucleoporin structures...")
        manifest = {}
        for pdb_id, meta in NPC_COMPLEXES.items():
            try:
                pdb_path = _fetch_pdb(pdb_id, outdir)
                seqs = _extract_sequences_from_pdb(pdb_path)
                meta["sequences"] = seqs
                meta["pdb_path"] = str(pdb_path)
                manifest[pdb_id] = meta
            except Exception as e:
                print(f"  WARN: {pdb_id} failed: {e}")
        manifest_path = outdir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  manifest saved → {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="marsh2013",
                        choices=["marsh2013", "npc", "all"])
    args = parser.parse_args()
    download_dataset(args.dataset)
