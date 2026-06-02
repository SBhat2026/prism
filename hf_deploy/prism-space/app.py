"""
FastAPI server for PRISM — Protein assembly simulation viewer.
HuggingFace Space deployment (CPU, no MPS, no webbrowser).
"""

import asyncio
import hashlib
import json
import math
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.stats import spearmanr as scipy_spearmanr

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mutation.mutate import apply_mutation, compute_kl_per_step, detect_pathway_redirect
from src.models.scorer_gnn import (
    AssemblyScorer, build_edge_features as gnn_build_edge, build_copy_indices,
)
from src.features.chirality_features import compute_chirality_features

# ── Constants ─────────────────────────────────────────────────────────────────

W_SASA = 0.40
W_ESM  = 0.35
W_GO   = 0.15
W_PPI  = 0.10

FABLE_URL = os.environ.get("FABLE_URL", "http://localhost:7860")

STATIC_DIR = Path(__file__).parent / "static"

RCSB_PDB_URL   = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
SIFTS_URL      = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
UNIPROT_JSON   = "https://rest.uniprot.org/uniprotkb/{uid}.json"

_AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

# ── Complex definitions ────────────────────────────────────────────────────────

COMPLEXES = {
    "2HHB": {
        "name": "Hemoglobin",
        "description": "Human hemoglobin α₂β₂ tetramer — oxygen transport",
        "type": "heteromer",
        "is_ring": False,
        "taxid": 9606,
        "subunits": [
            {"chain": "A", "uniprot": "P69905", "name": "Hb-α", "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"},
            {"chain": "B", "uniprot": "P68871", "name": "Hb-β", "sequence": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"},
            {"chain": "C", "uniprot": "P69905", "name": "Hb-α2", "sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"},
            {"chain": "D", "uniprot": "P68871", "name": "Hb-β2", "sequence": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"},
        ],
        "gt_order": [0, 2, 1, 3],
    },
    "1BRS": {
        "name": "Barnase / Barstar",
        "description": "Barnase ribonuclease + Barstar inhibitor — inhibition mechanism",
        "type": "heteromer",
        "is_ring": False,
        "taxid": 1423,
        "subunits": [
            {"chain": "A", "uniprot": "P00648", "name": "Barnase-1", "sequence": "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"},
            {"chain": "B", "uniprot": "P11540", "name": "Barstar-1", "sequence": "MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITI"},
            {"chain": "C", "uniprot": "P00648", "name": "Barnase-2", "sequence": "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"},
            {"chain": "D", "uniprot": "P11540", "name": "Barstar-2", "sequence": "MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITI"},
            {"chain": "E", "uniprot": "P00648", "name": "Barnase-3", "sequence": "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"},
            {"chain": "F", "uniprot": "P11540", "name": "Barstar-3", "sequence": "MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITI"},
        ],
        "gt_order": [0, 3, 1, 4, 2, 5],
    },
    "2PTC": {
        "name": "Trypsin / BPTI",
        "description": "Bovine trypsin + BPTI inhibitor — protease inhibition",
        "type": "heteromer",
        "is_ring": False,
        "taxid": 9913,
        "subunits": [
            {"chain": "E", "uniprot": "P00760", "name": "Trypsin", "sequence": "IVGGYTCAANSIPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLSSPAVINSRIVNGEEAVPGSWPWQVSLQDKTGFHFCGGSLINENWVVTAAHCVTTGVTTSDVVVAGEFDQGSDEENIQVLKIAKVFKNPKFSILTVNNDITLLKLSTAASFSQTVSAVCLPSASDDFAAGTTCVTTGWGLTRYTNANTPDRLQQASLPLLSNTNCKKYWGTKIKDAMICAGASGVSSCMGDSGGPLVCKKNGAWTLVGIVSWGSSTCSTSTPGVYARVTALVNWVQQTLAAN"},
            {"chain": "I", "uniprot": "P00974", "name": "BPTI", "sequence": "RPDDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"},
        ],
        "gt_order": [0, 1],
    },
    "1AON": {
        "name": "GroEL",
        "description": "E. coli GroEL chaperonin — 7-mer ring, protein folding",
        "type": "homomer",
        "is_ring": True,
        "taxid": 83333,
        "subunits": [
            {"chain": c, "uniprot": "P0A6F5", "name": f"GroEL-{c}", "sequence": "MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMGAQMVKEVASKANDAAGDGTTTATVLAQAIITEGLKAVAAGMNPMDLKRGIDKAVTAAVEELKALSVPCSDSKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLQDELDVVEGMQFDRGYLSPYFINKPETGAVELESPFILLADKKISNIREMLPVLEAVAKAGKPLLIIAEDVEGEALATLVVNTMRGIVKVAAVKAPGFGDRRKAMLQDIATLTGGTVISEEIGMELEKATLEDLGQAKRVVINKDTTTIIDGVGEEAAIQGRVAQIRQQIEEATSDYDREKLQERLAKLAGGVAVIKVGAATEVEMKEKKARVEDALHATRAAVEEGVVAGGGVALIRVASKLADLRGQNEDQNVGIKVALRAMEAPLRQIVLNCGEEPSVVANTVKGGDGNYGYNAATEEYGNMIDMGILDPTKVTRSALQYAASVAGLMITTECMVTDLPKNDAADLGAAGGMGGMGGMGGMM"} for c in ["A","B","C","D","E","F","G"]
        ],
        "gt_order": [0, 1, 2, 3, 4, 5, 6],
    },
    "1GRU": {
        "name": "GroES",
        "description": "E. coli GroES co-chaperonin — 7-mer ring, caps GroEL",
        "type": "homomer",
        "is_ring": True,
        "taxid": 83333,
        "subunits": [
            {"chain": c, "uniprot": "P0A6F9", "name": f"GroES-{c}", "sequence": "MSKGQEIKDLGKKLRDAVVAALEEHGASMADLKNKLLGGSGKIMVEEGKSVSEVILEDAGIQNIVKDMKAKVEDALREKVSHDDIQFLDFKLKAENGKGEKIEMRNMILPKEIVSGQDLDQVTDLSKKQ"} for c in ["A","B","C","D","E","F","G"]
        ],
        "gt_order": [0, 1, 2, 3, 4, 5, 6],
    },
    "1AY7": {
        "name": "RNase SA / Barstar",
        "description": "Streptomyces RNase SA with Barstar inhibitor",
        "type": "heteromer",
        "is_ring": False,
        "taxid": 1423,
        "subunits": [
            {"chain": "A", "uniprot": "P00655", "name": "RNase-SA", "sequence": "STSSATNNRPQLYKGQQWLEKPRPVFAVDPDSVTLIPASLAAQRLNKIDKSLKDAKYYIFKTLKENPNQNSMQVYEKMQGIQGLRPPNFSTPEYLRAFCETKKGVQVQVNLPTQFRGKDLGPYDGSSVKGNSMFKQLVKRPNSTQHKYAAGVKKELRKRNEFLNYMRFEGLYFDRTRSKEEPF"},
            {"chain": "B", "uniprot": "P11540", "name": "Barstar", "sequence": "MKKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITI"},
        ],
        "gt_order": [0, 1],
    },
    "1A2K": {
        "name": "Ran / Importin-β",
        "description": "Nuclear transport complex — 5-mer",
        "type": "heteromer",
        "is_ring": False,
        "taxid": 9606,
        "subunits": [
            {"chain": "A", "uniprot": "P62826", "name": "Ran-1", "sequence": "MAAQGEPQVVFANKSTGHELRKLQKERDIKFLNEQINLMKMDAMAGDKLADVYQEKTFEIPNLQEPGASGDISDDEDDEDGYEDDKYNESEEIYNKKASDILKNVKGIQIKEEPGFAPFEQSKKADAQKLGLSPERRFFLSIAPKEIKQVEDDFLQQTFESLQKEELKTRLHQDLPNLSEVPQKEEEYDVTLDLWHIVNEKDNAQFVLHPYIVNVEPVEDDFKLASVLQITPQNNLDDKMVEKLFQLCNQSQVQFKDVDVKEAGFQKLGLEVNKDMTSVKDLFRRTFADAMNEGLNRLQKTFNQK"},
            {"chain": "B", "uniprot": "O00505", "name": "Importin-β-1", "sequence": "MSSDSECQEIVSRLEEQFAKLKKEQEQEAEKEAQKELEALQKAQEELARKENEELKRANEKQKQLQNRLNQLEKARDQLEKELQNNGEENVAHNLTKTLKLAQKEAEQMLSIQREIEQQLAKDCESNNEEGIPESSFKSLEELQKELEQFNECVDQQLQKFKDDDDSFDIDDLDQKDMTFIAELEQLDPLQELQKEDDQLDNLFEKAQKERQKDLMQAQKNQEALEKLHEAMQTEYTELVKKLQELREELDQLKQEQDQLVQELEELKAQAMDKLQEKSEELQDLQAQAQELSNKIQELDQAMKFQQDELDLQEALTQKVQEEQKQLNQEQLKDQNFQAQRQKDLLQERQQFQRDYQQLRKDCERQREELEELRKKLEAQTTKLKEELQRELQQLDELKEKNEELRTKLQAQHQELQQDLQKRLQQELEQLQAENKELEMQLQRMQEEEQKEQQQTLKDLKEHVQEIKDDLLNLQKDREDLEKLQNQIQELLKRSQEELQKDMEQAKQLMNQLKEQIREELDQLQKEAEELVQNKEELQMELEKAKEAQAQQFQ"},
            {"chain": "C", "uniprot": "P62826", "name": "Ran-2", "sequence": "MAAQGEPQVVFANKSTGHELRKLQKERDIKFLNEQINLMKMDAMAGDKLADVYQEKTFEIPNLQEPGASGDISDDEDDEDGYEDDKYNESEEIYNKKASDILKNVKGIQIKEEPGFAPFEQSKKADAQKLGLSPERRFFLSIAPKEIKQVEDDFLQQTFESLQKEELKTRLHQDLPNLSEVPQKEEEYDVTLDLWHIVNEKDNAQFVLHPYIVNVEPVEDDFKLASVLQITPQNNLDDKMVEKLFQLCNQSQVQFKDVDVKEAGFQKLGLEVNKDMTSVKDLFRRTFADAMNEGLNRLQKTFNQK"},
            {"chain": "D", "uniprot": "O00505", "name": "Importin-β-2", "sequence": "MSSDSECQEIVSRLEEQFAKLKKEQEQEAEKEAQKELEALQKAQEELARKENEELKRANEKQKQLQNRLNQLEKARDQLEKELQNNGEENVAHNLTKTLKLAQKEAEQMLSIQREIEQQLAKDCESNNEEGIPESSFKSLEELQKELEQFNECVDQQLQKFKDDDDSFDIDDLDQKDMTFIAELEQLDPLQELQKEDDQLDNLFEKAQKERQKDLMQAQKNQEALEKLHEAMQTEYTELVKKLQELREELDQLKQEQDQLVQELEELKAQAMDKLQEKSEELQDLQAQAQELSNKIQELDQAMKFQQDELDLQEALTQKVQEEQKQLNQEQLKDQNFQAQRQKDLLQERQQFQRDYQQLRKDCERQREELEELRKKLEAQTTKLKEELQRELQQLDELKEKNEELRTKLQAQHQELQQDLQKRLQQELEQLQAENKELEMQLQRMQEEEQKEQQQTLKDLKEHVQEIKDDLLNLQKDREDLEKLQNQIQELLKRSQEELQKDMEQAKQLMNQLKEQIREELDQLQKEAEELVQNKEELQMELEKAKEAQAQQFQ"},
            {"chain": "E", "uniprot": "P62826", "name": "Ran-3", "sequence": "MAAQGEPQVVFANKSTGHELRKLQKERDIKFLNEQINLMKMDAMAGDKLADVYQEKTFEIPNLQEPGASGDISDDEDDEDGYEDDKYNESEEIYNKKASDILKNVKGIQIKEEPGFAPFEQSKKADAQKLGLSPERRFFLSIAPKEIKQVEDDFLQQTFESLQKEELKTRLHQDLPNLSEVPQKEEEYDVTLDLWHIVNEKDNAQFVLHPYIVNVEPVEDDFKLASVLQITPQNNLDDKMVEKLFQLCNQSQVQFKDVDVKEAGFQKLGLEVNKDMTSVKDLFRRTFADAMNEGLNRLQKTFNQK"},
        ],
        "gt_order": [1, 0, 3, 2, 4],
    },
    "3GBN": {
        "name": "Hemagglutinin / Antibody",
        "description": "Influenza HA with neutralizing antibody",
        "type": "heteromer",
        "is_ring": False,
        "taxid": 11520,
        "subunits": [
            {"chain": "A", "uniprot": "P03437", "name": "Hemagglutinin", "sequence": "MKTIIALSYIFCLVFAHKDNAIWYQNIDGTEGSGCQKLPVNPDTIIKIEHPNGIVKEKLAKMLDKSIDGAPFEKLMEWLKTRPILSPLTKGILGFVFTLTVPSERGLQHLNDSGTLYIVKSGYVNKQHILNLKNLCESVEHSNLKFFSNVPWSQNLNLRTYVQASKGNTYSCNPNFNGIIKLNQVNSSIVNLTNLNENVPNHSQNLNPEQVQNWSGDTPNISSQHITQSITQ"},
            {"chain": "B", "uniprot": "P01857", "name": "IgG1-Heavy", "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDYYGSNSFDYWGQGTLVTVSSDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"},
            {"chain": "C", "uniprot": "P01834", "name": "IgG1-Light", "sequence": "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"},
        ],
        "gt_order": [0, 2, 1],
    },
    "1DVQ": {
        "name": "Transthyretin",
        "description": "Human transthyretin — 4-mer ring, thyroid hormone transport",
        "type": "homomer",
        "is_ring": True,
        "taxid": 9606,
        "subunits": [
            {"chain": c, "uniprot": "P02766", "name": f"TTR-{c}", "sequence": "MASHRLLLLCLAGLVFVSEARAPQTAPGYSGQSPRPPSYPGSSPYSQPQAPAGQAKPPPSTSTSAPSPELSPVQPPTEAPTYAKPPPSTAPHTTAAACAPQKPSTAPATSAPGPAKPPPAAPPSTEAQAKPPQPSTPAAPTATKLPQPSPPATAQPPTEAKPPPSTTQPPPSTAPQSAAPAPPKPKEAPKKP"} for c in ["A","B","C","D"]
        ],
        "gt_order": [0, 1, 2, 3],
    },
    "1AXC": {
        "name": "PCNA",
        "description": "Human PCNA sliding clamp — 3-mer ring, DNA replication",
        "type": "homomer",
        "is_ring": True,
        "taxid": 9606,
        "subunits": [
            {"chain": c, "uniprot": "P12004", "name": f"PCNA-{c}", "sequence": "MFEARLVQGSILKKVLEALKDLINEACWDISSSGVNLQSMDSSHVSLVQLTLRSEGFDTYRCDRNLAMGDNLTSRVNAYACMDLSQFEKLTPQNLFIKPQKMDDYHASSGSTQRSAGVAASSETPEDISNTQDSSATSQLAQKDEEEAVRAQKDKAVQKELQNLKAELVSQKPEVKNLKQKADQDKLAQIAELERAMIQTNLEEFQNKASAALKLSALKNQALDSYKQLEQKLKEDLEAIKKQMDPEKIIQRLHQIIQKYRHTAKEAVEQSIQELEAELKTLKQQLNDLMLQKLEQRVLEKLEQENQKLVVKISQEETLQKEDILLTSVKDDLEDLQNKEDLSQEKLEQLVNELQSELQKAKEELNEALEQFQAL"} for c in ["A","B","C"]
        ],
        "gt_order": [0, 1, 2],
    },
}


# ── State ─────────────────────────────────────────────────────────────────────

_state = {
    "model": None,
    "tokenizer": None,
    "embeddings": {},
    "simulations": {},
    "sasa_matrices": {},   # pdb_id → (N,N) float32 BSA matrix (normalised 0-1)
    "go_cache": {},
    "fable_cache": {},
    "ppi_cache": {},
    "fable_available": None,
    "entropy_cache": {},    # pdb_id -> float (Shannon entropy of pathway distribution)
    "go_source_cache": {},  # uniprot_id -> "uniprot"|"sifts"|"fable"|"none"
    "taxid_cache": {},      # uniprot_id -> int taxon ID (0 = unknown)
    "ready": False,
    # GNN scorer (v5)
    "gnn_model":    None,   # AssemblyScorer instance
    "gnn_tensors":  {},     # pdb_id → (node_feats_t, edge_feats_t, adj_t, copy_indices_t)
    "gnn_version":  None,   # str, e.g. "v5"
}

BASE_DIR            = Path(__file__).parent
_TAXID_DISK_CACHE   = BASE_DIR.parent / "data" / "cache" / "taxids"
_MUTATION_CACHE_DIR = BASE_DIR.parent / "data" / "mutation_cache"
_PDB_CACHE_DIR      = BASE_DIR.parent / "data" / "cache" / "pdb"
_CHIRALITY_CACHE    = BASE_DIR.parent / "data" / "chirality_cache"
_PLDDT_CACHE_DIR    = BASE_DIR.parent / "data" / "plddt_cache"
_GEOM_CACHE_DIR     = BASE_DIR.parent / "data" / "feature_cache" / "af_pdbs"
# HF Space: load checkpoint from HF Hub (Sbhat2026/PRISM-model repo)
def _resolve_checkpoint() -> Path:
    local = BASE_DIR / "prism_v1.pt"
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id="Sbhat2026/PRISM", filename="prism_v1.pt",
                            repo_type="model", cache_dir=str(BASE_DIR / "hf_cache"))
        return Path(p)
    except Exception as e:
        print(f"[gnn] HF Hub download failed: {e}")
    return local  # will fail gracefully in _load_gnn

_GNN_CHECKPOINT = _resolve_checkpoint()
_PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── ESM utilities ─────────────────────────────────────────────────────────────

def _load_esm():
    try:
        import esm as esm_lib
        model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
        model.eval()
        _state["model"] = model
        _state["tokenizer"] = alphabet
        print("[startup] ESM-2 150M loaded")
        return True
    except Exception as e:
        print(f"[startup] ESM load failed: {e}")
        return False


_embed_cache: dict[str, np.ndarray] = {}
MAX_EMBED_LEN = 1022


def _embed_sequence(sequence: str) -> np.ndarray:
    if len(sequence) > MAX_EMBED_LEN:
        print(f"  [embed] truncating {len(sequence)} aa → {MAX_EMBED_LEN} aa for embedding")
    seq_trunc = sequence[:MAX_EMBED_LEN]
    cache_key = hashlib.md5(seq_trunc.encode()).hexdigest()
    if cache_key in _embed_cache:
        return _embed_cache[cache_key]

    model = _state["model"]
    alphabet = _state["tokenizer"]
    if model is None:
        emb = np.random.randn(640).astype(np.float32)
        _embed_cache[cache_key] = emb
        return emb

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", seq_trunc)])
    with torch.no_grad():
        results = model(tokens, repr_layers=[30])
    emb = results["representations"][30][0, 1:-1].mean(0).cpu().numpy().astype(np.float32)
    _embed_cache[cache_key] = emb
    return emb


# ── GO / PPI fetching ─────────────────────────────────────────────────────────

def _fetch_go_terms_uniprot(uniprot_id: str) -> list[dict]:
    """Tier 1: UniProt REST API GO-MF terms (timeout 8s)."""
    if uniprot_id in _state["go_cache"]:
        return _state["go_cache"][uniprot_id]
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        data = r.json()
        terms = []
        for ref in data.get("dbReferences", []):
            if ref.get("type") == "GO":
                props = {p["key"]: p["value"] for p in ref.get("properties", [])}
                term = props.get("term", "")
                if term:
                    terms.append({"term": term, "source": "uniprot"})
        _state["go_cache"][uniprot_id] = terms
        return terms
    except Exception:
        _state["go_cache"][uniprot_id] = []
        return []


def _fetch_canonical_uniprot_sifts(pdb_id: str, uniprot_hint: str) -> str:
    """Tier 2: SIFTS PDBe API to resolve canonical UniProt accession."""
    try:
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return uniprot_hint
        data = r.json()
        pdb_key = pdb_id.lower()
        entry = data.get(pdb_key, data.get(list(data.keys())[0], {})) if data else {}
        uniprot_section = entry.get("UniProt", {})
        hint_base = uniprot_hint.split("-")[0]
        for accession in uniprot_section:
            if accession.split("-")[0] == hint_base:
                return accession
        if uniprot_section:
            return next(iter(uniprot_section))
        return uniprot_hint
    except Exception:
        return uniprot_hint


def _probe_fable() -> bool:
    if _state["fable_available"] is not None:
        return _state["fable_available"]
    try:
        r = requests.get(f"{FABLE_URL}/api/health", timeout=3)
        ok = r.status_code == 200 and r.json().get("model_loaded", False)
        _state["fable_available"] = ok
        if ok:
            print(f"[go] FABLE reachable at {FABLE_URL}")
        else:
            print(f"[go] FABLE at {FABLE_URL} responded but model not loaded — using UniProt only")
        return ok
    except Exception:
        _state["fable_available"] = False
        print(f"[go] FABLE not reachable at {FABLE_URL} — using UniProt only")
        return False


def _fetch_go_terms_fable(sequence: str, uniprot_id: str = "") -> list[dict]:
    if not _probe_fable():
        return []
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    cache_key = (seq_hash, uniprot_id)
    if cache_key in _state["fable_cache"]:
        return _state["fable_cache"][cache_key]
    try:
        payload = {"sequence": sequence, "uniprot_id": uniprot_id, "taxon": "auto", "top_k": 20}
        r = requests.post(f"{FABLE_URL}/api/go_terms", json=payload, timeout=15)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        data = r.json()
        if "error" in data:
            raise ValueError(data["error"])
        terms = [
            {"term": f"F:{p['name']}", "source": "predicted", "prob": p["prob"]}
            for p in data.get("predictions", [])
        ]
        _state["fable_cache"][cache_key] = terms
        return terms
    except Exception as e:
        print(f"[go] FABLE fetch failed: {e}")
        _state["fable_cache"][(seq_hash, uniprot_id)] = []
        return []


def _get_all_go_terms(sequence: str, uniprot_id: str, pdb_id: str = "") -> dict:
    """
    Three-tier GO annotation:
      Tier 1: UniProt REST (experimental)
      Tier 2: SIFTS canonical ID → UniProt REST
      Tier 3: FABLE predicted logits
    Returns dict with keys: known, predicted, all, go_source
    """
    # Tier 1
    known = _fetch_go_terms_uniprot(uniprot_id)
    go_source = "none"

    if not known and pdb_id:
        # Tier 2: resolve canonical via SIFTS
        canonical = _fetch_canonical_uniprot_sifts(pdb_id, uniprot_id)
        if canonical != uniprot_id:
            tier2_terms = _fetch_go_terms_uniprot(canonical)
            if tier2_terms:
                known = tier2_terms
                go_source = "sifts"
                print(f"  [go] {uniprot_id} → canonical {canonical} via SIFTS ({len(known)} terms)")

    if known and go_source == "none":
        go_source = "uniprot"

    # Tier 3: FABLE (augments, not replaces)
    predicted = _fetch_go_terms_fable(sequence, uniprot_id)
    if not known and predicted:
        go_source = "fable"

    _state["go_source_cache"][uniprot_id] = go_source

    known_terms = {t["term"] for t in known}
    deduped_predicted = [t for t in predicted if t["term"] not in known_terms]

    return {
        "known":     known,
        "predicted": deduped_predicted,
        "all":       known + deduped_predicted,
        "go_source": go_source,
    }


def _fetch_uniprot_taxid(uniprot_id: str) -> int:
    """Return NCBI taxon ID for a UniProt accession. 0 = unknown."""
    if uniprot_id in _state["taxid_cache"]:
        return _state["taxid_cache"][uniprot_id]
    disk = _TAXID_DISK_CACHE / f"{uniprot_id}.json"
    if disk.exists():
        try:
            taxid = json.loads(disk.read_text()).get("taxid", 0)
            _state["taxid_cache"][uniprot_id] = taxid
            return taxid
        except Exception:
            pass
    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
                         timeout=8)
        taxid = 0
        if r.status_code == 200:
            taxid = r.json().get("organism", {}).get("taxonId", 0)
        _TAXID_DISK_CACHE.mkdir(parents=True, exist_ok=True)
        disk.write_text(json.dumps({"taxid": taxid}))
        _state["taxid_cache"][uniprot_id] = taxid
        return taxid
    except Exception:
        _state["taxid_cache"][uniprot_id] = 0
        return 0


def _fetch_ppi(uniprot_a: str, uniprot_b: str, taxid: int = 0) -> float:
    """Query STRING for PPI score. Resolves per-pair taxid; cross-species pairs return 0.5."""
    key = tuple(sorted([uniprot_a, uniprot_b]))
    if key in _state["ppi_cache"]:
        return _state["ppi_cache"][key]
    # Resolve organism per UniProt accession; short-circuit for cross-species pairs.
    ta = _fetch_uniprot_taxid(uniprot_a) or taxid
    tb = _fetch_uniprot_taxid(uniprot_b) or taxid
    if ta and tb and ta != tb:
        _state["ppi_cache"][key] = 0.5
        return 0.5
    resolved_taxid = ta or tb or taxid
    if not resolved_taxid:
        _state["ppi_cache"][key] = 0.5
        return 0.5
    try:
        ids = f"{uniprot_a}%0D{uniprot_b}"
        url = (
            f"https://string-db.org/api/json/network"
            f"?identifiers={ids}&species={resolved_taxid}&required_score=0"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        data = r.json()
        if not data:
            _state["ppi_cache"][key] = 0.5
            return 0.5
        score = data[0].get("score", 0.5)
        _state["ppi_cache"][key] = float(score)
        return float(score)
    except Exception:
        _state["ppi_cache"][key] = 0.5
        return 0.5


# ── Scoring ───────────────────────────────────────────────────────────────────

def _sasa_score(seq: str, assembled_seqs: list[str], total_complex_len: int = 0) -> float:
    if not assembled_seqs:
        if total_complex_len > 0:
            return len(seq) / total_complex_len
        return 0.5
    all_len = sum(len(s) for s in assembled_seqs) + len(seq)
    return len(seq) / all_len


def _esm_score(emb: np.ndarray, assembled_embs: list[np.ndarray]) -> float:
    if not assembled_embs:
        return 0.5
    sims = []
    for ae in assembled_embs:
        cos = float(np.dot(emb, ae) / (np.linalg.norm(emb) * np.linalg.norm(ae) + 1e-8))
        sims.append((cos + 1.0) / 2.0)
    return float(np.mean(sims))


def _go_coherence(go_a: list[dict], assembled_go: list[list[dict]]) -> float:
    if not assembled_go:
        return 0.5
    set_a = {t["term"] for t in go_a}
    scores = []
    for go_b in assembled_go:
        set_b = {t["term"] for t in go_b}
        union = set_a | set_b
        if not union:
            scores.append(0.5)
            continue
        scores.append(len(set_a & set_b) / len(union))
    return float(np.mean(scores))


def _ppi_score(uniprot_a: str, assembled_uniprots: list[str], taxid: int) -> float:
    if not assembled_uniprots:
        return 0.5
    scores = [_fetch_ppi(uniprot_a, up, taxid) for up in assembled_uniprots]
    return float(np.mean(scores))


def _combined_score(sasa: float, esm: float, go: float, ppi: float) -> float:
    return W_SASA * sasa + W_ESM * esm + W_GO * go + W_PPI * ppi


# ── Circular τ ────────────────────────────────────────────────────────────────

def _kendall_tau(pred: list[int], gt: list[int]) -> float:
    n = len(pred)
    if n < 2:
        return 1.0
    pos = {v: i for i, v in enumerate(gt)}
    arr = [pos[v] for v in pred]
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] < arr[j]:
                concordant += 1
            elif arr[i] > arr[j]:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else 1.0


def _circular_tau(pred: list[int], gt: list[int]) -> float:
    n = len(pred)
    best = -2.0
    for k in range(n):
        rotated = gt[k:] + gt[:k]
        best = max(best, _kendall_tau(pred, rotated))
        best = max(best, _kendall_tau(pred, list(reversed(rotated))))
    return best


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate(
    pdb_id: str,
    temperature: float = 1.0,
    stochastic: bool = False,
    _emb_overrides: Optional[dict] = None,
    _go_overrides: Optional[dict] = None,
) -> dict:
    cplx = COMPLEXES[pdb_id]
    subunits = cplx["subunits"]
    N = len(subunits)
    taxid = cplx["taxid"]

    embs = list(_state["embeddings"].get(pdb_id, []))
    go_data = [_get_all_go_terms(s["sequence"], s["uniprot"], pdb_id) for s in subunits]

    if _emb_overrides:
        for idx, emb in _emb_overrides.items():
            if idx < len(embs):
                embs[idx] = emb
    if _go_overrides:
        for idx, go in _go_overrides.items():
            go_data[idx] = go
    total_complex_len = sum(len(s["sequence"]) for s in subunits)

    assembled_indices = []
    assembled_embs = []
    assembled_go = []
    assembled_uniprots = []
    assembled_seqs = []

    steps = []
    gt_order = cplx["gt_order"]
    remaining = list(range(N))

    mrr_reciprocals = []
    ndcg_values = []
    gt_prob_per_step = []
    score_margin_per_step = []

    # GNN precomputed tensors (available after v5 model loaded + _precompute_gnn_tensors run)
    gnn_tensors = _state["gnn_tensors"].get(pdb_id)
    gnn_model   = _state["gnn_model"]
    use_gnn     = (gnn_tensors is not None and gnn_model is not None
                   and not stochastic and _emb_overrides is None)

    if use_gnn:
        node_feats_t, edge_feats_t, adj_t, copy_indices_t = gnn_tensors
        with torch.no_grad():
            h_global = gnn_model.embed(node_feats_t, adj_t, edge_feats_t, copy_indices_t)

    for step in range(N):
        gt_next = gt_order[step]
        candidates = []

        # GNN path: score all candidates at once, extract attention
        gnn_scores  = None
        attn_matrix = None
        if use_gnn:
            assembled_mask = torch.zeros(N, dtype=torch.bool)
            if assembled_indices:
                assembled_mask[assembled_indices] = True
            with torch.no_grad():
                gnn_logits, attn_matrix = gnn_model.forward_with_attn(
                    node_feats_t, adj_t, edge_feats_t,
                    assembled_mask, remaining, copy_indices_t,
                )
            # Map candidate index → GNN logit
            gnn_scores = {idx: float(gnn_logits[i].item()) for i, idx in enumerate(remaining)}

        sasa_mat = _state["sasa_matrices"].get(pdb_id)

        for idx in remaining:
            sub = subunits[idx]
            emb = embs[idx] if embs else np.random.randn(640).astype(np.float32)

            if sasa_mat is not None and assembled_indices:
                sasa = float(np.mean([sasa_mat[idx, j] for j in assembled_indices]))
            elif sasa_mat is not None:
                # First subunit: use total buried potential (row sum as proxy for reactivity)
                sasa = float(sasa_mat[idx].sum() / max(len(subunits) - 1, 1))
            else:
                sasa = _sasa_score(sub["sequence"], assembled_seqs, total_complex_len)
            esm  = _esm_score(emb, assembled_embs)
            go   = _go_coherence(go_data[idx]["all"], assembled_go)
            ppi  = _ppi_score(sub["uniprot"], assembled_uniprots, taxid)

            if gnn_scores is not None:
                raw_score = gnn_scores[idx]  # GNN logit (unbounded)
            else:
                raw_score = _combined_score(sasa, esm, go, ppi)

            candidates.append({
                "idx": idx,
                "chain": sub["chain"],
                "name": sub["name"],
                "uniprot": sub["uniprot"],
                "total_score": round(raw_score, 4),
                "sasa": round(sasa, 4),
                "esm": round(esm, 4),
                "go": round(go, 4),
                "ppi": round(ppi, 4),
                "scores": {
                    "sasa": round(sasa, 4),
                    "esm": round(esm, 4),
                    "go": round(go, 4),
                    "ppi": round(ppi, 4),
                    "combined": round(raw_score, 4),
                },
                "weighted": {
                    "sasa": round(W_SASA * sasa, 4),
                    "esm": round(W_ESM * esm, 4),
                    "go": round(W_GO * go, 4),
                    "ppi": round(W_PPI * ppi, 4),
                },
                "seq_len":            len(sub["sequence"]),
                "go_terms":           [t["term"] for t in go_data[idx]["all"][:8]],
                "go_terms_known":     [t["term"] for t in go_data[idx]["known"][:8]],
                "go_terms_predicted": [{"term": t["term"], "prob": t.get("prob", 0.0)}
                                       for t in go_data[idx]["predicted"][:8]],
            })

        scores_arr = np.array([c["scores"]["combined"] for c in candidates])
        logits = scores_arr / max(temperature, 1e-3)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        if stochastic:
            chosen_local = int(np.random.choice(len(candidates), p=probs))
        else:
            chosen_local = int(np.argmax(scores_arr))

        chosen = candidates[chosen_local]
        chosen_idx = chosen["idx"]

        sorted_cands = sorted(candidates, key=lambda c: c["scores"]["combined"], reverse=True)
        gt_candidate = next((c for c in candidates if c["idx"] == gt_next), None)

        # Extended metrics per step (A2)
        gt_rank = next((i + 1 for i, c in enumerate(sorted_cands) if c["idx"] == gt_next), len(sorted_cands))
        mrr_reciprocals.append(1.0 / gt_rank)
        ndcg_values.append(1.0 / math.log2(gt_rank + 1))

        gt_local = next((i for i, c in enumerate(candidates) if c["idx"] == gt_next), 0)
        gt_prob_per_step.append(round(float(probs[gt_local]), 4))

        margin = (sorted_cands[0]["scores"]["combined"] - sorted_cands[1]["scores"]["combined"]
                  if len(sorted_cands) > 1 else 0.0)
        score_margin_per_step.append(round(margin, 4))

        # go_source for this step's winner
        winner_go_source = go_data[chosen_idx].get("go_source", "none")

        steps.append({
            "step": step,
            "chosen": chosen,
            "chosen_prob": round(float(probs[chosen_local]), 4),
            "correct": chosen_idx == gt_next,
            "gt_idx": gt_next,
            "gt_candidate": gt_candidate,
            "candidates": sorted_cands,
            "probs": {c["idx"]: round(float(probs[i]), 4) for i, c in enumerate(candidates)},
            "score_spread": round(float(scores_arr.max() - scores_arr.min()), 4),
            "is_homomer_step": cplx["type"] == "homomer",
            "go_source": winner_go_source,
            # GNN attention weights (N×N flattened list) — None when heuristic used
            "attention_weights": attn_matrix,
            "scorer": "gnn" if use_gnn else "heuristic",
        })

        remaining.remove(chosen_idx)
        assembled_indices.append(chosen_idx)
        assembled_embs.append(embs[chosen_idx] if embs else np.random.randn(640).astype(np.float32))
        assembled_go.append(go_data[chosen_idx]["all"])
        assembled_uniprots.append(subunits[chosen_idx]["uniprot"])
        assembled_seqs.append(subunits[chosen_idx]["sequence"])

    pred_order = assembled_indices

    if cplx["is_ring"]:
        tau = _circular_tau(pred_order, gt_order)
    else:
        tau = _kendall_tau(pred_order, gt_order)

    correct = sum(p == g for p, g in zip(pred_order, gt_order))
    top1_acc = correct / N

    # Extended metrics (A2)
    spearman_rho = 0.0
    if N >= 2:
        try:
            rho = scipy_spearmanr(pred_order, gt_order).statistic
            spearman_rho = round(float(rho), 4) if not math.isnan(rho) else 0.0
        except Exception:
            spearman_rho = 0.0

    mrr = round(float(np.mean(mrr_reciprocals)), 4) if mrr_reciprocals else 0.0
    ndcg = round(float(np.mean(ndcg_values)), 4) if ndcg_values else 0.0

    first_subunit_correct = bool(pred_order[0] == gt_order[0]) if pred_order else False
    last_subunit_correct  = bool(pred_order[-1] == gt_order[-1]) if pred_order else False

    longest_correct_prefix = 0
    for p, g in zip(pred_order, gt_order):
        if p == g:
            longest_correct_prefix += 1
        else:
            break

    return {
        "pdb_id": pdb_id,
        "name": cplx["name"],
        "description": cplx["description"],
        "type": cplx["type"],
        "is_ring": cplx["is_ring"],
        "n_subunits": N,
        "chains": [s["chain"] for s in subunits],
        "pred_order": pred_order,
        "gt_order": gt_order,
        "tau": round(tau, 4),
        "top1_acc": round(top1_acc, 4),
        "scorer": "gnn" if use_gnn else "heuristic",
        "gnn_version": _state.get("gnn_version"),
        # Extended metrics
        "spearman_rho": spearman_rho,
        "mrr": mrr,
        "ndcg": ndcg,
        "first_subunit_correct": first_subunit_correct,
        "last_subunit_correct": last_subunit_correct,
        "longest_correct_prefix": longest_correct_prefix,
        "pathway_entropy": _state["entropy_cache"].get(pdb_id, None),
        "score_margin_per_step": score_margin_per_step,
        "gt_prob_per_step": gt_prob_per_step,
        "steps": steps,
        "subunits": [
            {
                "idx": i,
                "chain": s["chain"],
                "name": s["name"],
                "uniprot": s["uniprot"],
                "seq_len": len(s["sequence"]),
                "sequence": s["sequence"],
                "go_terms":           [t["term"] for t in go_data[i]["all"][:8]],
                "go_terms_known":     [t["term"] for t in go_data[i]["known"][:8]],
                "go_terms_predicted": [{"term": t["term"], "prob": t.get("prob", 0.0)}
                                       for t in go_data[i]["predicted"][:8]],
                "n_go_known":         len(go_data[i]["known"]),
                "n_go_predicted":     len(go_data[i]["predicted"]),
                "go_source":          go_data[i].get("go_source", "none"),
            }
            for i, s in enumerate(subunits)
        ],
        "fable_used": _state["fable_available"] is True,
    }


# ── Pathway entropy ───────────────────────────────────────────────────────────

def _compute_pathway_entropy(pdb_id: str, n_runs: int = 100, temperature: float = 0.5) -> float:
    path_counts: dict[tuple, int] = {}
    for _ in range(n_runs):
        result = _simulate(pdb_id, temperature=temperature, stochastic=True)
        key = tuple(result["pred_order"])
        path_counts[key] = path_counts.get(key, 0) + 1
    entropy = 0.0
    for count in path_counts.values():
        p = count / n_runs
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _entropy_worker():
    """Background thread: compute pathway entropy for all complexes."""
    for pdb_id in COMPLEXES:
        if pdb_id in _state["entropy_cache"]:
            continue
        try:
            entropy = _compute_pathway_entropy(pdb_id)
            _state["entropy_cache"][pdb_id] = entropy
            print(f"  [entropy] {pdb_id} H={entropy:.4f} bits")
        except Exception as e:
            print(f"  [entropy] {pdb_id} FAILED: {e}")
            _state["entropy_cache"][pdb_id] = 0.0


# ── GNN model loading ─────────────────────────────────────────────────────────

def _load_gnn():
    """Load trained AssemblyScorer; records version and dimensions."""
    if not _GNN_CHECKPOINT.exists():
        print(f"[gnn] checkpoint not found at {_GNN_CHECKPOINT} — GNN scoring disabled")
        return False
    try:
        ckpt = torch.load(str(_GNN_CHECKPOINT), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            node_dim   = ckpt.get("node_dim", 655)
            edge_dim   = ckpt.get("edge_dim", 5)
            version    = ckpt.get("version", "unknown")
        else:
            state_dict = ckpt
            # Infer dims from weight shapes
            node_dim = state_dict["node_proj.weight"].shape[1]
            edge_dim = state_dict["binding_prob.net.0.weight"].shape[1]
            version  = "unknown"

        model = AssemblyScorer(node_dim=node_dim, edge_dim=edge_dim)
        model.load_state_dict(state_dict)
        model.eval()
        _state["gnn_model"]   = model
        _state["gnn_version"] = version
        print(f"[gnn] AssemblyScorer {version} loaded (node_dim={node_dim}, edge_dim={edge_dim})")
        return True
    except Exception as e:
        print(f"[gnn] load failed: {e} — GNN scoring disabled")
        return False


def _fetch_pdb_file(pdb_id: str) -> Optional[Path]:
    """Return path to PDB file, fetching from RCSB into cache if needed."""
    cached = _PDB_CACHE_DIR / f"{pdb_id.lower()}.pdb"
    if cached.exists():
        return cached
    # Check local complex dirs
    for d in [
        Path(__file__).parent.parent / "data" / "complexes" / "marsh2013",
        Path(__file__).parent.parent / "data" / "complexes",
    ]:
        p = d / f"{pdb_id}.pdb"
        if p.exists():
            return p
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=20)
        if r.status_code == 200:
            cached.write_text(r.text)
            return cached
    except Exception as exc:
        print(f"  [pdb] fetch {pdb_id} failed: {exc}")
    return None


def _precompute_gnn_tensors(pdb_id: str) -> bool:
    """Build and cache GNN feature tensors for one viewer complex.
    Returns True on success; False if GNN model is not loaded or features fail."""
    gnn = _state["gnn_model"]
    if gnn is None:
        return False

    cplx     = COMPLEXES[pdb_id]
    subunits = cplx["subunits"]
    N        = len(subunits)
    chains   = [s["chain"] for s in subunits]
    seqs     = [s["sequence"] for s in subunits]
    uniprots = [s.get("uniprot", "") for s in subunits]

    # Determine expected dims from loaded model
    node_dim = _state["gnn_model"].node_proj.weight.shape[1]
    edge_dim = _state["gnn_model"].edge_dim

    try:
        # 1. ESM embeddings (640d)
        embs = np.stack(_state["embeddings"].get(pdb_id, [np.zeros(640)] * N))  # (N, 640)

        # 2. pLDDT stats (5d) — load from cache or use neutral defaults
        plddt = np.full((N, 5), [70.0, 0.5, 0.1, 60.0, 80.0], dtype=np.float32)
        for i, uid in enumerate(uniprots):
            cache_path = _PLDDT_CACHE_DIR / f"{pdb_id}_plddt.npy"
            if cache_path.exists():
                try:
                    arr = np.load(str(cache_path)).astype(np.float32)
                    if arr.shape == (N, 5):
                        plddt = arr
                        break
                except Exception:
                    pass

        # Normalise pLDDT cols 0, 3, 4 to [0,1]
        plddt_n = plddt.copy()
        plddt_n[:, 0] /= 100.0
        plddt_n[:, 3] /= 100.0
        plddt_n[:, 4] /= 100.0

        # 3. SASA diagonal (1d) — zeros from normalised BSA matrix (matches training precompute)
        sasa_norm_mat = _state["sasa_matrices"].get(pdb_id)
        if sasa_norm_mat is None:
            sasa_norm_mat = _load_sasa_matrix(pdb_id, N)
            if sasa_norm_mat is not None:
                _state["sasa_matrices"][pdb_id] = sasa_norm_mat
        sasa_diag = np.diag(sasa_norm_mat).astype(np.float32) if sasa_norm_mat is not None \
                    else np.zeros(N, dtype=np.float32)

        # 4. Geometry features (9d) — load from AlphaFold cache or zeros
        geom = np.zeros((N, 9), dtype=np.float32)
        try:
            from src.data_prep.geom_features import build_geom_matrix, GEOM_DIM
            geom = build_geom_matrix(uniprots, _GEOM_CACHE_DIR, seqs).astype(np.float32)
        except Exception:
            pass  # zeros fallback

        node_parts = [embs, plddt_n, sasa_diag.reshape(-1, 1), geom]

        # 5. Chirality node features (6d) — if model expects them
        if node_dim >= 661:
            chir_n_path = _CHIRALITY_CACHE / f"{pdb_id}_chirality_node.npy"
            chir_e_path = _CHIRALITY_CACHE / f"{pdb_id}_chirality_edge.npy"
            chirality_node, chirality_edge = None, None
            if chir_n_path.exists() and chir_e_path.exists():
                try:
                    chirality_node = np.load(str(chir_n_path)).astype(np.float32)
                    chirality_edge = np.load(str(chir_e_path)).astype(np.float32)
                except Exception:
                    pass
            if chirality_node is None:
                pdb_path = _fetch_pdb_file(pdb_id)
                if pdb_path:
                    try:
                        chirality_node, chirality_edge = compute_chirality_features(str(pdb_path), chains)
                        _CHIRALITY_CACHE.mkdir(parents=True, exist_ok=True)
                        np.save(str(chir_n_path), chirality_node)
                        np.save(str(chir_e_path), chirality_edge)
                    except Exception as exc:
                        print(f"  [{pdb_id}] chirality failed: {exc}")
            if chirality_node is not None and chirality_node.shape == (N, 6):
                node_parts.append(chirality_node)
            else:
                node_parts.append(np.zeros((N, 6), dtype=np.float32))

        node_feats = np.hstack(node_parts).astype(np.float32)

        # Pad / trim to exact node_dim expected by model
        if node_feats.shape[1] < node_dim:
            pad = np.zeros((N, node_dim - node_feats.shape[1]), dtype=np.float32)
            node_feats = np.hstack([node_feats, pad])
        node_feats = node_feats[:, :node_dim]

        # 6. Edge feature matrices
        norms  = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_n = embs / (norms + 1e-8)
        esm_cos = (embs_n @ embs_n.T).clip(-1, 1).astype(np.float32)

        # GO cosine from FABLE bridge (zeros if unavailable)
        go_cos = np.zeros((N, N), dtype=np.float32)

        # PPI from STRING cache (zeros if not yet fetched)
        ppi_mat = np.zeros((N, N), dtype=np.float32)
        taxid = cplx.get("taxid", 9606)
        for i in range(N):
            for j in range(N):
                if i != j:
                    key = (uniprots[i], uniprots[j], taxid)
                    if key in _state["ppi_cache"]:
                        ppi_mat[i, j] = _state["ppi_cache"][key]

        # SASA matrix — use real normalised BSA (already loaded at startup)
        sasa_mat = _state["sasa_matrices"].get(pdb_id)
        if sasa_mat is None:
            sasa_mat = np.zeros((N, N), dtype=np.float32)

        # Contact density and chirality edge features
        contact_density = None
        chir_edge_mat   = None
        if edge_dim >= 6:
            chir_e_path = _CHIRALITY_CACHE / f"{pdb_id}_chirality_edge.npy"
            if chir_e_path.exists():
                try:
                    ce = np.load(str(chir_e_path)).astype(np.float32)
                    if ce.shape == (N, N):
                        chir_edge_mat = ce
                except Exception:
                    pass

        edge_feats_t = gnn_build_edge(
            ppi_mat, esm_cos, go_cos, sasa_mat,
            contact_density=contact_density,
            handedness=chir_edge_mat,
        )

        # Pad edge features if needed
        if edge_feats_t.shape[2] < edge_dim:
            pad = torch.zeros(N, N, edge_dim - edge_feats_t.shape[2])
            edge_feats_t = torch.cat([edge_feats_t, pad], dim=2)
        edge_feats_t = edge_feats_t[:, :, :edge_dim]

        node_feats_t  = torch.tensor(node_feats, dtype=torch.float32)
        adj_t         = torch.ones(N, N, dtype=torch.bool)
        copy_indices_t = build_copy_indices(uniprots)

        _state["gnn_tensors"][pdb_id] = (node_feats_t, edge_feats_t, adj_t, copy_indices_t)
        return True

    except Exception as exc:
        print(f"  [{pdb_id}] GNN tensor precompute failed: {exc}")
        return False


# ── SASA loading ─────────────────────────────────────────────────────────────

_SASA_CACHE_DIR = BASE_DIR.parent / "data" / "sasa_cache"
_MARSH_PDB_DIR  = BASE_DIR.parent / "data" / "complexes" / "marsh2013"


def _load_sasa_matrix(pdb_id: str, n_chains: int) -> Optional[np.ndarray]:
    """Return normalised (N,N) BSA matrix. Tries disk cache, then freesasa, then None."""
    cache_path = _SASA_CACHE_DIR / f"{pdb_id}_sasa.npy"
    if cache_path.exists():
        try:
            mat = np.load(str(cache_path)).astype(np.float32)
            if mat.shape == (n_chains, n_chains):
                maxv = mat.max()
                return mat / (maxv + 1e-8) if maxv > 0 else mat
        except Exception:
            pass

    pdb_path = _fetch_pdb_file(pdb_id)
    if pdb_path is None:
        return None
    cplx = COMPLEXES[pdb_id]
    chains = [s["chain"] for s in cplx["subunits"]]
    try:
        from src.data_prep.fast_sasa import compute_bsa_matrix
        mat = compute_bsa_matrix(str(pdb_path), chains).astype(np.float32)
        _SASA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), mat)
        print(f"  [{pdb_id}] SASA computed via freesasa and cached")
        maxv = mat.max()
        return mat / (maxv + 1e-8) if maxv > 0 else mat
    except Exception as exc:
        print(f"  [{pdb_id}] SASA compute failed: {exc}")
        return None


# ── Startup ───────────────────────────────────────────────────────────────────

def _startup_worker():
    print("[startup] Loading ESM-2 150M...")
    esm_ok = _load_esm()
    _probe_fable()

    # Load GNN scorer (non-blocking — falls back to heuristic if unavailable)
    _load_gnn()

    all_seqs = [s["sequence"][:MAX_EMBED_LEN] for cplx in COMPLEXES.values() for s in cplx["subunits"]]
    n_unique = len(set(hashlib.md5(s.encode()).hexdigest() for s in all_seqs))
    print(f"[startup] {len(all_seqs)} total subunits, {n_unique} unique sequences to embed")

    print("[startup] Loading SASA matrices...")
    for pdb_id, cplx in COMPLEXES.items():
        N = len(cplx["subunits"])
        sasa = _load_sasa_matrix(pdb_id, N)
        if sasa is not None:
            _state["sasa_matrices"][pdb_id] = sasa
            print(f"  [{pdb_id}] SASA loaded ({N}×{N}, max_raw={sasa.max():.3f})")
        else:
            print(f"  [{pdb_id}] SASA unavailable — will use length proxy")

    for pdb_id, cplx in COMPLEXES.items():
        t0 = time.time()

        embs = []
        for sub in cplx["subunits"]:
            seq_key = hashlib.md5(sub["sequence"][:MAX_EMBED_LEN].encode()).hexdigest()
            hit = seq_key in _embed_cache
            emb = _embed_sequence(sub["sequence"])
            embs.append(emb)
            if not hit:
                print(f"  [embed] {pdb_id}/{sub['chain']} {len(sub['sequence'])} aa → computed")
            else:
                print(f"  [embed] {pdb_id}/{sub['chain']} → cache hit")
        _state["embeddings"][pdb_id] = embs

        # Precompute GNN tensors for this complex (requires embeddings to be set first)
        if _state["gnn_model"] is not None:
            _precompute_gnn_tensors(pdb_id)

        try:
            result = _simulate(pdb_id)
            _state["simulations"][pdb_id] = result
            elapsed = time.time() - t0
            print(f"  [ready] {pdb_id} τ={result['tau']:.3f} top1={result['top1_acc']:.3f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [startup] {pdb_id} FAILED: {e}")


    _state["ready"] = True
    print(f"[startup] All {len(_state['simulations'])} complexes ready")

    # Asynchronous entropy computation (doesn't block API)
    threading.Thread(target=_entropy_worker, daemon=True).start()
    print("[startup] Pathway entropy computation started in background")


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_startup_worker, daemon=True)
    thread.start()
    yield


# ── Models ───────────────────────────────────────────────────────────────────

class MutateRequest(BaseModel):
    pdb_id: str
    chain: str
    position: int
    original_aa: str
    mutant_aa: str
    n_runs: int = 100


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Protein Assembly Viewer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/status")
@app.get("/api/status")
def status():
    return {
        "ready": _state["ready"],
        "n_simulations": len(_state["simulations"]),
        "n_total": len(COMPLEXES),
        "fable_available": _state["fable_available"],
        "gnn_loaded": _state["gnn_model"] is not None,
        "gnn_version": _state.get("gnn_version"),
    }


@app.get("/pdb/{pdb_id}")
def get_pdb(pdb_id: str):
    """Serve a PDB file for 3Dmol.js rendering. Fetches from RCSB if not cached."""
    from fastapi.responses import PlainTextResponse
    pdb_id = pdb_id.upper().replace(".PDB", "")
    pdb_path = _fetch_pdb_file(pdb_id)
    if pdb_path is None:
        raise HTTPException(status_code=404, detail=f"PDB file not found for {pdb_id}")
    return PlainTextResponse(pdb_path.read_text(), media_type="text/plain")


@app.get("/complexes")
def list_complexes():
    result = []
    for pdb_id, cplx in COMPLEXES.items():
        sim = _state["simulations"].get(pdb_id)
        result.append({
            "pdb_id": pdb_id,
            "name": cplx["name"],
            "description": cplx["description"],
            "type": cplx["type"],
            "is_ring": cplx["is_ring"],
            "n_subunits": len(cplx["subunits"]),
            "tau": sim["tau"] if sim else None,
            "top1_acc": sim["top1_acc"] if sim else None,
            "spearman_rho": sim["spearman_rho"] if sim else None,
            "mrr": sim["mrr"] if sim else None,
            "ndcg": sim["ndcg"] if sim else None,
            "pathway_entropy": _state["entropy_cache"].get(pdb_id, None),
            "ready": sim is not None,
        })
    return result


@app.get("/complex/{pdb_id}")
@app.get("/api/simulate/{pdb_id}")
def get_complex(pdb_id: str, temperature: float = 1.0):
    pdb_id = pdb_id.upper()
    if pdb_id not in COMPLEXES:
        raise HTTPException(status_code=404, detail=f"Unknown complex: {pdb_id}")

    if temperature != 1.0 or pdb_id not in _state["simulations"]:
        try:
            result = _simulate(pdb_id, temperature=temperature)
            if temperature == 1.0:
                _state["simulations"][pdb_id] = result
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    sim = _state["simulations"].get(pdb_id)
    if sim is None:
        if not _state["ready"]:
            raise HTTPException(status_code=503, detail="Server still initializing")
        raise HTTPException(status_code=404, detail=f"Simulation not available for {pdb_id}")
    # Inject latest entropy if newly computed
    sim["pathway_entropy"] = _state["entropy_cache"].get(pdb_id, None)
    return sim


@app.post("/api/mutate")
def mutate(req: MutateRequest):
    pdb_id = req.pdb_id.upper()
    if pdb_id not in COMPLEXES:
        raise HTTPException(status_code=404, detail=f"Unknown complex: {pdb_id}")

    cplx = COMPLEXES[pdb_id]
    subunits = cplx["subunits"]

    # Find subunit index by chain letter
    chain_upper = req.chain.upper()
    sub_idx = next((i for i, s in enumerate(subunits) if s["chain"].upper() == chain_upper), None)
    if sub_idx is None:
        raise HTTPException(status_code=400, detail=f"Chain {req.chain!r} not found in {pdb_id}")

    original_seq = subunits[sub_idx]["sequence"]

    # Validate amino acids
    _AA1_VALID = frozenset("ACDEFGHIKLMNPQRSTVWY")
    if req.original_aa.upper() not in _AA1_VALID:
        raise HTTPException(status_code=400, detail=f"Invalid original_aa: {req.original_aa!r}")
    if req.mutant_aa.upper() not in _AA1_VALID:
        raise HTTPException(status_code=400, detail=f"Invalid mutant_aa: {req.mutant_aa!r}")

    # Validate position and original_aa match
    pos = req.position
    if not (1 <= pos <= len(original_seq)):
        raise HTTPException(status_code=400, detail=f"Position {pos} out of range [1, {len(original_seq)}]")
    actual_aa = original_seq[pos - 1].upper()
    if actual_aa != req.original_aa.upper():
        raise HTTPException(status_code=400,
            detail=f"Position {pos} has {actual_aa!r}, not {req.original_aa.upper()!r}")

    # Check mutation cache
    cache_key = f"{chain_upper}_{pos}_{req.original_aa.upper()}_{req.mutant_aa.upper()}"
    cache_path = _MUTATION_CACHE_DIR / pdb_id / f"{cache_key}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    # WT simulation — use cached if available
    wt_sim = _state["simulations"].get(pdb_id)
    if wt_sim is None:
        try:
            wt_sim = _simulate(pdb_id)
            _state["simulations"][pdb_id] = wt_sim
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"WT simulation failed: {e}")

    # Apply point mutation
    try:
        mut_seq = apply_mutation(original_seq, pos, req.mutant_aa.upper())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Re-embed mutant sequence
    try:
        mut_emb = _embed_sequence(mut_seq)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ESM embedding failed: {e}")

    # Recompute FABLE GO for mutant sequence (UniProt GO terms unchanged)
    uniprot_id = subunits[sub_idx]["uniprot"]
    known_go = _fetch_go_terms_uniprot(uniprot_id)
    predicted_go = _fetch_go_terms_fable(mut_seq, uniprot_id)
    known_terms = {t["term"] for t in known_go}
    deduped_predicted = [t for t in predicted_go if t["term"] not in known_terms]
    mut_go = {
        "known": known_go,
        "predicted": deduped_predicted,
        "all": known_go + deduped_predicted,
        "go_source": "uniprot" if known_go else ("fable" if predicted_go else "none"),
    }

    # Run mutant simulation with overrides
    try:
        mut_sim = _simulate(
            pdb_id,
            _emb_overrides={sub_idx: mut_emb},
            _go_overrides={sub_idx: mut_go},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mutant simulation failed: {e}")

    # Compute deltas
    delta_tau   = round(mut_sim["tau"] - wt_sim["tau"], 4)
    delta_top1  = round(mut_sim["top1_acc"] - wt_sim["top1_acc"], 4)

    wt_gt_probs  = wt_sim.get("gt_prob_per_step", [])
    mut_gt_probs = mut_sim.get("gt_prob_per_step", [])
    delta_gt_prob = [round(m - w, 4) for w, m in zip(wt_gt_probs, mut_gt_probs)]

    # KL divergence per step
    wt_probs_per_step  = [step["probs"] for step in wt_sim.get("steps", [])]
    mut_probs_per_step = [step["probs"] for step in mut_sim.get("steps", [])]
    kl_per_step = compute_kl_per_step(wt_probs_per_step, mut_probs_per_step)

    # Pathway redirect detection
    redirect = detect_pathway_redirect(wt_sim["pred_order"], mut_sim["pred_order"])

    # GO term delta
    wt_go_terms  = {t["term"] for t in wt_sim["subunits"][sub_idx].get("go_terms", [])}
    mut_go_all   = {t["term"] for t in mut_go["all"]}
    go_gained = sorted(mut_go_all - wt_go_terms)
    go_lost   = sorted(wt_go_terms - mut_go_all)

    result = {
        "pdb_id":       pdb_id,
        "chain":        chain_upper,
        "position":     pos,
        "original_aa":  req.original_aa.upper(),
        "mutant_aa":    req.mutant_aa.upper(),
        "wt":  {
            "tau":          wt_sim["tau"],
            "top1_acc":     wt_sim["top1_acc"],
            "pred_order":   wt_sim["pred_order"],
            "gt_prob_per_step": wt_gt_probs,
        },
        "mut": {
            "tau":          mut_sim["tau"],
            "top1_acc":     mut_sim["top1_acc"],
            "pred_order":   mut_sim["pred_order"],
            "gt_prob_per_step": mut_gt_probs,
        },
        "deltas": {
            "tau":           delta_tau,
            "top1_acc":      delta_top1,
            "gt_prob_per_step": delta_gt_prob,
        },
        "kl_per_step":  kl_per_step,
        "redirected":   redirect["redirected"],
        "redirect_info": redirect,
        "go_delta": {
            "gained": go_gained,
            "lost":   go_lost,
        },
        "subunit_name": subunits[sub_idx]["name"],
        "uniprot":      uniprot_id,
    }

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_path.write_text(json.dumps(result, indent=2))
    except Exception:
        pass

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
