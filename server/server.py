"""
FastAPI server for protein assembly simulation viewer.
Loads ESM-2 150M at startup, precomputes all 10 complex simulations, then serves them.
"""

import asyncio
import hashlib
import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.channel_health import channel_stats
from src.models.attribution import edge_channel_attribution, normalise_attribution

# Edge channel order produced by _build_gnn_features(), and the UI labels they
# map to. Keep in step with that function — the viewer reads these names.
GNN_EDGE_CHANNELS = {"ppi": 0, "esm": 1, "go": 2, "sasa": 3}

# ── GNN model (optional — falls back to weighted scoring if checkpoint missing) ─
_gnn_model = None
_gnn_node_dim = 646

def _load_gnn():
    global _gnn_model, _gnn_node_dim
    ckpt_path = Path(__file__).parent / "best_gnn.pt"
    if not ckpt_path.exists():
        print("[gnn] no checkpoint found — using weighted scoring")
        return
    try:
        from scorer_gnn import AssemblyScorer
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        _gnn_node_dim = ckpt.get("node_dim", 646)
        model = AssemblyScorer(
            node_dim=_gnn_node_dim, edge_dim=4, hidden=128, n_layers=3, heads=4, dropout=0.0
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _gnn_model = model
        print(f"[gnn] loaded checkpoint epoch={ckpt.get('epoch')}, "
              f"gt_prob={ckpt.get('gt_prob', 0):.4f}, tau={ckpt.get('mean_tau', 0):.4f}")
    except Exception as e:
        print(f"[gnn] failed to load ({e}) — using weighted scoring")

# ── Constants ─────────────────────────────────────────────────────────────────

_THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}
_PDB_CACHE_DIR = Path("/tmp/pdb_cache")

W_SASA = 0.40
W_ESM  = 0.35
W_GO   = 0.15
W_PPI  = 0.10

PROTFUNC_URL = os.environ.get("PROTFUNC_URL", "http://localhost:7860")

STATIC_DIR = Path(__file__).parent / "static"

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
    "embeddings": {},      # pdb_id -> list of per-subunit 640-d arrays
    "simulations": {},     # pdb_id -> simulation result dict
    "go_cache": {},        # uniprot -> list of {"term": "F:...", "source": "uniprot"}
    "protfunc_cache": {},  # seq_hash -> list of {"term": "F:...", "source": "predicted", "prob": float}
    "ppi_cache": {},       # (up1, up2) -> score float
    "protfunc_available": None,  # None=untested, True/False after first probe
    "ready": False,
    "gnn_available": False,
}


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


_embed_cache: dict[str, np.ndarray] = {}  # md5(seq[:MAX_EMBED_LEN]) -> embedding
MAX_EMBED_LEN = 1022  # ESM-2 hard max (1024 positions minus BOS/EOS)


def _embed_sequence(sequence: str) -> np.ndarray:
    """Embed a sequence with ESM-2. Results are cached by sequence content so
    identical chains (homomers) are only embedded once."""
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
    """Returns UniProt GO terms as [{"term": "F:...", "source": "uniprot"}]."""
    if uniprot_id in _state["go_cache"]:
        return _state["go_cache"][uniprot_id]
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        r = requests.get(url, timeout=10)
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


def _probe_protfunc() -> bool:
    """Check once whether ProtFunc is reachable; cache result."""
    if _state["protfunc_available"] is not None:
        return _state["protfunc_available"]
    try:
        r = requests.get(f"{PROTFUNC_URL}/api/health", timeout=3)
        ok = r.status_code == 200 and r.json().get("model_loaded", False)
        _state["protfunc_available"] = ok
        if ok:
            print(f"[go] ProtFunc reachable at {PROTFUNC_URL}")
        else:
            print(f"[go] ProtFunc at {PROTFUNC_URL} responded but model not loaded — using UniProt only")
        return ok
    except Exception:
        _state["protfunc_available"] = False
        print(f"[go] ProtFunc not reachable at {PROTFUNC_URL} — using UniProt only")
        return False


def _fetch_go_terms_protfunc(sequence: str, uniprot_id: str = "") -> list[dict]:
    """
    Returns ProtFunc-predicted GO-MF terms as
    [{"term": "F:name", "source": "predicted", "prob": float}].
    Falls back to [] if ProtFunc unreachable.
    """
    if not _probe_protfunc():
        return []
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    cache_key = (seq_hash, uniprot_id)
    if cache_key in _state["protfunc_cache"]:
        return _state["protfunc_cache"][cache_key]
    try:
        payload = {"sequence": sequence, "uniprot_id": uniprot_id, "taxon": "auto", "top_k": 20}
        r = requests.post(f"{PROTFUNC_URL}/api/go_terms", json=payload, timeout=15)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        data = r.json()
        if "error" in data:
            raise ValueError(data["error"])
        terms = [
            {"term": f"F:{p['name']}", "source": "predicted", "prob": p["prob"]}
            for p in data.get("predictions", [])
        ]
        _state["protfunc_cache"][cache_key] = terms
        return terms
    except Exception as e:
        print(f"[go] ProtFunc fetch failed: {e}")
        _state["protfunc_cache"][(seq_hash, uniprot_id)] = []
        return []


def _get_all_go_terms(sequence: str, uniprot_id: str) -> dict:
    """
    Merge UniProt experimental + ProtFunc predicted GO terms.
    Returns {"known": [...], "predicted": [...], "all": [...]}
    where each entry is {"term": "F:...", "source": "uniprot"|"predicted"}.
    Deduplication: UniProt wins on exact term string match.
    """
    known     = _fetch_go_terms_uniprot(uniprot_id)
    predicted = _fetch_go_terms_protfunc(sequence, uniprot_id)
    known_terms = {t["term"] for t in known}
    deduped_predicted = [t for t in predicted if t["term"] not in known_terms]
    return {
        "known":     known,
        "predicted": deduped_predicted,
        "all":       known + deduped_predicted,
    }


def _fetch_ppi(uniprot_a: str, uniprot_b: str, taxid: int) -> float:
    key = tuple(sorted([uniprot_a, uniprot_b]))
    if key in _state["ppi_cache"]:
        return _state["ppi_cache"][key]
    try:
        ids = f"{uniprot_a}%0D{uniprot_b}"
        url = (
            f"https://string-db.org/api/json/network"
            f"?identifiers={ids}&species={taxid}&required_score=0"
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
    """Proxy SASA score: longer sequence → more surface area → higher contribution.
    At step 0 (no assembled context), score relative to total complex mass so
    larger subunits score higher as nucleation seeds."""
    if not assembled_seqs:
        if total_complex_len > 0:
            return len(seq) / total_complex_len
        return 0.5
    all_len = sum(len(s) for s in assembled_seqs) + len(seq)
    return len(seq) / all_len


def _esm_score(emb: np.ndarray, assembled_embs: list[np.ndarray]) -> float:
    """Mean cosine similarity to already-assembled subunits."""
    if not assembled_embs:
        return 0.5
    sims = []
    for ae in assembled_embs:
        cos = float(np.dot(emb, ae) / (np.linalg.norm(emb) * np.linalg.norm(ae) + 1e-8))
        sims.append((cos + 1.0) / 2.0)
    return float(np.mean(sims))


def _go_coherence(go_a: list[dict], assembled_go: list[list[dict]]) -> float:
    """Jaccard similarity of GO terms with assembled subunits.
    Accepts list-of-dicts with 'term' key; uses full merged set."""
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


def _combined_score(
    sasa: float, esm: float, go: float, ppi: float
) -> float:
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


# ── GNN feature builders ─────────────────────────────────────────────────────

def _channel_values(ef: torch.Tensor, candidates: list, assembled: list) -> dict:
    """What each edge channel says about each candidate, given the assembled set.

    Mirrors how AssemblyScorer.score_candidates forms `edge_to_complex`: the
    mean of the candidate's edges to everything already placed. With nothing
    placed there is no pairwise context yet, so the channel genuinely has no
    opinion and we report the candidate's mean edge to the whole complex
    instead of inventing a neutral 0.5.
    """
    e = ef.detach().cpu().numpy()
    out = {}
    for name, ch in GNN_EDGE_CHANNELS.items():
        if ch >= e.shape[-1]:
            out[name] = [0.0] * len(candidates)
            continue
        vals = []
        for idx in candidates:
            row = e[idx, :, ch]
            sel = row[assembled] if assembled else np.delete(row, idx)
            vals.append(float(sel.mean()) if sel.size else 0.0)
        out[name] = vals
    return out


def _gnn_channel_health(ef: torch.Tensor) -> dict:
    """Classify each edge channel for this complex (ok / saturated / dead)."""
    e = ef.detach().cpu().numpy()
    mats = {name: (e[:, :, ch] if ch < e.shape[-1] else None)
            for name, ch in GNN_EDGE_CHANNELS.items()}
    return channel_stats(mats, reference="sasa")


def _build_gnn_features(pdb_id: str, subunits: list, embs: list, go_data: list, taxid: int):
    """Build (node_feats, edge_feats, adj) tensors for GNN inference."""
    N = len(subunits)
    ESM_DIM = 640
    PLDDT_DIM = 5  # [mean, frac_gt90, frac_lt50, q10, q90]

    emb_arr = np.stack(embs).astype(np.float32)  # (N, 640)

    # ESM cosine matrix
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    emb_n = emb_arr / (norms + 1e-8)
    esm_cos = (emb_n @ emb_n.T).clip(-1, 1).astype(np.float32)

    # pLDDT (neutral defaults — no AlphaFold lookup in HF space)
    plddt = np.zeros((N, PLDDT_DIM), dtype=np.float32)
    plddt[:, 0] = 0.70   # mean/100
    plddt[:, 1] = 0.30   # frac_gt90
    plddt[:, 2] = 0.10   # frac_lt50
    plddt[:, 3] = 0.50   # q10/100
    plddt[:, 4] = 0.90   # q90/100

    # SASA diagonal (sequence-length proxy)
    lens = np.array([len(s["sequence"]) for s in subunits], dtype=np.float32)
    sasa_diag = (lens / (lens.max() + 1e-8)).reshape(-1, 1)

    # SASA matrix (off-diagonal proxy)
    total_len = lens.sum()
    sasa_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                sasa_mat[i, j] = min(lens[i], lens[j]) / (total_len + 1e-8)
    sasa_norm = sasa_mat / (sasa_mat.max() + 1e-8)

    # PPI matrix
    ppi_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            s = _fetch_ppi(subunits[i]["uniprot"], subunits[j]["uniprot"], taxid)
            ppi_mat[i, j] = ppi_mat[j, i] = s

    # GO cosine (Jaccard of GO term sets)
    go_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        si = {t["term"] for t in go_data[i]["all"]}
        for j in range(N):
            sj = {t["term"] for t in go_data[j]["all"]}
            union = si | sj
            go_mat[i, j] = len(si & sj) / len(union) if union else 0.0

    # Node features: ESM(640) + pLDDT(5) + SASA_diag(1)
    node_feats = np.hstack([emb_arr, plddt, sasa_diag]).astype(np.float32)

    # Edge features: [ppi, esm_cos, go_cos, sasa_norm]
    edge_feats = np.stack([ppi_mat, esm_cos, go_mat, sasa_norm], axis=-1).astype(np.float32)

    adj = np.ones((N, N), dtype=bool)

    nf_t  = torch.tensor(node_feats)
    ef_t  = torch.tensor(edge_feats)
    adj_t = torch.tensor(adj)
    return nf_t, ef_t, adj_t, sasa_mat, ppi_mat, go_mat


def _simulate_gnn(pdb_id: str, temperature: float = 1.0) -> dict:
    """Run GNN-based assembly simulation."""
    cplx    = COMPLEXES[pdb_id]
    subunits = cplx["subunits"]
    N        = len(subunits)
    taxid    = cplx["taxid"]
    embs     = _state["embeddings"].get(pdb_id, [np.zeros(640)] * N)
    go_data  = [_get_all_go_terms(s["sequence"], s["uniprot"]) for s in subunits]

    nf, ef, adj, sasa_mat, ppi_mat, go_mat = _build_gnn_features(
        pdb_id, subunits, embs, go_data, taxid
    )
    health = _gnn_channel_health(ef)

    model    = _gnn_model
    gt_order = cplx["gt_order"]
    remaining = list(range(N))
    assembled = []
    steps = []

    with torch.no_grad():
        h = model.embed(nf, adj, ef)

    for step_idx in range(N):
        gt_next = gt_order[step_idx]
        candidates_info = []

        # Score all remaining candidates
        if remaining:
            am = torch.zeros(N, dtype=torch.bool)
            for idx in assembled:
                am[idx] = True

            with torch.no_grad():
                raw_scores = model.score_candidates(h, ef, am, remaining)
                logits = raw_scores / max(temperature, 1e-3)
                probs_t = torch.softmax(logits, dim=0).numpy()
                scores_np = raw_scores.numpy()

            # Per-channel readouts for the viewer. `scores` is what each channel
            # says about this candidate given what is already assembled;
            # `weighted` is occlusion attribution — how far the GNN's score
            # actually moves when that channel is neutralised. The GNN has no
            # linear decomposition, so attribution is the only honest analogue
            # of the heuristic scorer's w·x terms.
            chan_vals = _channel_values(ef, remaining, assembled)
            attr = normalise_attribution(edge_channel_attribution(
                model, nf, ef, adj, am, remaining, GNN_EDGE_CHANNELS,
            ))

            for local_i, idx in enumerate(remaining):
                sub = subunits[idx]
                candidates_info.append({
                    "idx":    idx,
                    "chain":  sub["chain"],
                    "name":   sub["name"],
                    "uniprot": sub["uniprot"],
                    "scores": {
                        "gnn":      round(float(scores_np[local_i]), 4),
                        "sasa":     round(chan_vals["sasa"][local_i], 4),
                        "esm":      round(chan_vals["esm"][local_i], 4),
                        "go":       round(chan_vals["go"][local_i], 4),
                        "ppi":      round(chan_vals["ppi"][local_i], 4),
                        "combined": round(float(scores_np[local_i]), 4),
                    },
                    "weighted": {
                        k: round(float(attr[k][local_i]), 4) for k in GNN_EDGE_CHANNELS
                    },
                    # Per-channel data availability. A dead channel renders as
                    # "no data" rather than as a real-looking bar — the previous
                    # hardcoded 0.5 made an empty channel look like a genuine
                    # mid-range score.
                    "available": {k: health[k].usable for k in GNN_EDGE_CHANNELS},
                    "seq_len":            len(sub["sequence"]),
                    "go_terms":           [t["term"] for t in go_data[idx]["all"][:8]],
                    "go_terms_known":     [t["term"] for t in go_data[idx]["known"][:8]],
                    "go_terms_predicted": [{"term": t["term"], "prob": t.get("prob", 0.0)}
                                           for t in go_data[idx]["predicted"][:8]],
                })

            best_local = int(np.argmax(scores_np))
            chosen_info = candidates_info[best_local]
            chosen_idx  = remaining[best_local]

            sorted_cands  = sorted(candidates_info, key=lambda c: c["scores"]["combined"], reverse=True)
            gt_candidate  = next((c for c in candidates_info if c["idx"] == gt_next), None)
            chosen_prob   = float(probs_t[best_local])

            steps.append({
                "step":           step_idx,
                "chosen":         chosen_info,
                "chosen_prob":    round(chosen_prob, 4),
                "correct":        chosen_idx == gt_next,
                "gt_idx":         gt_next,
                "gt_candidate":   gt_candidate,
                "candidates":     sorted_cands,
                "probs":          {c["idx"]: round(float(probs_t[i]), 4) for i, c in enumerate(candidates_info)},
                "score_spread":   round(float(scores_np.max() - scores_np.min()), 4),
                "is_homomer_step": cplx["type"] == "homomer",
            })

            remaining.remove(chosen_idx)
            assembled.append(chosen_idx)

    pred_order = assembled
    tau = _circular_tau(pred_order, gt_order) if cplx["is_ring"] else _kendall_tau(pred_order, gt_order)
    top1_acc = sum(p == g for p, g in zip(pred_order, gt_order)) / N

    return {
        "pdb_id":        pdb_id,
        "name":          cplx["name"],
        "description":   cplx["description"],
        "type":          cplx["type"],
        "is_ring":       cplx["is_ring"],
        "n_subunits":    N,
        "pred_order":    pred_order,
        "gt_order":      gt_order,
        "tau":           round(tau, 4),
        "top1_acc":      round(top1_acc, 4),
        "steps":         steps,
        "scorer":        "gnn",
        "subunits": [
            {
                "idx": i, "chain": s["chain"], "name": s["name"],
                "uniprot": s["uniprot"], "seq_len": len(s["sequence"]),
                "go_terms":           [t["term"] for t in go_data[i]["all"][:8]],
                "go_terms_known":     [t["term"] for t in go_data[i]["known"][:8]],
                "go_terms_predicted": [{"term": t["term"], "prob": t.get("prob", 0.0)}
                                       for t in go_data[i]["predicted"][:8]],
                "n_go_known":         len(go_data[i]["known"]),
                "n_go_predicted":     len(go_data[i]["predicted"]),
            }
            for i, s in enumerate(subunits)
        ],
        "protfunc_used": _state["protfunc_available"] is True,
    }


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate(pdb_id: str, temperature: float = 1.0) -> dict:
    cplx = COMPLEXES[pdb_id]
    subunits = cplx["subunits"]
    N = len(subunits)
    taxid = cplx["taxid"]

    embs = _state["embeddings"].get(pdb_id, [])
    go_data = [_get_all_go_terms(s["sequence"], s["uniprot"]) for s in subunits]
    total_complex_len = sum(len(s["sequence"]) for s in subunits)

    # Which channels actually have data behind them for this complex. The
    # scorers below fall back to 0.5 when a source is missing, which is a
    # sensible neutral for the *ranking* but is indistinguishable in the UI from
    # a genuine mid-range score — so report availability explicitly.
    _avail = {
        "sasa": True,                                   # always derivable from sequence
        "esm":  bool(embs),                             # else random vectors → cos≈0 → 0.5
        "go":   any(g["all"] for g in go_data),         # else Jaccard of empty sets
        "ppi":  all(s.get("uniprot") for s in subunits),  # else STRING cannot be queried
    }

    assembled_indices = []
    assembled_embs = []
    assembled_go = []
    assembled_uniprots = []
    assembled_seqs = []

    steps = []
    gt_order = cplx["gt_order"]
    remaining = list(range(N))

    for step in range(N):
        gt_next = gt_order[step]  # which subunit index GT says should be chosen now
        candidates = []
        for idx in remaining:
            sub = subunits[idx]
            emb = embs[idx] if embs else np.random.randn(640).astype(np.float32)

            sasa = _sasa_score(sub["sequence"], assembled_seqs, total_complex_len)
            esm  = _esm_score(emb, assembled_embs)
            go   = _go_coherence(go_data[idx]["all"], assembled_go)
            ppi  = _ppi_score(sub["uniprot"], assembled_uniprots, taxid)

            raw_score = _combined_score(sasa, esm, go, ppi)

            candidates.append({
                "idx": idx,
                "chain": sub["chain"],
                "name": sub["name"],
                "uniprot": sub["uniprot"],
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
                "available": dict(_avail),
                "seq_len":            len(sub["sequence"]),
                "go_terms":           [t["term"] for t in go_data[idx]["all"][:8]],
                "go_terms_known":     [t["term"] for t in go_data[idx]["known"][:8]],
                "go_terms_predicted": [{"term": t["term"], "prob": t.get("prob", 0.0)}
                                       for t in go_data[idx]["predicted"][:8]],
            })

        # temperature-scaled softmax selection
        scores = np.array([c["scores"]["combined"] for c in candidates])
        logits = scores / max(temperature, 1e-3)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        chosen_local = int(np.argmax(scores))
        chosen = candidates[chosen_local]
        chosen_idx = chosen["idx"]

        sorted_cands = sorted(candidates, key=lambda c: c["scores"]["combined"], reverse=True)
        gt_candidate = next((c for c in candidates if c["idx"] == gt_next), None)

        steps.append({
            "step": step,
            "chosen": chosen,
            "chosen_prob": round(float(probs[chosen_local]), 4),
            "correct": chosen_idx == gt_next,
            "gt_idx": gt_next,
            "gt_candidate": gt_candidate,
            "candidates": sorted_cands,
            "probs": {c["idx"]: round(float(probs[i]), 4) for i, c in enumerate(candidates)},
            "score_spread": round(float(scores.max() - scores.min()), 4),
            "is_homomer_step": cplx["type"] == "homomer",
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

    return {
        "pdb_id": pdb_id,
        "name": cplx["name"],
        "description": cplx["description"],
        "type": cplx["type"],
        "is_ring": cplx["is_ring"],
        "n_subunits": N,
        "pred_order": pred_order,
        "gt_order": gt_order,
        "tau": round(tau, 4),
        "top1_acc": round(top1_acc, 4),
        "steps": steps,
        "subunits": [
            {
                "idx": i,
                "chain": s["chain"],
                "name": s["name"],
                "uniprot": s["uniprot"],
                "seq_len": len(s["sequence"]),
                "go_terms":           [t["term"] for t in go_data[i]["all"][:8]],
                "go_terms_known":     [t["term"] for t in go_data[i]["known"][:8]],
                "go_terms_predicted": [{"term": t["term"], "prob": t.get("prob", 0.0)}
                                       for t in go_data[i]["predicted"][:8]],
                "n_go_known":         len(go_data[i]["known"]),
                "n_go_predicted":     len(go_data[i]["predicted"]),
            }
            for i, s in enumerate(subunits)
        ],
        "protfunc_used": _state["protfunc_available"] is True,
    }


# ── Dynamic (on-demand) simulation ───────────────────────────────────────────

def _download_pdb(pdb_id: str) -> Path:
    _PDB_CACHE_DIR.mkdir(exist_ok=True)
    pdb_path = _PDB_CACHE_DIR / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise ValueError(f"PDB {pdb_id} not found at RCSB (HTTP {r.status_code})")
        pdb_path.write_bytes(r.content)
    return pdb_path


def _extract_chains_from_pdb(pdb_path: Path) -> list[dict]:
    """Parse ATOM records, return one entry per protein chain (≥10 residues)."""
    chains: dict[str, dict] = {}
    seen: dict[str, set] = {}
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        res_name = line[17:20].strip()
        if res_name not in _THREE_TO_ONE:
            continue
        chain = line[21]
        resseq = line[22:26].strip()
        insert = line[26]
        res_key = f"{resseq}{insert}"
        if chain not in chains:
            chains[chain] = {}
            seen[chain] = set()
        if res_key not in seen[chain]:
            seen[chain].add(res_key)
            try:
                num = int(resseq)
            except ValueError:
                num = 0
            chains[chain][res_key] = (num, res_name)

    result = []
    for chain_id in sorted(chains.keys()):
        sorted_res = sorted(chains[chain_id].values(), key=lambda x: x[0])
        seq = "".join(_THREE_TO_ONE[r] for _, r in sorted_res)
        if len(seq) < 10:
            continue
        result.append({
            "chain": chain_id,
            "uniprot": f"CHAIN_{chain_id}",
            "name": f"Chain {chain_id}",
            "sequence": seq,
        })
    return result


def _simulate_dynamic_gnn(pdb_id: str, subunits: list) -> dict:
    """GNN assembly simulation for arbitrary PDB — no ground truth available."""
    N = len(subunits)
    embs = [_embed_sequence(s["sequence"]) for s in subunits]
    go_data = [{"known": [], "predicted": [], "all": []} for _ in subunits]

    nf, ef, adj, sasa_mat, ppi_mat, go_mat = _build_gnn_features(
        pdb_id, subunits, embs, go_data, taxid=9606
    )
    health = _gnn_channel_health(ef)

    model = _gnn_model
    remaining = list(range(N))
    assembled: list[int] = []
    steps = []

    with torch.no_grad():
        h = model.embed(nf, adj, ef)

    for step_idx in range(N):
        candidates_info = []
        if remaining:
            am = torch.zeros(N, dtype=torch.bool)
            for idx in assembled:
                am[idx] = True

            with torch.no_grad():
                raw_scores = model.score_candidates(h, ef, am, remaining)
                probs_t = torch.softmax(raw_scores, dim=0).numpy()
                scores_np = raw_scores.numpy()

            chan_vals = _channel_values(ef, remaining, assembled)
            attr = normalise_attribution(edge_channel_attribution(
                model, nf, ef, adj, am, remaining, GNN_EDGE_CHANNELS,
            ))

            for local_i, idx in enumerate(remaining):
                sub = subunits[idx]
                candidates_info.append({
                    "idx": idx,
                    "chain": sub["chain"],
                    "name": sub["name"],
                    "uniprot": sub["uniprot"],
                    "scores": {
                        "gnn":      round(float(scores_np[local_i]), 4),
                        "sasa":     round(chan_vals["sasa"][local_i], 4),
                        "esm":      round(chan_vals["esm"][local_i], 4),
                        "go":       round(chan_vals["go"][local_i], 4),
                        "ppi":      round(chan_vals["ppi"][local_i], 4),
                        "combined": round(float(scores_np[local_i]), 4),
                    },
                    "weighted": {
                        k: round(float(attr[k][local_i]), 4) for k in GNN_EDGE_CHANNELS
                    },
                    "available": {k: health[k].usable for k in GNN_EDGE_CHANNELS},
                    "seq_len": len(sub["sequence"]),
                    "go_terms": [], "go_terms_known": [], "go_terms_predicted": [],
                })

            best_local = int(np.argmax(scores_np))
            chosen_info = candidates_info[best_local]
            chosen_idx = remaining[best_local]
            sorted_cands = sorted(candidates_info, key=lambda c: c["scores"]["combined"], reverse=True)
            chosen_prob = float(probs_t[best_local])

            steps.append({
                "step":           step_idx,
                "chosen":         chosen_info,
                "chosen_prob":    round(chosen_prob, 4),
                "correct":        None,
                "gt_idx":         None,
                "gt_candidate":   None,
                "candidates":     sorted_cands,
                "probs":          {c["idx"]: round(float(probs_t[i]), 4) for i, c in enumerate(candidates_info)},
                "score_spread":   round(float(scores_np.max() - scores_np.min()), 4),
                "is_homomer_step": False,
            })

            remaining.remove(chosen_idx)
            assembled.append(chosen_idx)

    return {
        "pdb_id":      pdb_id,
        "name":        pdb_id,
        "description": f"On-demand simulation — {N} chains from PDB {pdb_id}",
        "type":        "unknown",
        "is_ring":     False,
        "n_subunits":  N,
        "pred_order":  assembled,
        "gt_order":    None,
        "tau":         None,
        "top1_acc":    None,
        "steps":       steps,
        "scorer":      "gnn",
        "no_gt":       True,
        "subunits": [
            {
                "idx": i, "chain": s["chain"], "name": s["name"],
                "uniprot": s["uniprot"], "seq_len": len(s["sequence"]),
                "go_terms": [], "go_terms_known": [], "go_terms_predicted": [],
                "n_go_known": 0, "n_go_predicted": 0,
            }
            for i, s in enumerate(subunits)
        ],
        "protfunc_used": False,
    }


# ── Startup ───────────────────────────────────────────────────────────────────

_CORPUS_DF: pd.DataFrame | None = None

def _load_corpus():
    global _CORPUS_DF
    corpus_path = Path(__file__).parent / "labeled_dataset.tsv"
    if not corpus_path.exists():
        print("[corpus] labeled_dataset.tsv not found — browse disabled")
        return
    try:
        df = pd.read_csv(corpus_path, sep="\t")
        df["n_subunits"] = df["n_subunits"].fillna(0).astype(int)
        df["gt_tier"]    = df["gt_tier"].fillna(3).astype(int)
        df["name"]       = df["name"].fillna("").astype(str)
        df["source"]     = df["source"].fillna("").astype(str)
        df["type"]       = df["type"].fillna("heteromer").astype(str)
        df["pdb_id"]     = df["pdb_id"].fillna("").astype(str)
        _CORPUS_DF = df
        print(f"[corpus] Loaded {len(df):,} complexes")
    except Exception as e:
        print(f"[corpus] Load failed: {e}")


def _startup_worker():
    _load_corpus()
    print("[startup] Loading ESM-2 150M...")
    esm_ok = _load_esm()
    _load_gnn()
    _state["gnn_available"] = _gnn_model is not None
    _probe_protfunc()

    # Count unique sequences up front so we can log cache hits
    all_seqs = [s["sequence"][:MAX_EMBED_LEN] for cplx in COMPLEXES.values() for s in cplx["subunits"]]
    n_unique = len(set(hashlib.md5(s.encode()).hexdigest() for s in all_seqs))
    print(f"[startup] {len(all_seqs)} total subunits, {n_unique} unique sequences to embed")

    for pdb_id, cplx in COMPLEXES.items():
        t0 = time.time()

        # ── Embed (cache hits are free) ───────────────────────────────────────
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

        # ── Simulate (GNN if available, else weighted scoring) ───────────────
        try:
            if _state["gnn_available"]:
                result = _simulate_gnn(pdb_id)
            else:
                result = _simulate(pdb_id)
            _state["simulations"][pdb_id] = result
            elapsed = time.time() - t0
            print(f"  [ready] {pdb_id} τ={result['tau']:.3f} top1={result['top1_acc']:.3f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [startup] {pdb_id} FAILED: {e}")

    _state["ready"] = True
    print(f"[startup] All {len(_state['simulations'])} complexes ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_startup_worker, daemon=True)
    thread.start()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Protein Assembly Viewer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/status")
def status():
    return {
        "ready":              _state["ready"],
        "n_simulations":      len(_state["simulations"]),
        "n_total":            len(COMPLEXES),
        "corpus_ready":       _CORPUS_DF is not None,
        "corpus_n":           len(_CORPUS_DF) if _CORPUS_DF is not None else 0,
        "protfunc_available": _state["protfunc_available"],
        "gnn_available":      _state["gnn_available"],
        "scorer":             "gnn" if _state["gnn_available"] else "weighted",
    }


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
            "ready": sim is not None,
        })
    return result


@app.get("/complex/{pdb_id}")
def get_complex(pdb_id: str, temperature: float = 1.0):
    pdb_id = pdb_id.upper()
    if pdb_id not in COMPLEXES:
        raise HTTPException(status_code=404, detail=f"Unknown complex: {pdb_id}")

    if temperature != 1.0 or pdb_id not in _state["simulations"]:
        try:
            if _state["gnn_available"]:
                result = _simulate_gnn(pdb_id, temperature=temperature)
            else:
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
    return sim


@app.get("/complex/dynamic/{pdb_id}")
def get_complex_dynamic(pdb_id: str):
    pdb_id = pdb_id.upper()
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Server still initializing — please wait")
    if _gnn_model is None:
        raise HTTPException(status_code=503, detail="GNN model not available for dynamic simulation")
    try:
        pdb_path = _download_pdb(pdb_id)
        subunits = _extract_chains_from_pdb(pdb_path)
        if not subunits:
            raise ValueError("No protein chains found in PDB file")
        if len(subunits) > 24:
            subunits = subunits[:24]
        print(f"[dynamic] {pdb_id}: {len(subunits)} chains")
        return _simulate_dynamic_gnn(pdb_id, subunits)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[dynamic] {pdb_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corpus/stats")
def corpus_stats():
    if _CORPUS_DF is None:
        return {"n": 0, "by_tier": {}, "by_source": {}, "n_subunits_dist": {}}
    df = _CORPUS_DF
    return {
        "n":              len(df),
        "by_tier":        {int(k): int(v) for k, v in df.groupby("gt_tier").size().items()},
        "by_source":      {str(k): int(v) for k, v in df.groupby("source").size().items()},
        "n_subunits_dist":{int(k): int(v) for k, v in df.groupby("n_subunits").size().items()},
    }


@app.get("/corpus/search")
def corpus_search(
    query:  str = Query(""),
    source: str = Query("all"),
    tier:   str = Query("all"),
    type_:  str = Query("all", alias="type"),
    n_min:  int = Query(2),
    n_max:  int = Query(12),
    limit:  int = Query(100),
    offset: int = Query(0),
):
    if _CORPUS_DF is None:
        return {"n": 0, "results": []}
    df    = _CORPUS_DF
    mask  = pd.Series([True] * len(df), index=df.index)
    if query.strip():
        q     = query.strip().upper()
        mask &= (df["name"].str.upper().str.contains(q, na=False) |
                 df["pdb_id"].str.upper().str.contains(q, na=False) |
                 df["uniprot_ids"].astype(str).str.upper().str.contains(q, na=False))
    src_map = {"corum": "corum", "rcsb": "rcsb_search",
               "complex_portal": "complex_portal", "experimental": "experimental_gt"}
    if source != "all":
        mask &= df["source"] == src_map.get(source, source)
    if tier != "all":
        try: mask &= df["gt_tier"] == int(tier)
        except ValueError: pass
    if type_ != "all":
        mask &= df["type"] == type_
    mask &= (df["n_subunits"] >= n_min) & (df["n_subunits"] <= n_max)
    result = df[mask].sort_values(["gt_tier", "name"])
    total  = len(result)
    tier_label = {1: "★ T1", 2: "◆ T2", 3: "◇ T3"}
    page = result.iloc[offset: offset + limit][
        ["pdb_id", "name", "n_subunits", "type", "gt_tier", "source"]
    ].copy()
    page["gt_tier"] = page["gt_tier"].map(tier_label)
    return {"n": total, "results": page.fillna("").to_dict("records")}


# ── Mutation endpoint ─────────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel

class _MutateRequest(_BaseModel):
    pdb_id: str
    chain: str
    position: int
    original_aa: str
    mutant_aa: str
    n_runs: int = 1

_AA1_MUT = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _kl_per_step(wt_steps, mut_steps, eps=1e-9):
    kls = []
    for ws, ms in zip(wt_steps, mut_steps):
        wt_p  = {c: p for c, p in zip(ws.get("candidates",[]),  ws.get("probs",[]))}
        mut_p = {c: p for c, p in zip(ms.get("candidates",[]), ms.get("probs",[]))}
        idx = sorted(set(wt_p) | set(mut_p))
        p = np.clip([wt_p.get(i, eps)  for i in idx], eps, None); p /= p.sum()
        q = np.clip([mut_p.get(i, eps) for i in idx], eps, None); q /= q.sum()
        kls.append(round(float(np.sum(p * np.log(p / q))), 4))
    return kls


def _mut_redirect(wt_order, mut_order):
    n = min(len(wt_order), len(mut_order))
    for i in range(n):
        if wt_order[i] != mut_order[i]:
            return {"redirected": True, "first_diff": i, "num_swapped": n - i}
    return {"redirected": False, "first_diff": None, "num_swapped": 0}


def _run_gnn_sim_with_embs(pdb_id: str, emb_override: dict | None = None) -> dict:
    """Run GNN simulation for pdb_id, optionally overriding embeddings at subunit indices."""
    cplx = COMPLEXES[pdb_id]
    subunits = cplx["subunits"]
    N = len(subunits)
    taxid = cplx["taxid"]

    base_embs = list(_state["embeddings"].get(pdb_id, [np.zeros(640)] * N))
    if emb_override:
        for idx, emb in emb_override.items():
            base_embs[idx] = emb

    go_data = [_get_all_go_terms(s["sequence"], s["uniprot"]) for s in subunits]
    nf, ef, adj, sasa_mat, ppi_mat, go_mat = _build_gnn_features(pdb_id, subunits, base_embs, go_data, taxid)

    model = _gnn_model
    gt_order = cplx["gt_order"]
    remaining = list(range(N))
    assembled = []
    steps = []

    with torch.no_grad():
        h = model.embed(nf, adj, ef)

    for step_idx in range(N):
        candidates = list(remaining)
        if not candidates:
            break
        am = torch.zeros(N, dtype=torch.bool)
        for idx in assembled:
            am[idx] = True
        with torch.no_grad():
            raw = model.score_candidates(h, ef, am, candidates)
            logits = raw
            probs_t = torch.softmax(logits, dim=0).numpy()
        best_i = int(raw.argmax())
        chosen = candidates[best_i]
        step_probs = {str(c): round(float(probs_t[li]), 6) for li, c in enumerate(candidates)}
        gt_next = gt_order[step_idx] if step_idx < len(gt_order) else None
        steps.append({
            "step": step_idx, "chosen": chosen, "candidates": candidates,
            "probs": step_probs,
            "gt_next": gt_next,
        })
        assembled.append(chosen)
        remaining.remove(chosen)

    pred_order = [s["chosen"] for s in steps]
    gt_order_l = list(gt_order)
    gt_probs = []
    for t, step in enumerate(steps):
        gt_nxt = gt_order_l[t] if t < len(gt_order_l) else None
        if gt_nxt is not None and str(gt_nxt) in step["probs"]:
            gt_probs.append(round(float(step["probs"][str(gt_nxt)]), 4))

    # tau
    if len(pred_order) >= 2 and len(gt_order_l) >= 2:
        from scipy.stats import kendalltau as _kt
        if cplx.get("is_ring", False):
            best_t = -1.0
            rev = list(reversed(gt_order_l))
            for i in range(N):
                for seq in (gt_order_l[i:]+gt_order_l[:i], rev[i:]+rev[:i]):
                    stat, _ = _kt([pred_order.index(x) for x in seq if x in pred_order], list(range(len(seq))))
                    if not np.isnan(stat): best_t = max(best_t, float(stat))
            tau = best_t
        else:
            stat, _ = _kt([pred_order.index(x) for x in gt_order_l if x in pred_order], list(range(len(gt_order_l))))
            tau = float(stat) if not np.isnan(stat) else 0.0
    else:
        tau = 0.0

    top1 = sum(1 for t, s in enumerate(steps) if t < len(gt_order_l) and s["chosen"] == gt_order_l[t]) / max(N-1, 1)

    return {
        "tau": round(tau, 4), "top1_acc": round(top1, 4),
        "pred_order": pred_order, "gt_prob_per_step": gt_probs,
        "steps": steps,
    }


@app.post("/api/mutate")
def mutate(req: _MutateRequest):
    pdb_id = req.pdb_id.strip().upper()
    chain  = req.chain.strip().upper()
    pos    = req.position

    if pdb_id not in COMPLEXES:
        raise HTTPException(404, f"Unknown complex: {pdb_id}")
    if req.original_aa.upper() not in _AA1_MUT:
        raise HTTPException(400, f"Invalid original_aa: {req.original_aa!r}")
    if req.mutant_aa.upper() not in _AA1_MUT:
        raise HTTPException(400, f"Invalid mutant_aa: {req.mutant_aa!r}")
    if _gnn_model is None:
        raise HTTPException(503, "GNN model not loaded")

    cplx = COMPLEXES[pdb_id]
    subunits = cplx["subunits"]
    sub_idx = next((i for i, s in enumerate(subunits) if s["chain"].upper() == chain), None)
    if sub_idx is None:
        raise HTTPException(400, f"Chain {chain!r} not found in {pdb_id}")

    orig_seq = subunits[sub_idx]["sequence"]
    if not (1 <= pos <= len(orig_seq)):
        raise HTTPException(400, f"Position {pos} out of range [1, {len(orig_seq)}]")
    actual_aa = orig_seq[pos - 1].upper()
    if actual_aa != req.original_aa.upper():
        raise HTTPException(400, f"Position {pos} has {actual_aa!r}, not {req.original_aa.upper()!r}")

    # WT simulation (use cached embeddings)
    wt_result = _run_gnn_sim_with_embs(pdb_id)

    # Mutant embedding
    mut_seq = orig_seq[:pos-1] + req.mutant_aa.upper() + orig_seq[pos:]
    mut_emb = _embed_sequence(mut_seq)
    mut_result = _run_gnn_sim_with_embs(pdb_id, emb_override={sub_idx: mut_emb})

    redirect = _mut_redirect(wt_result["pred_order"], mut_result["pred_order"])
    kl = _kl_per_step(wt_result["steps"], mut_result["steps"])

    return {
        "pdb_id":      pdb_id, "chain": chain, "position": pos,
        "original_aa": req.original_aa.upper(), "mutant_aa": req.mutant_aa.upper(),
        "wt":  {k: wt_result[k]  for k in ("tau","top1_acc","pred_order","gt_prob_per_step")},
        "mut": {k: mut_result[k] for k in ("tau","top1_acc","pred_order","gt_prob_per_step")},
        "deltas": {
            "tau":      round(mut_result["tau"] - wt_result["tau"], 4),
            "top1_acc": round(mut_result["top1_acc"] - wt_result["top1_acc"], 4),
            "gt_prob_per_step": [round(m-w, 4) for w, m in zip(wt_result["gt_prob_per_step"], mut_result["gt_prob_per_step"])],
        },
        "kl_per_step":  kl,
        "redirected":   redirect["redirected"],
        "redirect_info": redirect,
        "go_delta": {"gained": [], "lost": []},
        "subunit_name": subunits[sub_idx]["name"],
        "uniprot":      subunits[sub_idx]["uniprot"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
