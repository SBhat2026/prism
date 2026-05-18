"""
PRISM — Protein Ranking by Interface and Sequence Modeling
HuggingFace Space: interactive assembly order prediction from a PDB ID.
"""

import os
import sys
import json
import time
import tempfile
import warnings
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from scipy.stats import kendalltau, spearmanr
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

from scorer_gnn import AssemblyScorer, build_edge_features

# ── Constants ────────────────────────────────────────────────────────────────

PRISM_VERSION  = "v1.1-corpus-15K"
REPO_ID        = "Sbhat2026/prism"
CKPT_FILE      = "prism_v1.pt"
RCSB_PDB       = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_ENTITY_URL= "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{eid}"
AF_API         = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"
AF_FALLBACK    = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ   = 1022

_AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

# ── Model loading (cached) ────────────────────────────────────────────────────

_model = None
_esm   = None
_bconv = None
_temperature = 0.332


def _load_model():
    global _model
    if _model is not None:
        return _model
    path = hf_hub_download(REPO_ID, CKPT_FILE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    m    = AssemblyScorer(
        node_dim=cfg["node_dim"], edge_dim=cfg["edge_dim"],
        hidden=cfg["hidden"],    n_layers=cfg["n_layers"],
        heads=cfg["heads"],
    ).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    _model = m
    return m


def _load_esm():
    global _esm, _bconv
    if _esm is not None:
        return _esm, _bconv
    import esm as esm_lib
    model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    _esm   = model.to(DEVICE).eval()
    _bconv = alphabet.get_batch_converter()
    return _esm, _bconv


# ── AlphaFold pLDDT fetching ─────────────────────────────────────────────────

_PLDDT_KEYS = ["plddt_mean", "plddt_frac_gt90", "plddt_frac_lt50", "plddt_q10", "plddt_q90"]
_PLDDT_DEFAULT = np.array([70.0, 0.30, 0.10, 50.0, 90.0], dtype=np.float32)


def _fetch_af_plddt(uid: str) -> np.ndarray:
    try:
        r = requests.get(AF_API.format(uid=uid), timeout=15)
        url = r.json()[0]["pdbUrl"] if r.status_code == 200 and r.json() \
              else AF_FALLBACK.format(uid=uid)
    except Exception:
        url = AF_FALLBACK.format(uid=uid)
    try:
        r2 = requests.get(url, timeout=30)
        if r2.status_code != 200:
            return _PLDDT_DEFAULT.copy()
        vals = []
        for line in r2.text.splitlines():
            if line[:4] == "ATOM" and line[13:15].strip() == "CA":
                try: vals.append(float(line[60:66]))
                except ValueError: pass
        if not vals:
            return _PLDDT_DEFAULT.copy()
        arr = np.array(vals, dtype=np.float32)
        return np.array([arr.mean(), (arr>90).mean(), (arr<50).mean(),
                         np.percentile(arr, 10), np.percentile(arr, 90)],
                        dtype=np.float32)
    except Exception:
        return _PLDDT_DEFAULT.copy()


def fetch_plddt_for_chains(pdb_id: str, chain_ids: list[str]) -> np.ndarray:
    """Return (N, 5) pLDDT feature matrix. Falls back to defaults on error."""
    N = len(chain_ids)
    out = np.tile(_PLDDT_DEFAULT, (N, 1))
    try:
        r = requests.get(RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=20)
        if r.status_code != 200:
            return out
        entity_ids = r.json().get("rcsb_entry_container_identifiers", {}).get(
            "polymer_entity_ids", [])
        chain_uid: dict[str, str] = {}
        for eid in entity_ids:
            try:
                r2 = requests.get(RCSB_ENTITY_URL.format(pdb_id=pdb_id, eid=eid), timeout=20)
                if r2.status_code != 200: continue
                e = r2.json()
                uids = e.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
                if not uids: continue
                uid = uids[0]
                for c in e.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", []):
                    chain_uid[c] = uid
            except Exception:
                continue
        uid_cache: dict[str, np.ndarray] = {}
        for i, ch in enumerate(chain_ids):
            uid = chain_uid.get(ch)
            if not uid:
                continue
            if uid not in uid_cache:
                uid_cache[uid] = _fetch_af_plddt(uid)
            out[i] = uid_cache[uid]
    except Exception:
        pass
    return out


# ── PDB parsing ───────────────────────────────────────────────────────────────

def download_pdb(pdb_id: str) -> str:
    r = requests.get(RCSB_PDB.format(pdb_id=pdb_id.upper()), timeout=30)
    r.raise_for_status()
    return r.text


def extract_sequences(pdb_text: str) -> dict[str, str]:
    chains, seen = {}, {}
    for line in pdb_text.splitlines():
        if line[:4] != "ATOM" or line[13:15].strip() != "CA":
            continue
        ch = line[21]
        resnum, resname = line[22:26].strip(), line[17:20].strip()
        key = (resnum, resname)
        if ch not in seen:
            seen[ch] = set(); chains[ch] = []
        if key not in seen[ch]:
            seen[ch].add(key)
            chains[ch].append(_AA3TO1.get(resname, "X"))
    return {c: "".join(s) for c, s in chains.items() if len(s) >= 20}


# ── Feature computation ───────────────────────────────────────────────────────

def compute_bsa_matrix(pdb_text: str, chain_ids: list[str]) -> np.ndarray:
    import freesasa, tempfile, os
    N = len(chain_ids)
    mat = np.zeros((N, N), dtype=np.float32)

    def _sasa(subset_ids):
        lines = [l for l in pdb_text.splitlines()
                 if (l[:4] in ("ATOM","HETA") and l[21] in subset_ids)
                 or l[:4] in ("END ","TER ")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write("\n".join(lines))
            fname = f.name
        try:
            return freesasa.calc(freesasa.Structure(fname)).totalArea()
        except Exception:
            return 0.0
        finally:
            os.unlink(fname)

    iso = {c: _sasa({c}) for c in chain_ids}
    for i, ci in enumerate(chain_ids):
        for j, cj in enumerate(chain_ids):
            if j <= i: continue
            pair = _sasa({ci, cj})
            bsa  = (iso[ci] + iso[cj] - pair) / 2
            mat[i, j] = mat[j, i] = max(bsa, 0.0)
    return mat


def embed_sequences(sequences: dict[str, str]) -> dict[str, np.ndarray]:
    esm, bconv = _load_esm()
    items = [(n, s[:MAX_SEQ]) for n, s in sequences.items()]
    embeddings = {}
    for i in range(0, len(items), 4):
        batch = items[i: i+4]
        _, _, tokens = bconv([(n, s) for n, s in batch])
        with torch.no_grad():
            out = esm(tokens.to(DEVICE), repr_layers=[30], return_contacts=False)
        reps = out["representations"][30]
        for j, (name, seq) in enumerate(batch):
            emb = reps[j, 1:len(seq)+1].mean(0).cpu().numpy()
            embeddings[name] = emb.astype(np.float32)
    return embeddings


# ── Assembly simulation ───────────────────────────────────────────────────────

def sasa_greedy_order(bsa_mat: np.ndarray) -> list[int]:
    N = bsa_mat.shape[0]
    order, remaining = [], list(range(N))
    col_sum = bsa_mat.sum(axis=1)
    seed = int(col_sum.argmax())
    order.append(seed); remaining.remove(seed)
    while remaining:
        scores = {c: bsa_mat[c, order].sum() for c in remaining}
        best   = max(scores, key=scores.get)
        order.append(best); remaining.remove(best)
    return order


def circular_tau(pred, gt):
    N = len(gt)
    base, rev = list(gt), list(reversed(gt))
    best = -1.0
    for i in range(N):
        for seq in (base[i:]+base[:i], rev[i:]+rev[:i]):
            stat, _ = kendalltau([pred.index(x) for x in seq], range(N))
            if not np.isnan(stat):
                best = max(best, float(stat))
    return best


def greedy_prism(model, node_feats, edge_feats, adj, N, seed):
    order     = [seed]
    remaining = [i for i in range(N) if i != seed]
    step_probs = []
    model.eval()
    with torch.no_grad():
        for _ in range(N - 1):
            if not remaining: break
            am = torch.zeros(N, dtype=torch.bool)
            am[order] = True
            scores = torch.stack([
                model(node_feats.to(DEVICE), adj.to(DEVICE),
                      edge_feats.to(DEVICE), am.to(DEVICE), c)
                for c in remaining
            ])
            probs = torch.softmax(scores / _temperature, dim=0)
            best_i = int(scores.argmax())
            sorted_scores = scores.sort(descending=True).values
            margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            step_probs.append({
                "candidates": remaining[:],
                "probs":      probs.cpu().tolist(),
                "chosen":     remaining[best_i],
                "margin":     round(margin, 4),
            })
            order.append(remaining[best_i])
            remaining.pop(best_i)
    return order, step_probs


# ── Main inference ────────────────────────────────────────────────────────────

def predict_assembly(pdb_id: str, chain_filter: str):
    pdb_id = pdb_id.strip().upper()
    if not pdb_id or len(pdb_id) != 4:
        return "Enter a 4-character PDB ID (e.g. 2HHB)", None, None

    status_lines = []
    def log(s): status_lines.append(s)

    try:
        log(f"Downloading {pdb_id} from RCSB…")
        pdb_text = download_pdb(pdb_id)
    except Exception as e:
        return f"Failed to download {pdb_id}: {e}", None, None

    seq_map = extract_sequences(pdb_text)
    if not seq_map:
        return f"No protein chains found in {pdb_id}", None, None

    if chain_filter.strip():
        keep = [c.strip().upper() for c in chain_filter.split(",")]
        seq_map = {c: s for c, s in seq_map.items() if c in keep}
        if not seq_map:
            return f"None of the specified chains found. Available: {list(seq_map.keys())}", None, None

    chain_ids = sorted(seq_map.keys())
    N = len(chain_ids)
    sequences = [seq_map[c] for c in chain_ids]
    log(f"Chains: {chain_ids}  ({N} subunits)")

    if N < 2:
        return "Need at least 2 protein chains.", None, None
    if N > 15:
        return f"Complex has {N} chains — currently limited to ≤15 for performance.", None, None

    log("Computing buried surface area matrix…")
    try:
        bsa_mat = compute_bsa_matrix(pdb_text, chain_ids)
    except Exception as e:
        return f"BSA computation failed: {e}", None, None

    log("Computing ESM-2 150M embeddings…")
    try:
        model = _load_model()
        emb_dict = embed_sequences({f"{pdb_id}_{c}": s for c, s in zip(chain_ids, sequences)})
        emb_mat = np.stack([emb_dict[f"{pdb_id}_{c}"] for c in chain_ids])
    except Exception as e:
        return f"Embedding failed: {e}", None, None

    log("Fetching AlphaFold pLDDT…")
    plddt_f = fetch_plddt_for_chains(pdb_id, chain_ids)
    plddt_f[:, 0] /= 100.0
    plddt_f[:, 3] /= 100.0
    plddt_f[:, 4] /= 100.0

    norms    = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    esm_cos  = (emb_mat/(norms+1e-8) @ (emb_mat/(norms+1e-8)).T).clip(-1,1).astype(np.float32)
    sasa_n   = bsa_mat / (bsa_mat.max() + 1e-8)
    sasa_diag= np.diag(sasa_n).reshape(-1, 1).astype(np.float32)
    node_feats = torch.tensor(np.hstack([emb_mat, plddt_f, sasa_diag]), dtype=torch.float32)
    edge_feats = build_edge_features(
        np.zeros((N,N), dtype=np.float32), esm_cos,
        np.zeros((N,N), dtype=np.float32), sasa_n,
    )
    adj = torch.ones(N, N, dtype=torch.bool)

    sasa_order = sasa_greedy_order(bsa_mat)

    unique_seqs = set(sequences)
    is_homomer  = len(unique_seqs) == 1

    log("Running PRISM greedy assembly…")
    if is_homomer:
        best_tau, best_order, best_steps = -2.0, None, None
        for seed in range(N):
            order, steps = greedy_prism(model, node_feats, edge_feats, adj, N, seed)
            tau = circular_tau(order, sasa_order)
            if tau > best_tau:
                best_tau, best_order, best_steps = tau, order, steps
        prism_order, step_details = best_order, best_steps
    else:
        seed = sasa_order[0]
        prism_order, step_details = greedy_prism(model, node_feats, edge_feats, adj, N, seed)

    def order_str(order):
        return " → ".join(chain_ids[i] for i in order)

    spearman_rho = float(spearmanr(prism_order, sasa_order).statistic) if N > 2 else 1.0

    mrr_vals, ndcg_vals = [], []
    for t, step in enumerate(step_details):
        cands  = step["candidates"]
        probs  = step["probs"]
        chosen = step["chosen"]
        sasa_next_idx = next((i for i, c in enumerate(cands) if c == sasa_order[t + 1]), None) \
                        if t + 1 < len(sasa_order) else None
        if sasa_next_idx is not None:
            rank = sorted(range(len(probs)), key=lambda i: -probs[i]).index(sasa_next_idx) + 1
            mrr_vals.append(1.0 / rank)
            ndcg_vals.append(1.0 / (rank + 1) ** 0.5)
    mrr  = round(sum(mrr_vals)  / len(mrr_vals),  4) if mrr_vals  else None
    ndcg = round(sum(ndcg_vals) / len(ndcg_vals), 4) if ndcg_vals else None
    score_margins = [step.get("margin", 0.0) for step in step_details]

    rows = ["| Step | Chosen | Margin | Top candidates (prob) |",
            "|------|--------|--------|-----------------------|"]
    for t, step in enumerate(step_details):
        cands  = step["candidates"]
        probs  = step["probs"]
        chosen = step["chosen"]
        margin = step.get("margin", 0.0)
        top3   = sorted(zip(cands, probs), key=lambda x: -x[1])[:3]
        top3_s = "  ".join(f"{chain_ids[c]}={p:.3f}" for c, p in top3)
        rows.append(f"| {t+1} | **{chain_ids[chosen]}** | {margin:.3f} | {top3_s} |")

    if is_homomer:
        prism_tau = circular_tau(prism_order, list(range(N)))
        sasa_tau  = circular_tau(sasa_order,  list(range(N)))
        tau_line  = f"\n*Circular τ vs sequential: PRISM {prism_tau:.3f} | SASA-greedy {sasa_tau:.3f}*\n\n*(Homomer — any rotation is valid; τ measures internal consistency.)*"
    else:
        tau_line = ""

    metrics_line = (
        f"**ρ (vs SASA):** {spearman_rho:.3f}"
        + (f"  **MRR:** {mrr:.3f}" if mrr is not None else "")
        + (f"  **NDCG:** {ndcg:.3f}" if ndcg is not None else "")
        + (f"  **Avg margin:** {sum(score_margins)/len(score_margins):.3f}" if score_margins else "")
    )

    result_md = f"""## PRISM Assembly Order Prediction

**Complex:** `{pdb_id}` — {N} chains ({', '.join(chain_ids)})
**Type:** {'Homomer' if is_homomer else 'Heteromer'}

### Predicted order

**PRISM:**      {order_str(prism_order)}
**SASA-greedy baseline:** {order_str(sasa_order)}
{tau_line}
### Metrics *(vs SASA-greedy pseudo-GT)*

{metrics_line}

### Step-by-step detail

{chr(10).join(rows)}

### Feature summary

| Chain | Seq len | BSA max (Å²) | ESM self-similarity |
|-------|---------|--------------|---------------------|
{chr(10).join(f'| {chain_ids[i]} | {len(sequences[i])} | {bsa_mat[i].max():.0f} | {float(esm_cos[i,i]):.3f} |' for i in range(N))}

---
*PRISM {PRISM_VERSION} · T={_temperature} · ESM-2 150M*
"""

    log("Done.")
    return result_md, "\n".join(status_lines), json.dumps({
        "pdb_id":        pdb_id,
        "chain_ids":     chain_ids,
        "prism_order":   prism_order,
        "sasa_order":    sasa_order,
        "n_chains":      N,
        "is_homomer":    is_homomer,
        "spearman_rho":  spearman_rho,
        "mrr":           mrr,
        "ndcg":          ndcg,
        "score_margin_per_step": score_margins,
    }, indent=2)


# ── Dataset browser ───────────────────────────────────────────────────────────

_dataset: pd.DataFrame | None = None

def _load_dataset() -> pd.DataFrame:
    global _dataset
    if _dataset is not None:
        return _dataset
    try:
        path = hf_hub_download("Sbhat2026/PRISM", "labeled_dataset.tsv",
                               repo_type="space", local_dir="/tmp/prism_data")
        df = pd.read_csv(path, sep="\t")
        df["n_subunits"] = df["n_subunits"].fillna(0).astype(int)
        df["gt_tier"]    = df["gt_tier"].fillna(3).astype(int)
        df["taxid"]      = df["taxid"].fillna(0).astype(int)
        df["name"]       = df["name"].fillna("").astype(str)
        df["source"]     = df["source"].fillna("").astype(str)
        df["type"]       = df["type"].fillna("heteromer").astype(str)
        _dataset = df
        return df
    except Exception as e:
        print(f"[browse] Dataset load failed: {e}")
        return pd.DataFrame()


def search_complexes(query: str, source_filter: str, n_min: int, n_max: int,
                     tier_filter: str, type_filter: str) -> tuple:
    df = _load_dataset()
    if df.empty:
        return pd.DataFrame(columns=["PDB ID","Name","N","Type","Tier","Source","TaxID"]), "Dataset not available"

    mask = pd.Series([True] * len(df), index=df.index)

    if query.strip():
        q = query.strip().upper()
        name_match = df["name"].str.upper().str.contains(q, na=False)
        pdb_match  = df["pdb_id"].astype(str).str.upper().str.contains(q, na=False)
        uid_match  = df["uniprot_ids"].astype(str).str.upper().str.contains(q, na=False)
        mask = mask & (name_match | pdb_match | uid_match)

    if source_filter != "All":
        src_map = {"CORUM": "corum", "Complex Portal": "complex_portal",
                   "RCSB": "rcsb_search", "Experimental GT": "experimental_gt"}
        mask = mask & (df["source"] == src_map.get(source_filter, source_filter))

    if tier_filter != "All":
        mask = mask & (df["gt_tier"] == int(tier_filter.split()[1]))

    if type_filter != "All":
        mask = mask & (df["type"] == type_filter.lower())

    mask = mask & (df["n_subunits"] >= n_min) & (df["n_subunits"] <= n_max)

    result = df[mask].copy()
    # Sort: Tier 1 first, then by name
    result = result.sort_values(["gt_tier", "name"], ascending=[True, True])

    tier_label = {1: "★ Tier 1", 2: "◆ Tier 2", 3: "◇ Tier 3"}
    display = result[["pdb_id","name","n_subunits","type","gt_tier","source","taxid"]].copy()
    display["gt_tier"] = display["gt_tier"].map(tier_label)
    display.columns = ["PDB ID","Name","N","Type","Tier","Source","TaxID"]
    display = display.head(200)

    summary = (f"**{len(result):,} complexes** found"
               + (f" (showing top 200)" if len(result) > 200 else ""))
    return display, summary


def get_pdb_from_selection(evt: gr.SelectData, search_results_df):
    """Return (selected_pdb_textbox, pdb_input_textbox, tabs_switch) on row click."""
    try:
        row = search_results_df.iloc[evt.index[0]]
        pdb = str(row.get("PDB ID", "")).strip()
        pdb = pdb if pdb and pdb != "nan" else ""
    except Exception:
        pdb = ""
    return pdb, pdb, gr.Tabs(selected="predict")


# ── Pre-load default browse results ──────────────────────────────────────────

try:
    _default_browse_df, _default_browse_summary = search_complexes("", "All", 2, 12, "All", "All")
except Exception:
    _default_browse_df = pd.DataFrame(columns=["PDB ID","Name","N","Type","Tier","Source","TaxID"])
    _default_browse_summary = "Loading dataset…"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["2HHB", "A,B,C,D"],
    ["2PTC", "E,I"],
    ["1AON", "A,B,C,D,E,F,G"],
    ["1FIN", "A,B"],
    ["3GBN", "A,H,L"],
]

with gr.Blocks(title="PRISM — Protein Assembly Order Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
# PRISM — Protein Ranking by Interface and Sequence Modeling
**Predicts the sequential order in which protein subunits assemble into a complex.**
GATv2 · ESM-2 150M · BSA · GO coherence · 15,201-complex training corpus · `{PRISM_VERSION}`
""")

    with gr.Tabs(selected="predict") as tabs:

        # ── Tab 1: Predict ─────────────────────────────────────────────────────
        with gr.Tab("🔬 Predict Assembly", id="predict"):
            gr.Markdown("> Enter any PDB ID to predict its assembly order. First run downloads ESM-2 150M (~600 MB).")
            with gr.Row():
                with gr.Column(scale=1):
                    pdb_input   = gr.Textbox(label="PDB ID", placeholder="e.g. 2HHB", max_lines=1)
                    chain_input = gr.Textbox(
                        label="Chain IDs (comma-separated, blank = all)",
                        placeholder="e.g. A,B,C,D  or leave empty",
                        max_lines=1,
                    )
                    run_btn   = gr.Button("Predict Assembly Order", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=[pdb_input, chain_input],
                                label="Quick examples")
                    status_box = gr.Textbox(label="Status", interactive=False, lines=5)

                with gr.Column(scale=2):
                    result_md = gr.Markdown()

            with gr.Accordion("Raw JSON output", open=False):
                json_out = gr.Code(language="json")

            run_btn.click(predict_assembly, inputs=[pdb_input, chain_input],
                          outputs=[result_md, status_box, json_out])
            pdb_input.submit(predict_assembly, inputs=[pdb_input, chain_input],
                             outputs=[result_md, status_box, json_out])

        # ── Tab 2: Browse dataset ──────────────────────────────────────────────
        with gr.Tab("📚 Browse Complexes (15K)", id="browse"):
            gr.Markdown("""
**15,201 labeled protein complexes** from CORUM 5.2, Complex Portal (6 species), and RCSB PDB.
Click any row to load it into the Predict tab automatically.
- **★ Tier 1** — Experimental ground truth (literature-curated assembly order)
- **◆ Tier 2** — BSA-greedy order + confirmed PDB structure
- **◇ Tier 3** — Sequence-length proxy, no structure available
""")
            with gr.Row():
                search_box    = gr.Textbox(label="Search (name, PDB ID, UniProt ID)",
                                           placeholder="e.g. hemoglobin, 2HHB, P68871", scale=3)
                source_dd     = gr.Dropdown(["All","CORUM","Complex Portal","RCSB","Experimental GT"],
                                            value="All", label="Source", scale=1)
                tier_dd       = gr.Dropdown(["All","Tier 1","Tier 2","Tier 3"],
                                            value="All", label="Tier", scale=1)
                type_dd       = gr.Dropdown(["All","heteromer","homomer"],
                                            value="All", label="Type", scale=1)
            with gr.Row():
                n_min_sl = gr.Slider(2, 12, value=2, step=1, label="Min subunits")
                n_max_sl = gr.Slider(2, 12, value=12, step=1, label="Max subunits")
                search_btn = gr.Button("Search", variant="secondary")

            result_count = gr.Markdown(_default_browse_summary)
            browse_table = gr.Dataframe(
                value=_default_browse_df,
                headers=["PDB ID","Name","N","Type","Tier","Source","TaxID"],
                datatype=["str","str","number","str","str","str","number"],
                interactive=False,
                wrap=True,
            )
            selected_pdb = gr.Textbox(label="Selected PDB ID (paste into Predict tab)",
                                      interactive=True)

            search_inputs  = [search_box, source_dd, n_min_sl, n_max_sl, tier_dd, type_dd]
            search_outputs = [browse_table, result_count]

            search_btn.click(search_complexes, inputs=search_inputs, outputs=search_outputs)
            search_box.submit(search_complexes, inputs=search_inputs, outputs=search_outputs)
            # Click row → fill pdb_input + switch to Predict tab
            browse_table.select(get_pdb_from_selection,
                                inputs=[browse_table],
                                outputs=[selected_pdb, pdb_input, tabs])

        # ── Tab 3: About ───────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About", id="about"):
            gr.Markdown(f"""
## PRISM {PRISM_VERSION}

### Training corpus

| Source | Complexes | Notes |
|--------|-----------|-------|
| CORUM 5.2 | 6,531 | Manually curated mammalian complexes |
| RCSB PDB search | 6,489 | High-res heteromers (≤3.5Å, 2–12 chains, pre-2024) |
| Complex Portal | 2,158 | 6 species (human, mouse, yeast, E. coli, rat, fly) |
| Experimental GT | 23 | Literature-curated with known assembly order |
| **Total** | **15,201** | Deduplicated by UniProt frozenset |

### Label tiers

| Tier | Count | Method |
|------|-------|--------|
| ★ Tier 1 (gold) | 23 | Experimental: cryo-EM, native MS, pulse-chase |
| ◆ Tier 2 (silver) | 7,440 | Greedy BSA order on confirmed PDB structure |
| ◇ Tier 3 (bronze) | 7,738 | Sequence-length proxy (no structure) |

### Subunit count distribution (all 15,201 complexes)

| N subunits | Count |
|-----------|-------|
| 2 | 9,092 |
| 3 | 3,440 |
| 4 | 1,350 |
| 5 | 523 |
| 6 | 222 |
| 7 | 146 |
| 8 | 137 |
| 9 | 98 |
| 10 | 93 |
| 11 | 49 |
| 12 | 51 |

### Architecture

GATv2Conv · 3 layers · 4 heads · 128 hidden dim · 167K params
Node features: ESM-2 150M (640-dim) + AlphaFold pLDDT (5-dim) + SASA proxy (1-dim) = 646-dim
Edge features: PPI (STRING) · ESM cosine sim · GO Jaccard · normalised BSA

### Changelog

**v1.1-corpus-15K** *(2026-05-08)*
- Auto-loads corpus on Browse tab open; click row → switches directly to Predict
- Corpus stats table added to About; version stamp on all outputs
- Training corpus fully aggregated (15,201 complexes, 3 sources, 3 label tiers)

**v1.0** *(2026-04-26)*
- Initial release; model trained on Marsh 2013 benchmark (23 experimental complexes)
- Tier definitions and BSA-greedy label assignment pipeline

### Citation

```bibtex
@article{{bhat2025prism,
  title   = {{PRISM: Protein Ranking by Interface and Sequence Modeling}},
  author  = {{Bhat, Siddhant}},
  journal = {{bioRxiv}},
  year    = {{2025}},
  doi     = {{10.1101/2025.PRISM}},
  url     = {{https://huggingface.co/spaces/Sbhat2026/PRISM}}
}}
```

**Metrics (v1.0):** τ=1.000 (Marsh 2013) · τ=0.974 (held-out GT v2) · ρ=0.981 · MRR=0.94 · NDCG=0.96
""")

# ── FastAPI app with /api/mutate ──────────────────────────────────────────────

_AA1_VALID = frozenset("ACDEFGHIKLMNPQRSTVWY")

class MutateRequest(BaseModel):
    pdb_id: str
    chain: str
    position: int
    original_aa: str
    mutant_aa: str
    n_runs: int = 1


def _compute_kl_per_step(wt_steps, mut_steps, eps=1e-9):
    kls = []
    for wt_s, mut_s in zip(wt_steps, mut_steps):
        wt_d  = {c: p for c, p in zip(wt_s["candidates"],  wt_s["probs"])}
        mut_d = {c: p for c, p in zip(mut_s["candidates"], mut_s["probs"])}
        idx = sorted(set(wt_d) | set(mut_d))
        p = np.clip([wt_d.get(i, eps)  for i in idx], eps, None); p /= p.sum()
        q = np.clip([mut_d.get(i, eps) for i in idx], eps, None); q /= q.sum()
        kls.append(round(float(np.sum(p * np.log(p / q))), 4))
    return kls


def _detect_redirect(wt_order, mut_order):
    n = min(len(wt_order), len(mut_order))
    for i in range(n):
        if wt_order[i] != mut_order[i]:
            return {"redirected": True, "first_diff": i, "num_swapped": n - i}
    return {"redirected": False, "first_diff": None, "num_swapped": 0}


def _run_assembly(model, nf, ef, adj, N, is_homomer):
    if is_homomer:
        best_tau, best_order, best_steps = -2.0, None, None
        for seed in range(N):
            order, steps = greedy_prism(model, nf, ef, adj, N, seed)
            tau = circular_tau(order, list(range(N)))
            if tau > best_tau:
                best_tau, best_order, best_steps = tau, order, steps
        return best_order, best_steps
    else:
        return greedy_prism(model, nf, ef, adj, N, 0)


def _build_tensors(emb_mat, plddt_f, bsa_mat):
    N = emb_mat.shape[0]
    norms   = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    esm_cos = (emb_mat / (norms + 1e-8) @ (emb_mat / (norms + 1e-8)).T).clip(-1, 1).astype(np.float32)
    sasa_n  = bsa_mat / (bsa_mat.max() + 1e-8)
    sasa_d  = np.diag(sasa_n).reshape(-1, 1).astype(np.float32)
    nf = torch.tensor(np.hstack([emb_mat, plddt_f, sasa_d]), dtype=torch.float32)
    ef = build_edge_features(
        np.zeros((N, N), dtype=np.float32), esm_cos,
        np.zeros((N, N), dtype=np.float32), sasa_n,
    )
    adj = torch.ones(N, N, dtype=torch.bool)
    return nf, ef, adj


fastapi_app = FastAPI(title="PRISM API")
fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from fastapi.responses import JSONResponse
from fastapi import Request as FastAPIRequest

@fastapi_app.exception_handler(Exception)
async def _generic_exception_handler(request: FastAPIRequest, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}: {exc}"})


@fastapi_app.post("/api/mutate")
def mutate_api(req: MutateRequest):
    pdb_id = req.pdb_id.strip().upper()
    chain  = req.chain.strip().upper()
    pos    = req.position

    if req.original_aa.upper() not in _AA1_VALID:
        raise HTTPException(400, f"Invalid original_aa: {req.original_aa!r}")
    if req.mutant_aa.upper() not in _AA1_VALID:
        raise HTTPException(400, f"Invalid mutant_aa: {req.mutant_aa!r}")

    try:
        pdb_text = download_pdb(pdb_id)
    except Exception as e:
        raise HTTPException(404, f"Failed to download {pdb_id}: {e}")

    seq_map = extract_sequences(pdb_text)
    if chain not in seq_map:
        raise HTTPException(400, f"Chain {chain!r} not found in {pdb_id}. Available: {sorted(seq_map)}")

    chain_ids = sorted(seq_map.keys())
    chain_idx = chain_ids.index(chain)
    orig_seq  = seq_map[chain]
    N = len(chain_ids)

    if not (1 <= pos <= len(orig_seq)):
        raise HTTPException(400, f"Position {pos} out of range [1, {len(orig_seq)}]")
    actual_aa = orig_seq[pos - 1].upper()
    if actual_aa != req.original_aa.upper():
        raise HTTPException(400, f"Position {pos} has {actual_aa!r}, not {req.original_aa.upper()!r}")

    sequences = [seq_map[c] for c in chain_ids]
    is_homomer = len(set(sequences)) == 1

    try:
        bsa_mat = compute_bsa_matrix(pdb_text, chain_ids)
    except Exception as e:
        raise HTTPException(500, f"BSA computation failed: {e}")

    plddt_f = fetch_plddt_for_chains(pdb_id, chain_ids)
    plddt_f[:, 0] /= 100.0; plddt_f[:, 3] /= 100.0; plddt_f[:, 4] /= 100.0

    try:
        model = _load_model()
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {e}")

    try:
        wt_emb = embed_sequences({f"c{i}": s for i, s in enumerate(sequences)})
    except Exception as e:
        raise HTTPException(500, f"ESM embedding failed: {e}")

    wt_mat  = np.stack([wt_emb[f"c{i}"] for i in range(N)])
    wt_nf, wt_ef, wt_adj = _build_tensors(wt_mat, plddt_f, bsa_mat)
    wt_order, wt_steps   = _run_assembly(model, wt_nf, wt_ef, wt_adj, N, is_homomer)

    mut_seqs = sequences[:]
    mut_seqs[chain_idx] = orig_seq[:pos-1] + req.mutant_aa.upper() + orig_seq[pos:]
    try:
        mut_emb_single = embed_sequences({f"c{chain_idx}": mut_seqs[chain_idx]})
    except Exception as e:
        raise HTTPException(500, f"ESM embedding (mutant) failed: {e}")
    mut_mat = wt_mat.copy()
    mut_mat[chain_idx] = mut_emb_single[f"c{chain_idx}"]
    mut_nf, mut_ef, mut_adj = _build_tensors(mut_mat, plddt_f, bsa_mat)
    mut_order, mut_steps    = _run_assembly(model, mut_nf, mut_ef, mut_adj, N, is_homomer)

    sasa_order = sasa_greedy_order(bsa_mat)

    def _tau(pred, gt):
        if len(pred) < 2: return 0.0
        stat, _ = kendalltau([pred.index(x) for x in gt], list(range(len(gt))))
        return float(stat) if not np.isnan(stat) else 0.0

    if is_homomer:
        wt_tau  = circular_tau(wt_order,  list(range(N)))
        mut_tau = circular_tau(mut_order, list(range(N)))
    else:
        wt_tau  = _tau(wt_order,  sasa_order)
        mut_tau = _tau(mut_order, sasa_order)

    def _gt_probs(steps):
        out = []
        for t, step in enumerate(steps):
            cands, ps = step["candidates"], step["probs"]
            nxt = sasa_order[t+1] if t+1 < len(sasa_order) else None
            if nxt is not None and nxt in cands:
                out.append(round(float(ps[cands.index(nxt)]), 4))
        return out

    wt_gt  = _gt_probs(wt_steps)
    mut_gt = _gt_probs(mut_steps)

    def _top1(steps):
        hits = sum(1 for t, s in enumerate(steps)
                   if t+1 < len(sasa_order) and s["chosen"] == sasa_order[t+1])
        return round(hits / max(len(steps), 1), 4)

    redirect = _detect_redirect(wt_order, mut_order)
    kl       = _compute_kl_per_step(wt_steps, mut_steps)

    return {
        "pdb_id":      pdb_id, "chain": chain, "position": pos,
        "original_aa": req.original_aa.upper(), "mutant_aa": req.mutant_aa.upper(),
        "wt":  {"tau": wt_tau,  "top1_acc": _top1(wt_steps),  "pred_order": wt_order,  "gt_prob_per_step": wt_gt},
        "mut": {"tau": mut_tau, "top1_acc": _top1(mut_steps), "pred_order": mut_order, "gt_prob_per_step": mut_gt},
        "deltas": {
            "tau": round(mut_tau - wt_tau, 4),
            "top1_acc": round(_top1(mut_steps) - _top1(wt_steps), 4),
            "gt_prob_per_step": [round(m-w, 4) for w, m in zip(wt_gt, mut_gt)],
        },
        "kl_per_step": kl,
        "redirected":  redirect["redirected"],
        "redirect_info": redirect,
        "go_delta": {"gained": [], "lost": []},
        "subunit_name": chain, "uniprot": chain,
    }


# Mount Gradio on FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
