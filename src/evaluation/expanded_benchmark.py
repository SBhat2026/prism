"""
Expanded benchmark: Marsh 2013 (held-out) + experimental_gt_v2 complexes.

Loads the best GNN checkpoint, runs tau / top1 / mean_confidence / early_tau
on both sets, prints a combined table, and writes results to
results/expanded_benchmark/results.json and results/expanded_benchmark/report.png.

Usage:
    python src/evaluation/expanded_benchmark.py
    python src/evaluation/expanded_benchmark.py --no_v2     # Marsh only
    python src/evaluation/expanded_benchmark.py --no_marsh  # v2 only
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.scorer_gnn import AssemblyScorer, build_edge_features
from src.training.dataset import ComplexData
from src.evaluation.metrics import evaluate_complex, circular_tau, kendall_tau
from src.data_prep.plddt_fetcher import PLDDT_NODE_DIM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_PATH   = ROOT / "results" / "gnn_training" / "best_gnn.pt"
OUT_DIR     = ROOT / "results" / "expanded_benchmark"
EXP_DIR     = ROOT / "data" / "experimental_complexes"
PLDDT_CACHE = ROOT / "data" / "plddt_cache"
SASA_CACHE  = ROOT / "data" / "sasa_cache"

MARSH_BENCHMARK = {
    "2HHB": {"chains": ["A","B","C","D"],         "gt_order": [0,1,3,2],       "type": "heteromer"},
    "1BRS": {"chains": ["A","B","C","D","E","F"],  "gt_order": [0,3,1,4,2,5],  "type": "heteromer"},
    "1AY7": {"chains": ["A","B"],                  "gt_order": [0,1],           "type": "heteromer"},
    "1TGS": {"chains": ["Z","I"],                  "gt_order": [0,1],           "type": "heteromer"},
    "2PTC": {"chains": ["E","I"],                  "gt_order": [0,1],           "type": "heteromer"},
    "1A2K": {"chains": ["A","B","C","D","E"],      "gt_order": [1,0,3,2,4],    "type": "heteromer"},
    "1SBB": {"chains": ["A","B","C","D"],          "gt_order": [2,0,3,1],      "type": "heteromer"},
    "3GBN": {"chains": ["A","H","L"],              "gt_order": [0,1,2],         "type": "heteromer"},
    "1AON": {"chains": list("ABCDEFG"),            "gt_order": list(range(7)),  "type": "homomer"},
    "1GRU": {"chains": list("ABCDEFG"),            "gt_order": list(range(7)),  "type": "homomer"},
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_marsh_complex(pdb_id: str, meta: dict) -> ComplexData:
    from src.data_prep.plddt_fetcher import get_chain_uniprot_ids, fetch_plddt_stats, stats_to_node_vec

    MARSH_DIR  = ROOT / "data" / "complexes" / "marsh2013"
    EMBED_150M = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t30_150M_UR50D"
    EMBED_8M   = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t6_8M_UR50D"

    use_150m = EMBED_150M.exists()
    embed_dir = EMBED_150M if use_150m else EMBED_8M
    esm_dim   = 640 if use_150m else 320
    emb_data  = np.load(str(embed_dir / "embeddings.npz"))

    manifest_path = MARSH_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    chains = meta["chains"]
    N = len(chains)
    seqs = [manifest.get(pdb_id, {}).get("sequences", {}).get(c, "M"*50) for c in chains]

    embs = np.array([emb_data.get(f"{pdb_id}_{c}", np.zeros(esm_dim)) for c in chains], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / (norms + 1e-8)
    esm_mat = (embs_n @ embs_n.T).astype(np.float32).clip(-1, 1)

    sasa_cache = SASA_CACHE / f"{pdb_id}_sasa.npy"
    sasa_cache.parent.mkdir(parents=True, exist_ok=True)
    if sasa_cache.exists():
        sasa_mat = np.load(str(sasa_cache)).astype(np.float32)
        if sasa_mat.shape != (N, N):
            sasa_cache.unlink()
            sasa_mat = None
    else:
        sasa_mat = None
    if sasa_mat is None:
        pdb_path = str(MARSH_DIR / f"{pdb_id}.pdb")
        sasa_mat = np.zeros((N, N), dtype=np.float32)
        if Path(pdb_path).exists():
            try:
                from src.data_prep.fast_sasa import compute_bsa_matrix
                sasa_mat = compute_bsa_matrix(pdb_path, chains).astype(np.float32)
                np.save(str(sasa_cache), sasa_mat)
            except Exception as e:
                print(f"  [{pdb_id}] SASA failed: {e}")

    plddt_cache = PLDDT_CACHE / f"{pdb_id}_plddt.npy"
    if plddt_cache.exists():
        plddt = np.load(str(plddt_cache)).astype(np.float32)
    else:
        uniprot_map = get_chain_uniprot_ids(pdb_id)
        vecs = []
        for c in chains:
            uid = uniprot_map.get(c, "")
            stats = fetch_plddt_stats(uid, PLDDT_CACHE) if uid else None
            vecs.append(stats_to_node_vec(stats))
        plddt = np.stack(vecs).astype(np.float32)
        np.save(str(plddt_cache), plddt)

    ppi_path = ROOT / "data" / "string" / f"{pdb_id}_ppi.npy"
    ppi_mat = np.load(str(ppi_path)).astype(np.float32) if ppi_path.exists() \
              else np.zeros((N, N), dtype=np.float32)

    return ComplexData(
        pdb_id=pdb_id,
        complex_type=meta["type"],
        chain_ids=chains,
        sequences=seqs,
        gt_order=meta["gt_order"],
        sasa_matrix=sasa_mat,
        esm_matrix=esm_mat,
        go_matrix=np.zeros((N, N), dtype=np.float32),
        ppi_matrix=ppi_mat,
        esm_embeddings=embs,
        plddt=plddt,
        gt_source="experimental",
    )


def load_marsh_dataset() -> list[ComplexData]:
    dataset = []
    for pdb_id, meta in MARSH_BENCHMARK.items():
        try:
            cd = _load_marsh_complex(pdb_id, meta)
            dataset.append(cd)
        except Exception as e:
            print(f"  [marsh] {pdb_id} failed: {e}")
    print(f"[data] loaded {len(dataset)} Marsh 2013 complexes")
    return dataset


def load_v2_dataset() -> list[ComplexData]:
    index_path = EXP_DIR / "index.json"
    if not index_path.exists():
        print("[data] no experimental_complexes/index.json — run build_experimental_dataset.py first")
        return []

    index = json.loads(index_path.read_text())
    dataset = []
    for meta in index:
        pdb_id  = meta["pdb_id"]
        cplx_dir = EXP_DIR / pdb_id
        emb_path  = cplx_dir / "emb.npy"
        sasa_path = cplx_dir / "sasa.npy"
        if not (emb_path.exists() and sasa_path.exists()):
            print(f"  [v2] {pdb_id} missing files, skipping")
            continue
        chains = meta["chain_ids"]
        N = len(chains)
        emb_mat  = np.load(str(emb_path)).astype(np.float32)
        sasa_mat = np.load(str(sasa_path)).astype(np.float32)
        plddt    = np.load(str(cplx_dir / "plddt.npy")).astype(np.float32) \
                   if (cplx_dir / "plddt.npy").exists() \
                   else np.full((N, PLDDT_NODE_DIM), 70.0, dtype=np.float32)
        ppi_mat  = np.load(str(cplx_dir / "ppi.npy")).astype(np.float32) \
                   if (cplx_dir / "ppi.npy").exists() \
                   else np.zeros((N, N), dtype=np.float32)
        norms  = np.linalg.norm(emb_mat, axis=1, keepdims=True)
        emb_n  = emb_mat / (norms + 1e-8)
        esm_cos= (emb_n @ emb_n.T).clip(-1, 1).astype(np.float32)
        dataset.append(ComplexData(
            pdb_id=pdb_id,
            complex_type=meta["complex_type"],
            chain_ids=chains,
            sequences=meta.get("sequences", [""] * N),
            gt_order=meta["gt_order"],
            sasa_matrix=sasa_mat,
            esm_matrix=esm_cos,
            go_matrix=np.zeros((N, N), dtype=np.float32),
            ppi_matrix=ppi_mat,
            esm_embeddings=emb_mat,
            plddt=plddt,
            gt_source="experimental",
        ))
    print(f"[data] loaded {len(dataset)} experimental_v2 complexes")
    return dataset


# ── GNN precompute ────────────────────────────────────────────────────────────

def precompute(cdata: ComplexData):
    N = len(cdata.chain_ids)
    sasa = cdata.sasa_matrix / (cdata.sasa_matrix.max() + 1e-8)

    if cdata.plddt is not None:
        plddt_feats = cdata.plddt.astype(np.float32).copy()
        plddt_feats[:, 0] /= 100.0
        plddt_feats[:, 3] /= 100.0
        plddt_feats[:, 4] /= 100.0
    else:
        plddt_feats = np.zeros((N, PLDDT_NODE_DIM), dtype=np.float32)
        plddt_feats[:, 0] = 0.70

    sasa_diag = np.diag(sasa).astype(np.float32)
    node_feats = np.hstack([
        cdata.esm_embeddings.astype(np.float32),
        plddt_feats,
        sasa_diag.reshape(-1, 1),
    ])
    node_feats_t = torch.tensor(node_feats, dtype=torch.float32)
    edge_feats_t = build_edge_features(cdata.ppi_matrix, cdata.esm_matrix, cdata.go_matrix, sasa)
    adj = torch.ones(N, N, dtype=torch.bool)
    return node_feats_t, edge_feats_t, adj


# ── Report ────────────────────────────────────────────────────────────────────

def _fmt(v):
    return f"{v:.4f}"


def print_table(rows: list[dict], title: str):
    header = f"{'PDB':>6}  {'type':>10}  {'source':>8}  {'τ':>7}  {'top1':>7}  {'conf':>7}  {'eτ':>7}  {'N':>3}"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    print(header)
    print('-'*60)
    for r in rows:
        print(f"  {r['pdb_id']:>6}  {r['complex_type']:>10}  {r['source']:>8}  "
              f"{_fmt(r['tau']):>7}  {_fmt(r['top1_accuracy']):>7}  "
              f"{_fmt(r['mean_confidence']):>7}  {_fmt(r['early_step_tau']):>7}  {r['n_chains']:>3}")
    taus  = [r['tau'] for r in rows]
    top1s = [r['top1_accuracy'] for r in rows]
    confs = [r['mean_confidence'] for r in rows]
    etaus = [r['early_step_tau'] for r in rows]
    print('-'*60)
    print(f"  {'mean':>6}  {'':>10}  {'':>8}  "
          f"{_fmt(np.mean(taus)):>7}  {_fmt(np.mean(top1s)):>7}  "
          f"{_fmt(np.mean(confs)):>7}  {_fmt(np.mean(etaus)):>7}  {len(rows):>3}")
    print('='*60)


def print_size_stratified(rows: list[dict]):
    """Print per-N and per-bucket performance summary."""
    BUCKETS = [
        ("small  (N=2–4)",   lambda n: n <= 4),
        ("medium (N=5–9)",   lambda n: 5 <= n <= 9),
        ("large  (N=10–15)", lambda n: n >= 10),
    ]

    print(f"\n{'='*60}")
    print("  Performance by subunit count")
    print('='*60)
    print(f"  {'bucket':20s}  {'n':>4}  {'τ':>7}  {'top1':>7}  {'conf':>7}  {'eτ':>7}")
    print('-'*60)

    # Per-N rows
    per_n: dict[int, list] = {}
    for r in rows:
        per_n.setdefault(r["n_chains"], []).append(r)
    for n in sorted(per_n):
        rs = per_n[n]
        print(f"  {'N='+str(n):20s}  {len(rs):>4}  "
              f"{np.mean([r['tau'] for r in rs]):>7.4f}  "
              f"{np.mean([r['top1_accuracy'] for r in rs]):>7.4f}  "
              f"{np.mean([r['mean_confidence'] for r in rs]):>7.4f}  "
              f"{np.mean([r['early_step_tau'] for r in rs]):>7.4f}")

    print('-'*60)
    for label, fn in BUCKETS:
        rs = [r for r in rows if fn(r["n_chains"])]
        if not rs:
            continue
        print(f"  {label:20s}  {len(rs):>4}  "
              f"{np.mean([r['tau'] for r in rs]):>7.4f}  "
              f"{np.mean([r['top1_accuracy'] for r in rs]):>7.4f}  "
              f"{np.mean([r['mean_confidence'] for r in rs]):>7.4f}  "
              f"{np.mean([r['early_step_tau'] for r in rs]):>7.4f}")
    print('='*60)


def plot_report(marsh_rows, v2_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["tau", "top1_accuracy", "mean_confidence", "early_step_tau"]
    labels  = ["Kendall's τ", "Top-1 Acc", "Mean Confidence", "Early-step τ"]

    all_rows = [(r, "Marsh2013") for r in marsh_rows] + [(r, "ExpV2") for r in v2_rows]
    pdb_ids  = [r["pdb_id"] for r, _ in all_rows]
    colors   = ["#4C72B0" if s == "Marsh2013" else "#DD8452" for _, s in all_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Expanded Benchmark: Marsh 2013 + Experimental GT v2", fontsize=13, fontweight="bold")

    for ax, metric, label in zip(axes.flat, metrics, labels):
        vals = [r[metric] for r, _ in all_rows]
        bars = ax.bar(range(len(pdb_ids)), vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(len(pdb_ids)))
        ax.set_xticklabels(pdb_ids, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.axhline(np.mean(vals), color="black", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"mean={np.mean(vals):.3f}")
        if marsh_rows:
            marsh_mean = np.mean([r[metric] for r in marsh_rows])
            ax.axhline(marsh_mean, color="#4C72B0", linestyle=":", linewidth=1, alpha=0.6,
                       label=f"marsh={marsh_mean:.3f}")
        if v2_rows:
            v2_mean = np.mean([r[metric] for r in v2_rows])
            ax.axhline(v2_mean, color="#DD8452", linestyle=":", linewidth=1, alpha=0.6,
                       label=f"v2={v2_mean:.3f}")
        ax.legend(fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4C72B0", label="Marsh 2013"),
                       Patch(facecolor="#DD8452", label="Exp GT v2")]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "report.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved → {out}")

    # Size-stratified plot
    combined = marsh_rows + v2_rows
    if not combined:
        return
    per_n: dict[int, list] = {}
    for r in combined:
        per_n.setdefault(r["n_chains"], []).append(r)
    ns = sorted(per_n)
    n_labels = [f"N={n}\n(k={len(per_n[n])})" for n in ns]

    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig2.suptitle("PRISM Performance by Subunit Count", fontsize=12, fontweight="bold")
    SIZE_COLORS = {n: plt.cm.viridis(i / max(len(ns)-1, 1)) for i, n in enumerate(ns)}

    for ax2, metric, label in zip(axes2, metrics, labels):
        means = [np.mean([r[metric] for r in per_n[n]]) for n in ns]
        bar_colors = [SIZE_COLORS[n] for n in ns]
        ax2.bar(range(len(ns)), means, color=bar_colors, alpha=0.85, edgecolor="white")
        ax2.set_xticks(range(len(ns)))
        ax2.set_xticklabels(n_labels, fontsize=8)
        ax2.set_title(label, fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.axhline(np.mean([r[metric] for r in combined]), color="black",
                    linestyle="--", linewidth=1, alpha=0.5, label="overall mean")
        ax2.legend(fontsize=7)

    plt.tight_layout()
    out2 = OUT_DIR / "size_stratified.png"
    plt.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved → {out2}")


# ── Main ─────────────────────────────────────────────────────────────────────

def _load_go_matrix(pdb_id: str, n: int) -> np.ndarray:
    """Load cached GO matrix if available, else zeros."""
    go_cache = ROOT / "data" / "go_cache" / f"{pdb_id}_go.npy"
    if go_cache.exists():
        try:
            mat = np.load(str(go_cache)).astype(np.float32)
            if mat.shape == (n, n):
                return mat
        except Exception:
            pass
    return np.zeros((n, n), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_v2",    action="store_true", help="Skip experimental v2")
    parser.add_argument("--no_marsh", action="store_true", help="Skip Marsh 2013")
    parser.add_argument("--ckpt",     default=str(CKPT_PATH), help="Checkpoint path")
    parser.add_argument("--no_go",    action="store_true", help="Skip GO matrix loading")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}")
        sys.exit(1)

    state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    # Support both plain state_dict and wrapped dict format
    if "model_state_dict" in state:
        sd       = state["model_state_dict"]
        node_dim = state.get("node_dim", 646)
        meta_str = (f"  epoch={state.get('epoch','?')}  "
                    f"gt_prob={state.get('gt_prob',0.0):.4f}  "
                    f"τ={state.get('mean_tau',0.0):.4f}")
    else:
        sd       = state
        # Infer node_dim and hidden from checkpoint shapes
        node_dim = sd["node_proj.weight"].shape[1]
        meta_str = ""

    # Infer hidden from checkpoint (node_proj maps node_dim → hidden)
    hidden_dim = sd["node_proj.weight"].shape[0]

    model = AssemblyScorer(node_dim=node_dim, edge_dim=4, hidden=hidden_dim).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] loaded {ckpt_path.name} (node_dim={node_dim}, hidden={hidden_dim}){meta_str}")

    # Load calibrated temperature if available
    temp_path = ROOT / "results" / "gnn_training" / "temperature.json"
    temperature = 1.0
    if temp_path.exists():
        try:
            temperature = json.load(open(temp_path))["temperature"]
            print(f"[calib] using temperature T={temperature:.3f}")
        except Exception:
            pass

    marsh_rows, v2_rows = [], []

    if not args.no_marsh:
        marsh_data = load_marsh_dataset()
        if not args.no_go:
            for cd in marsh_data:
                cd.go_matrix = _load_go_matrix(cd.pdb_id, len(cd.chain_ids))
        print("[eval] Marsh 2013...")
        for cd in marsh_data:
            nf, ef, adj = precompute(cd)
            m = evaluate_complex(model, cd, nf, ef, adj, device=DEVICE)
            marsh_rows.append({
                "pdb_id": cd.pdb_id,
                "complex_type": cd.complex_type,
                "source": "Marsh2013",
                "n_chains": len(cd.chain_ids),
                "temperature": temperature,
                **m,
            })

    if not args.no_v2:
        v2_data = load_v2_dataset()
        if not args.no_go:
            for cd in v2_data:
                cd.go_matrix = _load_go_matrix(cd.pdb_id, len(cd.chain_ids))
        print("[eval] Experimental GT v2...")
        for cd in v2_data:
            nf, ef, adj = precompute(cd)
            m = evaluate_complex(model, cd, nf, ef, adj, device=DEVICE)
            v2_rows.append({
                "pdb_id": cd.pdb_id,
                "complex_type": cd.complex_type,
                "source": "ExpV2",
                "n_chains": len(cd.chain_ids),
                "temperature": temperature,
                **m,
            })

    if marsh_rows:
        print_table(marsh_rows, "Marsh 2013 (held-out)")
    if v2_rows:
        print_table(v2_rows, "Experimental GT v2")

    all_rows = marsh_rows + v2_rows
    if all_rows:
        print_table(all_rows, "Combined")
        print_size_stratified(all_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _summary(rows):
        if not rows: return {}
        return {
            "mean_tau":        float(np.mean([r["tau"] for r in rows])),
            "mean_top1":       float(np.mean([r["top1_accuracy"] for r in rows])),
            "mean_confidence": float(np.mean([r["mean_confidence"] for r in rows])),
            "mean_early_tau":  float(np.mean([r["early_step_tau"] for r in rows])),
            "n": len(rows),
        }

    # Size-bucket summaries for JSON
    buckets = {"small_2_4": [], "medium_5_9": [], "large_10_plus": []}
    for r in all_rows:
        n = r["n_chains"]
        if n <= 4:      buckets["small_2_4"].append(r)
        elif n <= 9:    buckets["medium_5_9"].append(r)
        else:           buckets["large_10_plus"].append(r)

    results = {
        "marsh":          marsh_rows,
        "v2":             v2_rows,
        "marsh_summary":  _summary(marsh_rows),
        "v2_summary":     _summary(v2_rows),
        "combined_summary": _summary(all_rows),
        "by_size": {k: _summary(v) for k, v in buckets.items()},
    }
    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out_json}")

    if marsh_rows or v2_rows:
        plot_report(marsh_rows, v2_rows)


if __name__ == "__main__":
    main()
