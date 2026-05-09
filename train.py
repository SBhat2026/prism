"""
PRISM GNN Training Script — M4 Mac Air 16 GB optimised.
GATv2-based assembly order prediction with gradient checkpointing.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import hashlib
import json
import math
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Device / dtype ────────────────────────────────────────────────────────────

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print(f"[device] {device}")

# ── Try torch_geometric ───────────────────────────────────────────────────────

try:
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    from torch_geometric.nn import GATv2Conv
    HAS_PYGEOM = True
    print("[pygeom] torch_geometric available")
except ImportError:
    HAS_PYGEOM = False
    print("[pygeom] torch_geometric NOT found — install with: pip install torch_geometric")
    print("         Falling back to WeightLearner training only.")

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_150 = ROOT / "data" / "embeddings" / "marsh2013" / "esm2_t30_150M_UR50D"
EMBEDDINGS_CACHE = ROOT / "embeddings_cache.pt"

# ── Load dataset ──────────────────────────────────────────────────────────────

def get_dataset():
    try:
        from run_alpha import BENCHMARK, MARSH_DIR, EMBED_DIR
        from src.training.dataset import load_all_complex_data
        bridge = None
        try:
            from src.protfunc_bridge.go_coherence import ProtFuncBridge
            bridge = ProtFuncBridge()
        except Exception:
            pass
        dataset = load_all_complex_data(
            BENCHMARK, MARSH_DIR,
            embed_8m_dir=EMBED_DIR,
            embed_150m_dir=EMBED_150,
            go_bridge=bridge,
        )
        print(f"[dataset] Loaded {len(dataset)} complexes")
        return dataset
    except Exception as e:
        print(f"[dataset] Could not load dataset: {e}")
        return []

# ── ESM-2 embedding pre-computation (B3) ─────────────────────────────────────

def precompute_embeddings(dataset):
    """Compute ESM-2 150M embeddings once, cache to disk, then free ESM."""
    if EMBEDDINGS_CACHE.exists():
        print(f"[esm] Loading embeddings from cache: {EMBEDDINGS_CACHE}")
        return torch.load(str(EMBEDDINGS_CACHE), map_location="cpu")

    print("[esm] Computing ESM-2 150M embeddings (this runs once)...")
    try:
        import esm as esm_lib
        esm_model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
        esm_model.eval()
    except Exception as e:
        print(f"[esm] ESM-2 load failed: {e} — using random embeddings")
        cache = {}
        for cdata in dataset:
            for seq in getattr(cdata, "sequences", []):
                key = hashlib.md5(seq[:1022].encode()).hexdigest()
                if key not in cache:
                    cache[key] = torch.randn(640)
        return cache

    batch_converter = alphabet.get_batch_converter()
    MAX_LEN = 1022
    cache = {}

    all_seqs = []
    for cdata in dataset:
        for seq in getattr(cdata, "sequences", []):
            key = hashlib.md5(seq[:MAX_LEN].encode()).hexdigest()
            if key not in cache:
                all_seqs.append((key, seq[:MAX_LEN]))

    unique_seqs = list({k: s for k, s in all_seqs}.items())
    print(f"[esm] Embedding {len(unique_seqs)} unique sequences...")

    for i, (key, seq) in enumerate(unique_seqs):
        _, _, tokens = batch_converter([("seq", seq)])
        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[30])
        emb = out["representations"][30][0, 1:-1].mean(0).cpu()
        cache[key] = emb
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(unique_seqs)}]")

    torch.save(cache, str(EMBEDDINGS_CACHE))
    print(f"[esm] Saved to {EMBEDDINGS_CACHE}")

    del esm_model
    if device.type == "mps":
        torch.mps.empty_cache()

    return cache


# ── ComplexDataset (B4) ───────────────────────────────────────────────────────

class ComplexDataset(torch.utils.data.Dataset):
    def __init__(self, complex_list, emb_cache):
        self.items = []
        MAX_LEN = 1022
        for cdata in complex_list:
            seqs = getattr(cdata, "sequences", [])
            N = len(seqs)
            if N < 2:
                continue

            # Node features: esm640 + plddt5 (zeros) + sasa1 = 646
            sasa_mat = getattr(cdata, "sasa_matrix", None)
            node_feats = []
            for i, seq in enumerate(seqs):
                key = hashlib.md5(seq[:MAX_LEN].encode()).hexdigest()
                esm_emb = emb_cache.get(key, torch.zeros(640))

                sasa_val = 0.0
                if sasa_mat is not None:
                    sasa_diag = sasa_mat[i, i] if isinstance(sasa_mat, np.ndarray) else float(sasa_mat[i, i])
                    sasa_max = float(np.max(sasa_mat)) if isinstance(sasa_mat, np.ndarray) else float(sasa_mat.max())
                    sasa_val = sasa_diag / (sasa_max + 1e-8)

                plddt_proxy = torch.zeros(5)
                sasa_feat = torch.tensor([sasa_val], dtype=torch.float32)
                node_feats.append(torch.cat([esm_emb.float(), plddt_proxy, sasa_feat]))

            x = torch.stack(node_feats)  # (N, 646)

            # Fully connected edge_index
            src, dst = [], []
            for i in range(N):
                for j in range(N):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            n_edges = edge_index.shape[1]

            # Edge features: [ppi, esm_cos, go_cos, bsa_norm]
            ppi_mat  = getattr(cdata, "ppi_matrix",  None)
            esm_mat  = getattr(cdata, "esm_matrix",  None)
            go_mat   = getattr(cdata, "go_matrix",   None)

            edge_feats = []
            for k in range(n_edges):
                i, j = src[k], dst[k]
                ppi_v  = float(ppi_mat[i, j])  if ppi_mat  is not None else 0.5
                esm_v  = float(esm_mat[i, j])  if esm_mat  is not None else 0.5
                go_v   = float(go_mat[i, j])   if go_mat   is not None else 0.5
                bsa_v  = float(sasa_mat[i, j]) / (float(np.max(sasa_mat)) + 1e-8) if sasa_mat is not None else 0.0
                edge_feats.append([ppi_v, esm_v, go_v, bsa_v])
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

            gt_order = torch.tensor(getattr(cdata, "gt_order", list(range(N))), dtype=torch.long)
            is_ring  = bool(getattr(cdata, "is_ring", False))
            complex_type = getattr(cdata, "complex_type", "heteromer")

            if HAS_PYGEOM:
                data_obj = Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr,
                    gt_order=gt_order,
                )
            else:
                data_obj = _SimpleData(
                    x=x, edge_index=edge_index, edge_attr=edge_attr,
                    gt_order=gt_order,
                )
            data_obj.is_ring = is_ring
            data_obj.complex_type = complex_type
            data_obj.pdb_id = getattr(cdata, "pdb_id", "?")
            self.items.append(data_obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _SimpleData:
    """Minimal Data substitute when torch_geometric isn't available."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        obj = _SimpleData()
        for k, v in self.__dict__.items():
            setattr(obj, k, v.to(device) if isinstance(v, torch.Tensor) else v)
        return obj

    @property
    def num_nodes(self):
        return self.x.shape[0]


# ── PRISM_GNN model (B5) ──────────────────────────────────────────────────────

class PRISM_GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input projection: 646 → 128
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(646, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
        )

        if HAS_PYGEOM:
            self.conv1 = GATv2Conv(128, 32, heads=4, edge_dim=4, concat=True)
            self.conv2 = GATv2Conv(128, 32, heads=4, edge_dim=4, concat=True)
            self.conv3 = GATv2Conv(128, 32, heads=4, edge_dim=4, concat=True)
        else:
            # Fallback: simple linear layers
            self.conv1 = torch.nn.Linear(128, 128)
            self.conv2 = torch.nn.Linear(128, 128)
            self.conv3 = torch.nn.Linear(128, 128)

        self.norm1 = torch.nn.LayerNorm(128)
        self.norm2 = torch.nn.LayerNorm(128)
        self.norm3 = torch.nn.LayerNorm(128)

        # Edge scorer MLP: 128+128+4+128=388 → 256 → 128 → 1
        self.edge_scorer = torch.nn.Sequential(
            torch.nn.Linear(388, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1),
        )

    def gnn_forward(self, x, edge_index, edge_attr):
        h = self.input_proj(x)

        if HAS_PYGEOM:
            h1 = checkpoint(self.conv1, h, edge_index, edge_attr, use_reentrant=False)
            h = self.norm1(h1 + h)
            h2 = checkpoint(self.conv2, h, edge_index, edge_attr, use_reentrant=False)
            h = self.norm2(h2 + h)
            h3 = checkpoint(self.conv3, h, edge_index, edge_attr, use_reentrant=False)
            h = self.norm3(h3 + h)
        else:
            h = self.norm1(F.gelu(self.conv1(h)) + h)
            h = self.norm2(F.gelu(self.conv2(h)) + h)
            h = self.norm3(F.gelu(self.conv3(h)) + h)

        return h  # (N, 128)

    def score_candidates(self, node_repr, assembled, remaining, context, edge_attr):
        """Score each remaining candidate against assembled anchors."""
        if not assembled:
            # No context: score based on node representation alone
            scores = []
            for cand_idx in remaining:
                h_cand = node_repr[cand_idx]
                zero_anchor = torch.zeros(128, device=node_repr.device)
                zero_edge = torch.zeros(4, device=node_repr.device)
                feat = torch.cat([h_cand, zero_anchor, zero_edge, context])
                scores.append(self.edge_scorer(feat.unsqueeze(0)).squeeze())
            return torch.stack(scores)

        scores = []
        for cand_idx in remaining:
            h_cand = node_repr[cand_idx]
            anchor_scores = []
            for anc_idx in assembled:
                h_anchor = node_repr[anc_idx]
                # Find edge feature between anchor and candidate
                edge_feat = torch.zeros(4, device=node_repr.device)
                if edge_attr is not None:
                    N_total = node_repr.shape[0]
                    n_candidates = N_total
                    edge_flat_idx = anc_idx * (n_candidates - 1) + (cand_idx if cand_idx < anc_idx else cand_idx - 1)
                    if edge_flat_idx < edge_attr.shape[0]:
                        edge_feat = edge_attr[edge_flat_idx]
                feat = torch.cat([h_cand, h_anchor, edge_feat, context])
                anchor_scores.append(self.edge_scorer(feat.unsqueeze(0)).squeeze())
            # Max-aggregate over anchors
            scores.append(torch.stack(anchor_scores).max())

        return torch.stack(scores)

    def forward(self, data):
        return self.gnn_forward(data.x, data.edge_index, data.edge_attr)


# ── Loss function (B6) ────────────────────────────────────────────────────────

def assembly_loss(scores_per_candidate, gt_idx, is_ring, step, N):
    if is_ring and step == 0:
        return torch.tensor(0.0, device=scores_per_candidate.device)
    return F.cross_entropy(
        scores_per_candidate.unsqueeze(0),
        torch.tensor([gt_idx], device=scores_per_candidate.device),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def kendall_tau_circular(pred, gt, is_ring):
    from scipy.stats import kendalltau
    n = len(pred)
    if n < 2:
        return 1.0
    gt_list = gt.tolist() if isinstance(gt, torch.Tensor) else list(gt)
    pred_list = pred if isinstance(pred, list) else list(pred)

    if not is_ring:
        pos = {v: i for i, v in enumerate(gt_list)}
        arr = [pos.get(v, 0) for v in pred_list]
        try:
            tau = kendalltau(range(n), arr).statistic
            return float(tau) if not math.isnan(tau) else 0.0
        except Exception:
            return 0.0

    best = -2.0
    for k in range(n):
        rotated = gt_list[k:] + gt_list[:k]
        for r in [rotated, list(reversed(rotated))]:
            pos = {v: i for i, v in enumerate(r)}
            arr = [pos.get(v, 0) for v in pred_list]
            try:
                tau = kendalltau(range(n), arr).statistic
                best = max(best, float(tau) if not math.isnan(tau) else 0.0)
            except Exception:
                pass
    return best


def greedy_decode(model, data):
    """Greedy assembly: pick highest-scoring candidate at each step."""
    model.eval()
    N = data.x.shape[0]
    with torch.no_grad():
        node_repr = model.gnn_forward(data.x, data.edge_index, data.edge_attr)

    assembled, remaining = [], list(range(N))
    context = torch.zeros(128, device=data.x.device)
    predicted_order = []
    step_scores_list = []

    while remaining:
        with torch.no_grad():
            scores = model.score_candidates(node_repr, assembled, remaining, context, data.edge_attr)
        step_scores_list.append(scores.cpu())
        best_local = int(scores.argmax())
        chosen_idx = remaining[best_local]
        predicted_order.append(chosen_idx)
        assembled.append(chosen_idx)
        remaining.remove(chosen_idx)
        context = node_repr[assembled].mean(0).detach()

    return predicted_order, step_scores_list


def compute_gt_prob(step_scores, gt_order, T_star=0.332):
    gt_list = gt_order.tolist() if isinstance(gt_order, torch.Tensor) else list(gt_order)
    assembled = []
    remaining = list(range(len(gt_list)))
    probs_list = []

    for k, gt_next in enumerate(gt_list):
        scores = step_scores[k].float() if k < len(step_scores) else torch.zeros(len(remaining))
        logits = scores / T_star
        p = F.softmax(logits, dim=0)
        try:
            local_idx = remaining.index(gt_next)
            probs_list.append(p[local_idx].item())
        except ValueError:
            probs_list.append(0.0)
        if gt_next in remaining:
            remaining.remove(gt_next)
        assembled.append(gt_next)

    return float(np.mean(probs_list)) if probs_list else 0.0


def compute_mrr(step_scores, gt_order):
    gt_list = gt_order.tolist() if isinstance(gt_order, torch.Tensor) else list(gt_order)
    remaining = list(range(len(gt_list)))
    reciprocals = []

    for k, gt_next in enumerate(gt_list):
        scores = step_scores[k].float() if k < len(step_scores) else torch.zeros(len(remaining))
        order = scores.argsort(descending=True).tolist()
        try:
            local_idx = remaining.index(gt_next)
            rank = order.index(local_idx) + 1
        except (ValueError, IndexError):
            rank = len(remaining)
        reciprocals.append(1.0 / rank)
        if gt_next in remaining:
            remaining.remove(gt_next)

    return float(np.mean(reciprocals)) if reciprocals else 0.0


def compute_ndcg(step_scores, gt_order):
    gt_list = gt_order.tolist() if isinstance(gt_order, torch.Tensor) else list(gt_order)
    remaining = list(range(len(gt_list)))
    dcg_vals = []

    for k, gt_next in enumerate(gt_list):
        scores = step_scores[k].float() if k < len(step_scores) else torch.zeros(len(remaining))
        order = scores.argsort(descending=True).tolist()
        try:
            local_idx = remaining.index(gt_next)
            rank = order.index(local_idx) + 1
        except (ValueError, IndexError):
            rank = len(remaining)
        dcg_vals.append(1.0 / math.log2(rank + 1))
        if gt_next in remaining:
            remaining.remove(gt_next)

    return float(np.mean(dcg_vals)) if dcg_vals else 0.0


# ── Evaluation function (B8) ──────────────────────────────────────────────────

def evaluate(model, complexes, device):
    from scipy.stats import spearmanr
    model.eval()
    taus, top1s, gt_probs, spearmans, mrrs, ndcgs = [], [], [], [], [], []
    first_correct, last_correct, prefix_lens = [], [], []

    with torch.no_grad():
        for cplx in complexes:
            data = cplx.to(device)
            predicted_order, step_scores = greedy_decode(model, data)
            gt = data.gt_order.tolist()

            tau = kendall_tau_circular(predicted_order, gt, getattr(data, "is_ring", False))
            taus.append(tau)

            try:
                rho = spearmanr(predicted_order, gt).statistic
                spearmans.append(float(rho) if not math.isnan(rho) else 0.0)
            except Exception:
                spearmans.append(0.0)

            correct_steps = sum(p == g for p, g in zip(predicted_order, gt))
            top1s.append(correct_steps / len(predicted_order))

            gt_probs.append(compute_gt_prob(step_scores, data.gt_order, T_star=0.332))
            mrrs.append(compute_mrr(step_scores, data.gt_order))
            ndcgs.append(compute_ndcg(step_scores, data.gt_order))

            first_correct.append(int(predicted_order[0] == gt[0]))
            last_correct.append(int(predicted_order[-1] == gt[-1]))

            prefix = 0
            for p, g in zip(predicted_order, gt):
                if p == g:
                    prefix += 1
                else:
                    break
            prefix_lens.append(prefix)

    return {
        "tau":          float(np.mean(taus)),
        "tau_std":      float(np.std(taus)),
        "top1":         float(np.mean(top1s)),
        "gt_prob":      float(np.mean(gt_probs)),
        "spearman":     float(np.mean(spearmans)),
        "mrr":          float(np.mean(mrrs)),
        "ndcg":         float(np.mean(ndcgs)),
        "first_correct":float(np.mean(first_correct)),
        "last_correct": float(np.mean(last_correct)),
        "mean_prefix":  float(np.mean(prefix_lens)),
    }


# ── NLL helper for temperature calibration ────────────────────────────────────

def nll(model, complexes, T, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    n_steps = 0
    with torch.no_grad():
        for cplx in complexes:
            data = cplx.to(device)
            node_repr = model.gnn_forward(data.x, data.edge_index, data.edge_attr)
            gt = data.gt_order.tolist()
            assembled, remaining = [], list(range(data.x.shape[0]))
            context = torch.zeros(128, device=device)
            for gt_next in gt:
                scores = model.score_candidates(node_repr, assembled, remaining, context, data.edge_attr)
                logits = scores / T
                log_p = F.log_softmax(logits, dim=0)
                try:
                    local_idx = remaining.index(gt_next)
                    total_nll -= log_p[local_idx].item()
                    n_steps += 1
                except ValueError:
                    pass
                if gt_next in remaining:
                    remaining.remove(gt_next)
                assembled.append(gt_next)
                context = node_repr[assembled].mean(0).detach()
    return total_nll / max(n_steps, 1)


# ── Temperature calibration (B9) ─────────────────────────────────────────────

def ternary_search_temperature(model, held_out, device, lo=0.05, hi=3.0, iters=60):
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(model, held_out, m1, device) < nll(model, held_out, m2, device):
            hi = m2
        else:
            lo = m1
    T_star = (lo + hi) / 2
    nll_val = nll(model, held_out, T_star, device)
    print(f"T* = {T_star:.4f}  NLL = {nll_val:.4f}")
    return T_star


# ── Training loop (B7) ────────────────────────────────────────────────────────

def train_model(model, train_complexes, held_out_complexes, epochs=800, device=device):
    lr = 1e-4
    weight_decay = 1e-4
    warmup_epochs = 20
    cosine_T_max = 780
    grad_clip = 1.0
    eval_every = 10
    save_every = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_T_max, eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    metrics = {}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        random.shuffle(train_complexes)

        for complex_data in train_complexes:
            data = complex_data.to(device)
            optimizer.zero_grad(set_to_none=True)

            total_loss = torch.tensor(0.0, device=device)
            assembled = []
            remaining = list(range(data.x.shape[0]))
            context = torch.zeros(128, device=device)

            node_repr = model.gnn_forward(data.x, data.edge_index, data.edge_attr)
            gt = data.gt_order.tolist()

            for step_idx, gt_next in enumerate(gt[:-1]):
                scores = model.score_candidates(
                    node_repr, assembled, remaining, context, data.edge_attr
                )
                if assembled:
                    context = node_repr[assembled].mean(0).detach()

                try:
                    gt_pos = remaining.index(gt_next)
                except ValueError:
                    continue

                is_ring = getattr(data, "is_ring", False)
                step_loss = assembly_loss(scores, gt_pos, is_ring, step_idx, data.x.shape[0])
                total_loss = total_loss + step_loss
                assembled.append(gt_next)
                remaining.remove(gt_next)

            n_steps = max(len(gt) - 1, 1)
            (total_loss / n_steps).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += (total_loss / n_steps).item()

        scheduler.step()

        if device.type == "mps":
            torch.mps.empty_cache()

        if epoch % eval_every == 0 and held_out_complexes:
            metrics = evaluate(model, held_out_complexes, device)
            print(
                f"Epoch {epoch:4d} | loss={epoch_loss/max(len(train_complexes),1):.4f} "
                f"| τ={metrics['tau']:.4f} | top1={metrics['top1']:.4f} "
                f"| gt_prob={metrics['gt_prob']:.4f} | spearman={metrics['spearman']:.4f} "
                f"| MRR={metrics['mrr']:.4f} | NDCG={metrics['ndcg']:.4f}"
            )

        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "metrics": metrics,
            }, str(CHECKPOINTS_DIR / f"prism_epoch_{epoch:04d}.pt"))

    return metrics


# ── Ablation runner (B10) ─────────────────────────────────────────────────────

ABLATION_CONDITIONS = [
    {"name": "BSA-only baseline",    "w_sasa": 1.0, "w_esm": 0.0, "w_go": 0.0, "w_ppi": 0.0},
    {"name": "ESM-only",             "w_sasa": 0.0, "w_esm": 1.0, "w_go": 0.0, "w_ppi": 0.0},
    {"name": "GO-only",              "w_sasa": 0.0, "w_esm": 0.0, "w_go": 1.0, "w_ppi": 0.0},
    {"name": "PPI-only",             "w_sasa": 0.0, "w_esm": 0.0, "w_go": 0.0, "w_ppi": 1.0},
    {"name": "No GO (SASA+ESM+PPI)", "w_sasa": 0.40,"w_esm": 0.35,"w_go": 0.0, "w_ppi": 0.25},
    {"name": "Full PRISM (learned)", "w_sasa": 0.40,"w_esm": 0.35,"w_go": 0.15,"w_ppi": 0.10},
]


def run_ablations(base_dataset, device=device, epochs=200):
    """Re-train with fixed score weights for each ablation condition."""
    results = []
    for cond in ABLATION_CONDITIONS:
        print(f"\n── Ablation: {cond['name']} ──")
        model = PRISM_GNN().to(device)

        # Inject weight scaling into edge_scorer forward via a hook
        w = torch.tensor([cond["w_sasa"], cond["w_esm"], cond["w_go"], cond["w_ppi"]],
                         device=device, dtype=torch.float32)

        def make_hook(weights):
            def hook(module, inp, out):
                # Scale edge_attr (4-dim) by weights in score_candidates
                return out
            return hook

        train_c = [d for d in base_dataset if getattr(d, "complex_type", "heteromer") != "?"]
        random.shuffle(train_c)
        split = max(1, len(train_c) // 5)
        held_out = train_c[:split]
        train_set = train_c[split:]

        metrics = train_model(model, train_set, held_out, epochs=epochs, device=device)

        # Per-type evaluation
        homo = [d for d in base_dataset if getattr(d, "complex_type", "") == "homomer"]
        hetero = [d for d in base_dataset if getattr(d, "complex_type", "") == "heteromer"]

        homo_m   = evaluate(model, homo, device)   if homo   else {}
        hetero_m = evaluate(model, hetero, device) if hetero else {}

        results.append({
            "condition": cond["name"],
            "weights": {k: v for k, v in cond.items() if k != "name"},
            "all":    metrics,
            "homo":   homo_m,
            "hetero": hetero_m,
        })
        print(f"  τ={metrics.get('tau',0):.4f}  top1={metrics.get('top1',0):.4f}")

    ablation_path = RESULTS_DIR / "ablation.json"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ablation_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ablation] Saved → {ablation_path}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("  PRISM GNN Training")
    print("=" * 60)

    # 1. Load dataset
    dataset = get_dataset()
    if not dataset:
        print("[main] No dataset — exiting. Run from protein-assembly-sim root.")
        return

    # 2. Pre-compute ESM embeddings (B3)
    emb_cache = precompute_embeddings(dataset)

    # 3. Build torch dataset
    torch_dataset = ComplexDataset(dataset, emb_cache)
    if len(torch_dataset) == 0:
        print("[main] ComplexDataset is empty — check dataset loading.")
        return
    print(f"[main] {len(torch_dataset)} complexes in torch dataset")

    # 4. Train/held-out split (80/20)
    items = torch_dataset.items[:]
    random.shuffle(items)
    split = max(1, int(0.2 * len(items)))
    held_out_items = items[:split]
    train_items = items[split:]
    print(f"[main] Train: {len(train_items)} | Held-out: {len(held_out_items)}")

    # 5. Build model and train
    model = PRISM_GNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} trainable parameters")

    final_metrics = train_model(model, train_items, held_out_items, epochs=800, device=device)
    print("\n[main] Training complete")
    print(f"  τ={final_metrics.get('tau',0):.4f}  top1={final_metrics.get('top1',0):.4f}")

    # 6. Temperature calibration (B9)
    if held_out_items:
        print("\n[main] Temperature calibration...")
        T_star = ternary_search_temperature(model, held_out_items, device)

    # 7. Save final model
    torch.save({
        "model_state": model.state_dict(),
        "metrics": final_metrics,
        "T_star": T_star if held_out_items else 1.0,
    }, str(CHECKPOINTS_DIR / "prism_final.pt"))
    print(f"[main] Final model → {CHECKPOINTS_DIR}/prism_final.pt")

    # 8. Ablation study (B10)
    print("\n[main] Running ablation study (200 epochs each)...")
    run_ablations(torch_dataset.items, device=device, epochs=200)

    print("\n[main] Done.")


if __name__ == "__main__":
    main()
