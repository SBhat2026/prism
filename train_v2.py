"""
PRISM v2 trainer — 7K-complex corpus (Tier 1 + Tier 2).

Reads feature bundles from data/labeled_15k/ (written by precompute_features.py),
builds train/val/test splits, and trains AssemblyScorer with minibatching.

Split strategy:
  test  — all 23 Tier 1 (Marsh 2013) complexes, held entirely out
  train — 80% of Tier 2 keys (random seed 42)
  val   — 10% of Tier 2 keys
  holdout — 10% of Tier 2 keys (not used during training)

Outputs:
  results/v2/best_v2.pt         — best checkpoint by val gt_prob
  results/v2/final_v2.pt        — end-of-training checkpoint
  results/v2/temperature.json   — calibrated temperature T*
  results/v2/metrics.json       — per-epoch train/val metrics
  data/labeled_15k/splits.json  — reproducible split assignment

Usage:
  python train_v2.py --epochs 200 --batch_size 32
  python train_v2.py --epochs 200 --batch_size 32 --resume
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.scorer_gnn import AssemblyScorer, build_edge_features

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR    = ROOT / "data" / "labeled_15k"
RESULTS_DIR = ROOT / "results" / "v2"
SPLITS_JSON = DATA_DIR / "splits.json"
INDEX_JSON  = DATA_DIR / "index.json"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Marsh 2013 Tier 1 PDB IDs — held out as test set
MARSH_PDBS = {
    "2HHB", "1BRS", "1AY7", "1TGS", "2PTC", "1A2K",
    "1SBB", "3GBN", "1AON", "1GRU",
}


# ── Data loading ──────────────────────────────────────────────────────────────

class ComplexBundle:
    """Lazy-loaded feature bundle for one complex."""
    __slots__ = ("key", "meta", "node_feats", "edge_feats", "adj",
                 "gt_order", "complex_type", "is_ring", "n")

    def __init__(self, key: str, meta: dict,
                 node_feats: torch.Tensor, edge_feats: torch.Tensor,
                 adj: torch.Tensor):
        self.key          = key
        self.meta         = meta
        self.node_feats   = node_feats
        self.edge_feats   = edge_feats
        self.adj          = adj
        self.gt_order     = meta["gt_order"]
        self.complex_type = meta.get("complex_type", "heteromer")
        self.is_ring      = meta.get("is_ring", False)
        self.n            = len(meta["chain_ids"])


def _load_bundle(key: str) -> ComplexBundle | None:
    d = DATA_DIR / key
    if not (d / "meta.json").exists():
        return None
    try:
        meta    = json.loads((d / "meta.json").read_text())
        emb     = np.load(d / "emb.npy")     # (N, 640)
        sasa    = np.load(d / "sasa.npy")    # (N, N)
        plddt   = np.load(d / "plddt.npy")   # (N, 5)
        ppi     = np.load(d / "ppi.npy")     # (N, N)
        go      = np.load(d / "go.npy")      # (N, N)
    except Exception as e:
        print(f"  [load] {key} failed: {e}")
        return None

    N = emb.shape[0]
    if N < 2:
        return None
    # Validate all matrix shapes match (STRING/SASA can return fewer entries)
    if sasa.shape != (N, N) or ppi.shape != (N, N) or go.shape != (N, N):
        print(f"  [load] {key} shape mismatch: emb={emb.shape} sasa={sasa.shape} ppi={ppi.shape} go={go.shape} — skipping")
        return None
    # Validate gt_order
    gt_order = meta.get("gt_order", [])
    if len(gt_order) != N:
        gt_order = list(range(N))
        meta["gt_order"] = gt_order

    # Node features: ESM (640) + pLDDT (5) + SASA diag (1) = 646
    sasa_norm = sasa / (sasa.max() + 1e-8)
    sasa_diag = np.diag(sasa_norm).reshape(-1, 1).astype(np.float32)
    # pLDDT normalise mean/q10/q90 → [0,1]
    plddt_f = plddt.copy()
    plddt_f[:, 0] /= 100.0
    plddt_f[:, 3] /= 100.0
    plddt_f[:, 4] /= 100.0
    node_np = np.hstack([emb, plddt_f, sasa_diag]).astype(np.float32)

    # Edge features: [ppi, esm_cos, go_cos, sasa_norm]
    norms   = np.linalg.norm(emb, axis=1, keepdims=True)
    esm_cos = (emb / (norms + 1e-8) @ (emb / (norms + 1e-8)).T).clip(-1, 1).astype(np.float32)
    edge_np = build_edge_features(ppi, esm_cos, go, sasa_norm)  # returns (N,N,4) tensor

    node_t = torch.tensor(node_np, dtype=torch.float32)
    adj_t  = torch.ones(N, N, dtype=torch.bool)

    return ComplexBundle(key, meta, node_t, edge_np, adj_t)


def load_split(keys: list[str]) -> list[ComplexBundle]:
    bundles = []
    for k in keys:
        b = _load_bundle(k)
        if b is not None:
            bundles.append(b)
    return bundles


# ── Splits ────────────────────────────────────────────────────────────────────

def _discover_bundles() -> dict[str, dict]:
    """
    Discover all ready complexes from filesystem (not just index.json).
    Falls back to index.json entries too, so whichever is more up-to-date wins.
    """
    discovered: dict[str, dict] = {}

    # From index.json (may lag behind filesystem by up to 100 rows)
    if INDEX_JSON.exists():
        try:
            for k, v in json.loads(INDEX_JSON.read_text()).items():
                if v.get("has_features"):
                    discovered[k] = v
        except Exception:
            pass

    # Filesystem scan — picks up entries not yet in index
    if DATA_DIR.exists():
        for d in DATA_DIR.iterdir():
            if not d.is_dir():
                continue
            meta_f = d / "meta.json"
            emb_f  = d / "emb.npy"
            if not (meta_f.exists() and emb_f.exists()):
                continue
            k = d.name
            if k in discovered:
                continue
            try:
                meta = json.loads(meta_f.read_text())
                discovered[k] = {
                    "tier":        meta.get("gt_tier", 3),
                    "n_subunits":  meta.get("n_subunits", 0),
                    "source":      meta.get("source", ""),
                    "has_features": True,
                }
            except Exception:
                pass
    return discovered


def build_splits(force: bool = False) -> dict[str, list[str]]:
    if SPLITS_JSON.exists() and not force:
        return json.loads(SPLITS_JSON.read_text())

    index = _discover_bundles()
    all_keys = list(index.keys())

    tier1 = [k for k in all_keys if index[k]["tier"] == 1]
    tier2 = [k for k in all_keys if index[k]["tier"] == 2]

    # Warn if any Marsh PDB leaked into Tier 2
    leaked = [k for k in tier2 if k.upper() in MARSH_PDBS]
    if leaked:
        print(f"[split] WARNING: Marsh PDBs in Tier 2 — removing: {leaked}")
        tier2 = [k for k in tier2 if k.upper() not in MARSH_PDBS]

    rng = random.Random(42)
    rng.shuffle(tier2)
    n = len(tier2)
    n_train    = int(0.8 * n)
    n_val      = int(0.1 * n)
    train_keys = tier2[:n_train]
    val_keys   = tier2[n_train: n_train + n_val]
    hold_keys  = tier2[n_train + n_val:]

    splits = {
        "test":    tier1,
        "train":   train_keys,
        "val":     val_keys,
        "holdout": hold_keys,
    }
    SPLITS_JSON.write_text(json.dumps(splits, indent=2))
    print(f"[split] test={len(tier1)}  train={len(train_keys)}  "
          f"val={len(val_keys)}  holdout={len(hold_keys)}")
    return splits


# ── Loss ──────────────────────────────────────────────────────────────────────

def _get_orders(b: ComplexBundle) -> list[list[int]]:
    """All valid orderings for this complex (ring augmentation for homomers)."""
    base = b.gt_order
    N    = b.n
    if b.complex_type == "homomer" or b.is_ring:
        rev = list(reversed(base))
        orders = []
        for i in range(N):
            orders.append(base[i:] + base[:i])
            orders.append(rev[i:]  + rev[:i])
        return orders
    return [base]


def _order_loss(model: AssemblyScorer, N: int, order: list[int],
                h: torch.Tensor, edge_feats: torch.Tensor,
                temperature: float, eps: float = 0.1) -> torch.Tensor:
    step_losses = []
    for step in range(1, N):
        assembled  = order[:step]
        correct    = order[step]
        remaining  = [i for i in range(N) if i not in assembled]
        if not remaining:
            continue
        am = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        am[assembled] = True
        scores      = model.score_candidates(h, edge_feats.to(DEVICE), am, remaining) / temperature
        correct_idx = remaining.index(correct)
        nc = scores.size(0)
        soft = torch.full((nc,), eps / nc, device=DEVICE)
        soft[correct_idx] += 1.0 - eps
        step_losses.append(-(soft * F.log_softmax(scores, dim=0)).sum())
    if not step_losses:
        return torch.zeros(1, device=DEVICE).squeeze()
    return torch.stack(step_losses).mean()


def ranking_loss_batch(model: AssemblyScorer, batch: list[ComplexBundle],
                       temperature: float = 1.0) -> torch.Tensor:
    """Compute mean ranking loss over a minibatch of complexes."""
    model.train()
    losses = []
    for b in batch:
        nf = b.node_feats.to(DEVICE)
        ef = b.edge_feats.to(DEVICE)
        ad = b.adj.to(DEVICE)
        h  = model.embed(nf, ad, ef)
        orders = _get_orders(b)
        if len(orders) > 1:
            ol = [_order_loss(model, b.n, o, h, ef, temperature) for o in orders]
            losses.append(torch.stack(ol).min())
        else:
            losses.append(_order_loss(model, b.n, orders[0], h, ef, temperature))
    if not losses:
        return torch.zeros(1, device=DEVICE, requires_grad=True).squeeze()
    return torch.stack(losses).mean()


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_order(model: AssemblyScorer, b: ComplexBundle,
                 temperature: float) -> list[int]:
    model.eval()
    N  = b.n
    nf = b.node_feats.to(DEVICE)
    ef = b.edge_feats.to(DEVICE)
    ad = b.adj.to(DEVICE)
    h  = model.embed(nf, ad, ef)
    seed = b.gt_order[0] if b.gt_order else 0
    order = [seed]
    remaining = [i for i in range(N) if i != seed]
    for _ in range(N - 1):
        if not remaining:
            break
        am = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        am[order] = True
        scores = model.score_candidates(h, ef, am, remaining) / temperature
        best   = remaining[int(scores.argmax())]
        order.append(best)
        remaining.remove(best)
    return order


def eval_set(model: AssemblyScorer, bundles: list[ComplexBundle],
             temperature: float) -> dict:
    taus, top1s, gt_probs = [], [], []
    for b in bundles:
        pred = greedy_order(model, b, temperature)
        gt   = b.gt_order
        N    = b.n
        if N < 2:
            continue
        tau_val, _ = kendalltau(pred, gt)
        if not math.isnan(tau_val):
            taus.append(tau_val)
        top1s.append(int(pred[1] == gt[1]) if N >= 2 else 1)
        # gt_prob: softmax prob assigned to correct choice at step 1
        nf = b.node_feats.to(DEVICE)
        ef = b.edge_feats.to(DEVICE)
        ad = b.adj.to(DEVICE)
        with torch.no_grad():
            h  = model.embed(nf, ad, ef)
            am = torch.zeros(N, dtype=torch.bool, device=DEVICE)
            am[gt[0]] = True
            remaining = [i for i in range(N) if i != gt[0]]
            if remaining and len(gt) > 1:
                scores = model.score_candidates(h, ef, am, remaining) / temperature
                probs  = torch.softmax(scores, dim=0)
                correct_idx = remaining.index(gt[1])
                gt_probs.append(float(probs[correct_idx]))

    return {
        "tau":     float(np.mean(taus))    if taus     else 0.0,
        "top1":    float(np.mean(top1s))   if top1s    else 0.0,
        "gt_prob": float(np.mean(gt_probs))if gt_probs else 0.0,
        "n":       len(taus),
    }


# ── Temperature calibration ───────────────────────────────────────────────────

def calibrate_temperature(model: AssemblyScorer, bundles: list[ComplexBundle],
                           lo: float = 0.05, hi: float = 5.0,
                           iters: int = 30) -> float:
    """Ternary search for T* that maximises mean gt_prob on val set."""
    for _ in range(iters):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        s1 = eval_set(model, bundles, m1)["gt_prob"]
        s2 = eval_set(model, bundles, m2)["gt_prob"]
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    return (lo + hi) / 2


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Splits
    splits = build_splits(force=args.reset_splits)

    print(f"[data] loading train ({len(splits['train'])} keys)…")
    train_bundles = load_split(splits["train"])
    print(f"[data] loading val ({len(splits['val'])} keys)…")
    val_bundles   = load_split(splits["val"])
    print(f"[data] loading test ({len(splits['test'])} keys)…")
    test_bundles  = load_split(splits["test"])

    print(f"[data] loaded: train={len(train_bundles)} val={len(val_bundles)} "
          f"test={len(test_bundles)}")

    if not train_bundles:
        print("[error] No training data found. Run precompute_features.py first.")
        sys.exit(1)

    # Model
    model = AssemblyScorer(
        node_dim=646, edge_dim=4, hidden=128, n_layers=3, heads=4, dropout=0.1
    ).to(DEVICE)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup_epochs   = 10
    cosine_epochs   = args.epochs - warmup_epochs
    warmup_sched    = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cosine_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    temperature = 1.0  # will be calibrated after training
    best_gt_prob = -1.0
    patience_counter = 0
    metrics_log: list[dict] = []

    # Resume
    start_epoch = 1
    best_ckpt   = RESULTS_DIR / "best_v2.pt"
    resume_ckpt = RESULTS_DIR / "final_v2.pt"
    if args.resume and resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_gt_prob = ckpt.get("best_gt_prob", -1.0)
        print(f"[resume] from epoch {start_epoch}, best_gt_prob={best_gt_prob:.4f}")

    # Metrics log resume
    metrics_path = RESULTS_DIR / "metrics.json"
    if args.resume and metrics_path.exists():
        metrics_log = json.loads(metrics_path.read_text())

    batch_size = args.batch_size
    rng = random.Random(0)

    print(f"\n[train] {args.epochs} epochs · batch={batch_size} · "
          f"lr={args.lr} · device={DEVICE}")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        # Shuffle + minibatch
        rng.shuffle(train_bundles)
        epoch_losses = []
        for i in range(0, len(train_bundles), batch_size):
            batch = train_bundles[i: i + batch_size]
            optimiser.zero_grad()
            loss = ranking_loss_batch(model, batch, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        lr_now    = optimiser.param_groups[0]["lr"]

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_m  = eval_set(model, val_bundles, temperature)
            row = {
                "epoch":    epoch,
                "loss":     round(mean_loss, 5),
                "val_tau":  round(val_m["tau"], 4),
                "val_top1": round(val_m["top1"], 4),
                "val_gtp":  round(val_m["gt_prob"], 4),
                "lr":       round(lr_now, 7),
                "elapsed":  round(time.time() - t0, 1),
            }
            metrics_log.append(row)
            metrics_path.write_text(json.dumps(metrics_log, indent=2))

            print(f"[{epoch:4d}/{args.epochs}] loss={mean_loss:.4f}  "
                  f"val τ={val_m['tau']:.3f}  top1={val_m['top1']:.3f}  "
                  f"gt_prob={val_m['gt_prob']:.3f}  lr={lr_now:.2e}")

            if val_m["gt_prob"] > best_gt_prob:
                best_gt_prob     = val_m["gt_prob"]
                patience_counter = 0
                torch.save({
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "best_gt_prob":       best_gt_prob,
                    "config": {
                        "node_dim": 646, "edge_dim": 4,
                        "hidden": 128,   "n_layers": 3,
                        "heads": 4,      "dropout": 0.1,
                    },
                }, best_ckpt)
                print(f"  ✓ new best saved (gt_prob={best_gt_prob:.4f})")
            else:
                patience_counter += args.eval_every
                if patience_counter >= args.patience:
                    print(f"[early stop] patience {args.patience} exceeded at epoch {epoch}")
                    break
        else:
            if epoch % 10 == 0:
                print(f"[{epoch:4d}/{args.epochs}] loss={mean_loss:.4f}  lr={lr_now:.2e}")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "best_gt_prob":         best_gt_prob,
                "config": {
                    "node_dim": 646, "edge_dim": 4,
                    "hidden": 128,   "n_layers": 3,
                    "heads": 4,      "dropout": 0.1,
                },
            }, resume_ckpt)

    # Final checkpoint
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "best_gt_prob":       best_gt_prob,
        "config": {
            "node_dim": 646, "edge_dim": 4,
            "hidden": 128,   "n_layers": 3,
            "heads": 4,      "dropout": 0.1,
        },
    }, resume_ckpt)

    # Load best model for calibration + test eval
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n[calibrate] ternary search on {len(val_bundles)} val complexes…")
        T_star = calibrate_temperature(model, val_bundles, lo=0.05, hi=5.0, iters=40)
        print(f"[calibrate] T* = {T_star:.4f}")
        (RESULTS_DIR / "temperature.json").write_text(
            json.dumps({"T_star": round(T_star, 6)}, indent=2))

        print(f"\n[test] evaluating on {len(test_bundles)} Marsh-2013 complexes…")
        test_m = eval_set(model, test_bundles, T_star)
        print(f"[test] τ={test_m['tau']:.4f}  top1={test_m['top1']:.4f}  "
              f"gt_prob={test_m['gt_prob']:.4f}  n={test_m['n']}")

        val_m_final = eval_set(model, val_bundles, T_star)
        summary = {
            "best_epoch":      ckpt.get("epoch"),
            "T_star":          round(T_star, 6),
            "val_tau":         round(val_m_final["tau"], 4),
            "val_top1":        round(val_m_final["top1"], 4),
            "val_gt_prob":     round(val_m_final["gt_prob"], 4),
            "test_tau_marsh":  round(test_m["tau"], 4),
            "test_top1_marsh": round(test_m["top1"], 4),
            "test_gt_prob_marsh": round(test_m["gt_prob"], 4),
        }
        (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\n[summary]")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print(f"\n[done] outputs in {RESULTS_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",       type=int,   default=200)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--eval_every",   type=int,   default=5)
    ap.add_argument("--save_every",   type=int,   default=25)
    ap.add_argument("--patience",     type=int,   default=40)
    ap.add_argument("--resume",       action="store_true")
    ap.add_argument("--reset_splits", action="store_true",
                    help="Regenerate train/val/test split (ignores splits.json)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
