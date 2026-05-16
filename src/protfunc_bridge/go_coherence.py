"""
FABLE bridge: derives GO-MF functional coherence scores from existing FABLE model.

Uses unified_v1.pth (ImprovedResidualMLP, in_dim=331, hidden=2048, n_blocks=8, 8124 labels).
Input: 320d ESM-2 8M embedding + 11 sequence-only features (computable from FASTA).
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

# ── Paths ───────────────────────────────────────────────────────────────────
PROTFUNC_DIR  = Path.home() / "insecta_webapp"
MODEL_PATH    = PROTFUNC_DIR / "unified_v1.pth"
MLB_PATH      = PROTFUNC_DIR / "protfunc" / "mlb_public_v1.pkl"
THRESH_PATH   = PROTFUNC_DIR / "protfunc" / "per_label_thresholds.json"

ESM_DIM = 320


# ── Sequence feature computation ────────────────────────────────────────────
# Hydrophobicity scale (Kyte-Doolittle)
_KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,
       "G":-0.4,"H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,
       "P":-1.6,"S":-0.8,"T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
_CHARGE = {"R":1,"K":1,"H":0.1,"D":-1,"E":-1}
# Chou-Fasman propensities (helix, sheet)
_CF_H = {"A":1.42,"R":0.98,"N":0.67,"D":1.01,"C":0.70,"Q":1.11,"E":1.51,
         "G":0.57,"H":1.00,"I":1.08,"L":1.21,"K":1.16,"M":1.45,"F":1.13,
         "P":0.57,"S":0.77,"T":0.83,"W":1.08,"Y":0.69,"V":1.06}
_CF_S = {"A":0.83,"R":0.93,"N":0.89,"D":0.54,"C":1.19,"Q":1.10,"E":0.37,
         "G":0.75,"H":0.87,"I":1.60,"L":1.30,"K":0.74,"M":1.05,"F":1.38,
         "P":0.55,"S":0.75,"T":1.19,"W":1.37,"Y":1.47,"V":1.70}
# TM helix tendency (Wimley-White)
_TM = {"L","I","V","F","M","C","A","W"}
# Low complexity proxy: poly-runs
def _lowcomp(seq):
    runs = sum(1 for i in range(len(seq)-3) if len(set(seq[i:i+4]))==1)
    return runs / max(len(seq)-3, 1)
def _signal_peptide(seq):
    return 1.0 if seq[:20].count("L")+seq[:20].count("I")+seq[:20].count("V") >= 6 else 0.0
def _tm_frac(seq):
    return sum(1 for aa in seq if aa in _TM) / max(len(seq), 1)
def _tm_any(seq):
    window = 19
    for i in range(len(seq)-window+1):
        w = seq[i:i+window]
        if sum(1 for aa in w if aa in _TM) >= 14:
            return 1.0
    return 0.0
def _idr_frac(seq):
    disordered = set("RQEGKSP")
    return sum(1 for aa in seq if aa in disordered) / max(len(seq), 1)


def compute_seq_features(seq: str) -> np.ndarray:
    """Compute 11 SEQ_ONLY_COLS features from raw sequence."""
    seq = seq.upper().replace("X","G").replace("U","C").replace("B","N").replace("Z","E")
    n = len(seq)
    feats = np.array([
        float(n),                                                      # f_seq_len
        np.mean([_KD.get(aa, 0.0) for aa in seq]),                    # f_mean_hydro
        sum(_CHARGE.get(aa, 0.0) for aa in seq) / n,                  # f_net_charge
        # uversky disorder: sum of charged+polar / total
        sum(1 for aa in seq if aa in "RKEQDNST") / n,                 # f_uversky_disorder
        _idr_frac(seq),                                                # f_idr_frac_proxy
        _lowcomp(seq),                                                 # f_lowcomp_proxy
        _tm_frac(seq),                                                 # f_tm_frac_proxy
        _tm_any(seq),                                                  # f_tm_any_proxy
        _signal_peptide(seq),                                          # f_signal_peptide_proxy
        np.mean([_CF_H.get(aa, 1.0) for aa in seq]),                  # f_cf_helix_mean
        np.mean([_CF_S.get(aa, 1.0) for aa in seq]),                  # f_cf_sheet_mean
    ], dtype=np.float32)
    return feats


def normalize_seq_features(feats: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (feats - mu[:11]) / (sd[:11] + 1e-8)


# ── Architecture ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.net(x)


class ImprovedResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, n_blocks: int, dropout: float = 0.2):
        super().__init__()
        self.fc_in  = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.fc_out = nn.Sequential(
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        h = self.fc_in(x)
        for b in self.blocks:
            h = b(h)
        return self.fc_out(h)


# ── Bridge ───────────────────────────────────────────────────────────────────

class FABLEBridge:
    """
    Wraps unified_v1.pth FABLE model for GO-MF probability inference.
    Input: protein sequence → 8124-dim GO-MF probability vector.
    Caches ESM embeddings and GO predictions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        device: Optional[str] = None,
        threshold: float = 0.1,
    ):
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.threshold       = threshold
        self.model_path      = Path(model_path or MODEL_PATH)
        self.esm_model_name  = esm_model_name
        self._esm_model      = None
        self._batch_converter = None
        self._classifier     = None
        self._mlb            = None
        self._thresholds: dict[str, float] = {}
        self._mu: Optional[np.ndarray] = None
        self._sd: Optional[np.ndarray] = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._prediction_cache: dict[str, np.ndarray] = {}

    def _ensure_loaded(self):
        if self._classifier is not None:
            return

        import esm as esm_lib
        import joblib

        # ESM
        esm_model, alphabet = getattr(esm_lib.pretrained, self.esm_model_name)()
        self._esm_model       = esm_model.to(self.device).eval()
        self._batch_converter = alphabet.get_batch_converter()

        # Checkpoint
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        sd   = ckpt["model"]
        in_dim   = ckpt.get("in_dim",   331)
        hidden   = ckpt.get("hidden",   2048)
        n_blocks = ckpt.get("n_blocks", 8)
        out_dim  = sd["fc_out.3.weight"].shape[0]

        # Normalization stats (first 11 = SEQ_ONLY_COLS)
        mu_all = ckpt.get("supp_mu", [])
        sd_all = ckpt.get("supp_sd", [])
        if isinstance(mu_all, torch.Tensor):
            mu_all = mu_all.numpy()
        if isinstance(sd_all, torch.Tensor):
            sd_all = sd_all.numpy()
        self._mu = np.array(mu_all[:11], dtype=np.float32) if len(mu_all) >= 11 else np.zeros(11, dtype=np.float32)
        self._sd = np.array(sd_all[:11], dtype=np.float32) if len(sd_all) >= 11 else np.ones(11,  dtype=np.float32)
        self._sd[self._sd < 1e-8] = 1.0

        clf = ImprovedResidualMLP(in_dim, out_dim, hidden, n_blocks).to(self.device)
        clf.load_state_dict(sd, strict=True)
        clf.eval()
        self._classifier = clf

        # Labels + thresholds
        if MLB_PATH.exists():
            self._mlb = joblib.load(MLB_PATH)
        if THRESH_PATH.exists():
            with open(THRESH_PATH) as f:
                self._thresholds = json.load(f)

        print(f"[FABLEBridge] loaded — {n_blocks} blocks, {out_dim} labels, device={self.device}")

    @torch.no_grad()
    def get_embedding(self, sequence: str) -> np.ndarray:
        if sequence in self._embedding_cache:
            return self._embedding_cache[sequence]
        self._ensure_loaded()
        _, _, tokens = self._batch_converter([("p", sequence)])
        tokens = tokens.to(self.device)
        out = self._esm_model(tokens, repr_layers=[6], return_contacts=False)
        emb = out["representations"][6][0, 1:len(sequence)+1].mean(0).cpu().numpy()
        self._embedding_cache[sequence] = emb
        return emb

    @torch.no_grad()
    def predict_go_probs(self, sequence: str) -> np.ndarray:
        if sequence in self._prediction_cache:
            return self._prediction_cache[sequence]
        self._ensure_loaded()

        emb      = self.get_embedding(sequence)                         # (320,)
        seq_feat = compute_seq_features(sequence)                       # (11,)
        seq_norm = normalize_seq_features(seq_feat, self._mu, self._sd) # (11,)
        x        = np.concatenate([emb, seq_norm])                      # (331,)

        xt     = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self._classifier(xt).squeeze(0)
        probs  = torch.sigmoid(logits).cpu().numpy()
        self._prediction_cache[sequence] = probs
        return probs

    @torch.no_grad()
    def get_logits(self, sequence: str) -> np.ndarray:
        """
        Raw pre-sigmoid logits. More discriminating than probabilities for
        out-of-distribution (human) sequences where sigmoid saturates near 1.
        Use for cosine similarity when proteins are outside training domain.
        """
        self._ensure_loaded()

        emb      = self.get_embedding(sequence)
        seq_feat = compute_seq_features(sequence)
        seq_norm = normalize_seq_features(seq_feat, self._mu, self._sd)
        x        = np.concatenate([emb, seq_norm])

        xt     = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self._classifier(xt).squeeze(0).cpu().numpy()
        return logits

    def predict_go_terms(self, sequence: str) -> set[str]:
        probs = self.predict_go_probs(sequence)
        if self._mlb is not None:
            return {
                self._mlb.classes_[i]
                for i, p in enumerate(probs)
                if p >= self._thresholds.get(self._mlb.classes_[i], self.threshold)
            }
        return {f"GO_idx_{i}" for i, p in enumerate(probs) if p >= self.threshold}


# ── Scoring helpers ──────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def go_coherence_score(
    new_seq: str,
    complex_seqs: list[str],
    bridge: "FABLEBridge",
    use_logits: bool = True,
) -> float:
    """
    Cosine similarity between new_seq and mean complex GO profile.
    use_logits=True (default): uses pre-sigmoid logits for better discrimination
    on out-of-distribution (human) sequences where sigmoid saturates.
    """
    fn = bridge.get_logits if use_logits else bridge.predict_go_probs
    new_p = fn(new_seq)
    if not complex_seqs:
        return 0.5
    complex_mean = np.mean([fn(s) for s in complex_seqs], axis=0)
    return _cosine(new_p, complex_mean)


def pairwise_go_matrix(sequences: list[str], bridge: "FABLEBridge",
                       use_logits: bool = True) -> np.ndarray:
    """
    N×N pairwise GO cosine similarity. Uses logits by default.
    """
    N    = len(sequences)
    fn   = bridge.get_logits if use_logits else bridge.predict_go_probs
    vecs = [fn(s) for s in sequences]
    mat  = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                mat[i, j] = _cosine(vecs[i], vecs[j])
    return mat
