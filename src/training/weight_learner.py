"""
Learnable 4-weight linear scorer.
Learns w_sasa, w_esm, w_go, w_ppi via softmax ranking loss.
This is Table 1's "learned weights" row and the interpretable baseline for the GNN.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class WeightLearner(nn.Module):
    """
    score(candidate) = w_sasa * sasa_max + w_esm * esm_max + w_go * go_max + w_ppi * ppi_max

    features[:, 0] = sasa_max
    features[:, 1] = esm_max
    features[:, 2] = go_max
    features[:, 3] = ppi_max
    (features[:, 4:8] = mean equivalents, used as auxiliary by MLP variant)
    """

    def __init__(self, init_weights: Optional[list[float]] = None, use_mean: bool = False):
        super().__init__()
        self.use_mean = use_mean
        n_feat = 8 if use_mean else 4
        w_init = init_weights or [0.25, 0.25, 0.25, 0.25]
        if use_mean:
            w_init = w_init + [0.0, 0.0, 0.0, 0.0]
        self.weights = nn.Parameter(torch.tensor(w_init, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (k, 8) → scores: (k,)"""
        n_feat = 8 if self.use_mean else 4
        w = torch.softmax(self.weights[:n_feat], dim=0) * 4.0  # keep sum=4
        return (features[:, :n_feat] * w).sum(dim=-1)

    def get_weights(self) -> np.ndarray:
        """Returns [w_sasa, w_esm, w_go, w_ppi] normalized to sum=1."""
        with torch.no_grad():
            w = torch.softmax(self.weights[:4], dim=0).cpu().numpy()
        return w

    def named_weights(self) -> dict[str, float]:
        w = self.get_weights()
        return {"w_sasa": w[0], "w_esm": w[1], "w_go": w[2], "w_ppi": w[3]}


class MLPScorer(nn.Module):
    """
    Small MLP on 8-dim features. Slightly more expressive than WeightLearner.
    Still very interpretable for ablation.
    """

    def __init__(self, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def get_weights(self) -> np.ndarray:
        # Approximate weights via feature importance (gradient at mean features)
        x = torch.ones(1, 8) * 0.5
        x.requires_grad_(True)
        y = self.net(x).sum()
        y.backward()
        grad = x.grad.abs()[0, :4].detach().numpy()
        total = grad.sum() + 1e-8
        return grad / total

    def named_weights(self) -> dict[str, float]:
        w = self.get_weights()
        return {"w_sasa": w[0], "w_esm": w[1], "w_go": w[2], "w_ppi": w[3]}
