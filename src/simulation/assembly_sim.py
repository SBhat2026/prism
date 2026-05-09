"""
Assembly simulation engine.

Implements greedy and stochastic assembly order prediction.
Evaluates against ground-truth orders using Kendall's τ.

Scoring terms:
    S(i → complex) = w_sasa * ΔSASA_norm
                   + w_ppi  * PPI_score(i, complex)
                   + w_esm  * ESM_cosine(i, complex)
                   + w_go   * GO_coherence(i, complex)

Run standalone:
    python assembly_sim.py  # toy 4-subunit demo
"""

import random
import math
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from scipy.stats import kendalltau


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Complex:
    """A protein complex with N subunits indexed 0..N-1."""
    name: str
    sequences: list[str]           # one per subunit
    chain_ids: list[str]           # e.g. ["A", "B", "C", ...]
    pdb_path: Optional[str] = None

    # Precomputed pairwise matrices (filled by build_scoring_matrices)
    sasa_matrix: Optional[np.ndarray] = None   # (N, N) buried SASA nm²
    ppi_matrix:  Optional[np.ndarray] = None   # (N, N) STRING score [0,1]
    esm_matrix:  Optional[np.ndarray] = None   # (N, N) ESM cosine sim
    go_matrix:   Optional[np.ndarray] = None   # (N, N) GO coherence

    ground_truth_order: Optional[list[int]] = None  # 0-indexed assembly order


@dataclass
class AssemblyState:
    """Mutable state during a simulation run."""
    assembled: list[int] = field(default_factory=list)
    unassembled: list[int] = field(default_factory=list)
    step_scores: list[float] = field(default_factory=list)

    @property
    def order(self) -> list[int]:
        return self.assembled.copy()


@dataclass
class SimulationResult:
    predicted_order: list[int]
    ground_truth_order: Optional[list[int]]
    kendall_tau: Optional[float]
    kendall_pval: Optional[float]
    step_scores: list[float]
    mode: str  # "greedy" | "stochastic"


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def candidate_score(
    candidate: int,
    assembled: list[int],
    sasa_matrix: np.ndarray,
    ppi_matrix: np.ndarray,
    esm_matrix: np.ndarray,
    go_matrix: np.ndarray,
    w_sasa: float = 1.0,
    w_ppi:  float = 0.0,
    w_esm:  float = 0.0,
    w_go:   float = 0.0,
) -> float:
    """Score candidate subunit joining current partial complex."""
    if not assembled:
        return 0.0

    sasa = sasa_matrix[candidate, assembled].max()
    ppi  = ppi_matrix[candidate, assembled].max()
    esm  = esm_matrix[candidate, assembled].max()
    go   = go_matrix[candidate, assembled].max()

    # Normalize SASA to [0,1]
    sasa_max = sasa_matrix.max()
    sasa_n = sasa / (sasa_max + 1e-8)

    return w_sasa * sasa_n + w_ppi * ppi + w_esm * esm + w_go * go


# ──────────────────────────────────────────────────────────────────────────────
# Simulation modes
# ──────────────────────────────────────────────────────────────────────────────

def greedy_assembly(
    cplx: Complex,
    seed_subunit: int = 0,
    w_sasa: float = 1.0,
    w_ppi:  float = 0.0,
    w_esm:  float = 0.0,
    w_go:   float = 0.0,
) -> SimulationResult:
    """Deterministic greedy: always add the highest-scoring candidate."""
    N = len(cplx.sequences)
    state = AssemblyState(
        assembled=[seed_subunit],
        unassembled=[i for i in range(N) if i != seed_subunit],
    )

    kwargs = dict(
        sasa_matrix=_ensure(cplx.sasa_matrix, N),
        ppi_matrix=_ensure(cplx.ppi_matrix, N),
        esm_matrix=_ensure(cplx.esm_matrix, N),
        go_matrix=_ensure(cplx.go_matrix, N),
        w_sasa=w_sasa, w_ppi=w_ppi, w_esm=w_esm, w_go=w_go,
    )

    while state.unassembled:
        scores = [
            candidate_score(c, state.assembled, **kwargs)
            for c in state.unassembled
        ]
        best_idx = int(np.argmax(scores))
        chosen = state.unassembled[best_idx]
        state.step_scores.append(scores[best_idx])
        state.assembled.append(chosen)
        state.unassembled.remove(chosen)

    return _make_result(state, cplx, "greedy")


def stochastic_assembly(
    cplx: Complex,
    seed_subunit: int = 0,
    temperature: float = 1.0,
    w_sasa: float = 1.0,
    w_ppi:  float = 0.0,
    w_esm:  float = 0.0,
    w_go:   float = 0.0,
    rng: Optional[random.Random] = None,
) -> SimulationResult:
    """
    Stochastic assembly: sample next subunit proportional to softmax(scores/T).
    temperature→0 recovers greedy; temperature→∞ gives uniform random.
    """
    if rng is None:
        rng = random.Random()

    N = len(cplx.sequences)
    state = AssemblyState(
        assembled=[seed_subunit],
        unassembled=[i for i in range(N) if i != seed_subunit],
    )

    kwargs = dict(
        sasa_matrix=_ensure(cplx.sasa_matrix, N),
        ppi_matrix=_ensure(cplx.ppi_matrix, N),
        esm_matrix=_ensure(cplx.esm_matrix, N),
        go_matrix=_ensure(cplx.go_matrix, N),
        w_sasa=w_sasa, w_ppi=w_ppi, w_esm=w_esm, w_go=w_go,
    )

    while state.unassembled:
        scores = np.array([
            candidate_score(c, state.assembled, **kwargs)
            for c in state.unassembled
        ])
        # Softmax sampling
        scores_t = scores / max(temperature, 1e-8)
        scores_t -= scores_t.max()  # numerical stability
        probs = np.exp(scores_t)
        probs /= probs.sum()
        idx = rng.choices(range(len(state.unassembled)), weights=probs.tolist())[0]
        chosen = state.unassembled[idx]
        state.step_scores.append(float(scores[idx]))
        state.assembled.append(chosen)
        state.unassembled.remove(chosen)

    return _make_result(state, cplx, "stochastic")


def ensemble_assembly(
    cplx: Complex,
    n_runs: int = 100,
    temperature: float = 0.5,
    w_sasa: float = 1.0,
    w_ppi:  float = 0.0,
    w_esm:  float = 0.0,
    w_go:   float = 0.0,
    seed: int = 42,
) -> tuple[list[SimulationResult], dict]:
    """
    Run stochastic ensemble, return all results + pathway statistics.
    Statistics: mean order, order frequency distribution, bottleneck steps.
    """
    rng = random.Random(seed)
    results = [
        stochastic_assembly(
            cplx, temperature=temperature,
            w_sasa=w_sasa, w_ppi=w_ppi, w_esm=w_esm, w_go=w_go,
            rng=rng,
        )
        for _ in range(n_runs)
    ]

    # Pathway frequency
    order_counts: dict[tuple, int] = {}
    for r in results:
        key = tuple(r.predicted_order)
        order_counts[key] = order_counts.get(key, 0) + 1

    # Mean Kendall τ
    taus = [r.kendall_tau for r in results if r.kendall_tau is not None]
    mean_tau = float(np.mean(taus)) if taus else None

    # Step entropy: diversity at each assembly step
    N = len(cplx.sequences)
    step_entropies = []
    for step in range(1, N):
        choices = [r.predicted_order[step] for r in results]
        counts = np.bincount(choices, minlength=N).astype(float)
        counts /= counts.sum()
        ent = -np.sum(counts[counts > 0] * np.log2(counts[counts > 0]))
        step_entropies.append(ent)

    stats = {
        "mean_tau": mean_tau,
        "order_distribution": sorted(order_counts.items(), key=lambda x: -x[1]),
        "step_entropies": step_entropies,
        "max_entropy_step": int(np.argmax(step_entropies)) + 1 if step_entropies else None,
        "n_unique_pathways": len(order_counts),
    }
    return results, stats


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_kendall_tau(
    predicted: list[int], ground_truth: list[int]
) -> tuple[float, float]:
    """Kendall's τ between predicted and ground-truth assembly orders."""
    assert len(predicted) == len(ground_truth)
    # Map to rank arrays
    n = len(predicted)
    gt_rank = {v: i for i, v in enumerate(ground_truth)}
    pred_ranks = list(range(n))
    gt_ranks = [gt_rank[v] for v in predicted]
    tau, pval = kendalltau(pred_ranks, gt_ranks)
    return float(tau), float(pval)


def ablation_table(cplx: Complex, n_stochastic: int = 50) -> list[dict]:
    """
    Run all ablation conditions and return rows for the paper table.
    Matches the ABLATION_CONDITIONS from the research plan.
    """
    conditions = [
        {"name": "SASA only (Marsh baseline)",  "w_sasa": 1.0, "w_ppi": 0.0, "w_esm": 0.0, "w_go": 0.0},
        {"name": "PPI only",                    "w_sasa": 0.0, "w_ppi": 1.0, "w_esm": 0.0, "w_go": 0.0},
        {"name": "ESM compat only",             "w_sasa": 0.0, "w_ppi": 0.0, "w_esm": 1.0, "w_go": 0.0},
        {"name": "GO coherence only",           "w_sasa": 0.0, "w_ppi": 0.0, "w_esm": 0.0, "w_go": 1.0},
        {"name": "PPI + ESM (no GO)",           "w_sasa": 0.0, "w_ppi": 0.5, "w_esm": 0.5, "w_go": 0.0},
        {"name": "All terms (equal weights)",   "w_sasa": 0.25,"w_ppi": 0.25,"w_esm": 0.25,"w_go": 0.25},
    ]

    rows = []
    for cond in conditions:
        w = {k: v for k, v in cond.items() if k.startswith("w_")}
        greedy_r = greedy_assembly(cplx, **w)
        _, stats = ensemble_assembly(cplx, n_runs=n_stochastic, **w)
        rows.append({
            "condition": cond["name"],
            "greedy_tau": greedy_r.kendall_tau,
            "mean_stochastic_tau": stats["mean_tau"],
            "n_unique_pathways": stats["n_unique_pathways"],
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure(mat: Optional[np.ndarray], N: int) -> np.ndarray:
    return mat if mat is not None else np.zeros((N, N))


def _make_result(state: AssemblyState, cplx: Complex, mode: str) -> SimulationResult:
    gt = cplx.ground_truth_order
    tau, pval = (evaluate_kendall_tau(state.assembled, gt) if gt else (None, None))
    return SimulationResult(
        predicted_order=state.assembled,
        ground_truth_order=gt,
        kendall_tau=tau,
        kendall_pval=pval,
        step_scores=state.step_scores,
        mode=mode,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Toy demo
# ──────────────────────────────────────────────────────────────────────────────

def _toy_demo():
    """4-subunit complex with synthetic SASA matrix. Verifies everything runs."""
    rng = np.random.default_rng(0)
    N = 4
    # Synthetic: subunit 0 is the seed, then 2, 1, 3 in order (larger buried area = earlier)
    sasa = np.array([
        [0,   8.0, 3.0, 1.0],
        [8.0, 0,   2.0, 4.0],
        [3.0, 2.0, 0,   5.0],
        [1.0, 4.0, 5.0, 0  ],
    ])
    cplx = Complex(
        name="toy",
        sequences=["ACDEF"] * N,
        chain_ids=["A", "B", "C", "D"],
        sasa_matrix=sasa,
        ppi_matrix=sasa / 10,
        esm_matrix=rng.uniform(0, 1, (N, N)),
        go_matrix=rng.uniform(0, 1, (N, N)),
        ground_truth_order=[0, 1, 2, 3],
    )

    print("=== Greedy (SASA only) ===")
    r = greedy_assembly(cplx, w_sasa=1.0)
    print(f"  Order: {r.predicted_order}  τ={r.kendall_tau:.3f}")

    print("=== Ensemble (all terms) ===")
    _, stats = ensemble_assembly(cplx, n_runs=20, w_sasa=0.25, w_ppi=0.25, w_esm=0.25, w_go=0.25)
    print(f"  Mean τ={stats['mean_tau']:.3f}  Unique pathways: {stats['n_unique_pathways']}")

    print("=== Ablation table ===")
    rows = ablation_table(cplx, n_stochastic=20)
    header = f"{'Condition':<35} {'Greedy τ':>10} {'Mean τ':>10} {'Paths':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        tau_g = f"{row['greedy_tau']:.3f}" if row['greedy_tau'] is not None else "  N/A"
        tau_s = f"{row['mean_stochastic_tau']:.3f}" if row['mean_stochastic_tau'] is not None else "  N/A"
        print(f"{row['condition']:<35} {tau_g:>10} {tau_s:>10} {row['n_unique_pathways']:>7}")


if __name__ == "__main__":
    _toy_demo()
