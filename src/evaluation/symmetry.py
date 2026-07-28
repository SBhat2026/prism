"""
symmetry.py — Symmetry detection, asymmetric-unit reduction, and symmetry-aware
evaluation for assembly-order prediction.

Implements Lever 1 (symmetry-detect-then-tile) and Lever 6 (evaluation that
respects symmetry) of PRISM_scaling_roadmap.md.

Core idea
---------
In a symmetric complex the assembly order is only defined *up to the symmetry
group*. Scoring an n>40 prediction with exact-order Kendall τ makes a correct
model look broken: swapping two chains that are related by a symmetry operator
is a zero-information "error".

We therefore define:

  orbits          — sets of chains that are interchangeable (same UniProt /
                    same sequence, i.e. the same crystallographic "entity").
  tau_mod_symmetry— Kendall τ after optimally relabelling chains *within* each
                    orbit. Within an exchangeable orbit the optimal relabelling
                    is the sorted-to-sorted match (proved below), so this is
                    exact and O(n log n), not a search over |G| permutations.
  cyclic detection— for an orbit whose centroids lie on a circle, recover the
                    C_k rotational order and the axis, so ring predictions can
                    be scored up to rotation *and* reflection.

Optimality of sorted-to-sorted matching
---------------------------------------
Kendall τ depends only on the relative ranks of the two orderings. If chains
{a1..ak} form an orbit, any bijection between predicted positions
P = {p1<...<pk} and true positions T = {t1<...<tk} is allowed. Assigning the
i-th smallest predicted position to the i-th smallest true position removes all
*within-orbit* discordant pairs (they become concordant by construction) and
leaves every cross-orbit pair untouched, so it maximises the concordant count.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
SYM_CACHE = ROOT / "data" / "symmetry_cache"


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate loading
# ──────────────────────────────────────────────────────────────────────────────

_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


def load_ca_coords(pdb_path: str | Path, chain_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Parse CA coordinates per chain. Returns {chain_id: (n_res, 3)}."""
    wanted = set(chain_ids)
    out: dict[str, list] = defaultdict(list)
    try:
        with open(pdb_path, errors="ignore") as fh:
            for line in fh:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                if line[17:20].strip() not in _AA3:
                    continue
                ch = line[21].strip() or "_"
                if ch not in wanted:
                    continue
                try:
                    out[ch].append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    except OSError:
        return {}
    return {c: np.asarray(v, dtype=np.float64) for c, v in out.items() if len(v) >= 3}


def chain_centroids(coords: dict[str, np.ndarray], chain_ids: Sequence[str]) -> Optional[np.ndarray]:
    """(N, 3) centroid per chain, or None if any chain is missing coordinates."""
    pts = []
    for c in chain_ids:
        arr = coords.get(c)
        if arr is None or len(arr) == 0:
            return None
        pts.append(arr.mean(axis=0))
    return np.asarray(pts)


# ──────────────────────────────────────────────────────────────────────────────
# Orbits (interchangeable chain sets)
# ──────────────────────────────────────────────────────────────────────────────

def build_orbits(
    uniprot_ids: Optional[Sequence[str]] = None,
    sequences: Optional[Sequence[str]] = None,
    n_chains: int = 0,
) -> list[list[int]]:
    """
    Group chain indices into orbits of interchangeable subunits.

    Priority: UniProt accession → exact sequence → singleton orbits.
    Empty/unknown UniProt falls through to the sequence key so that unannotated
    chains are not all merged into one bogus orbit.
    """
    n = n_chains or len(uniprot_ids or sequences or [])
    keys: list[str] = []
    for i in range(n):
        uid = (uniprot_ids[i].strip() if uniprot_ids and i < len(uniprot_ids) and uniprot_ids[i] else "")
        if uid:
            keys.append(f"U:{uid}")
            continue
        seq = (sequences[i] if sequences and i < len(sequences) else "") or ""
        keys.append(f"S:{seq}" if len(seq) >= 10 else f"I:{i}")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    return [sorted(v) for v in groups.values()]


def orbit_id_map(orbits: list[list[int]]) -> dict[int, int]:
    return {idx: oi for oi, orb in enumerate(orbits) for idx in orb}


# ──────────────────────────────────────────────────────────────────────────────
# Cyclic symmetry detection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SymmetryInfo:
    pdb_id: str
    n_chains: int
    orbits: list[list[int]]
    point_group: str = "C1"            # C1 / Ck / Dk (approximate)
    fold: int = 1                      # k for Ck
    axis: Optional[list[float]] = None
    ring_orbits: list[list[int]] = field(default_factory=list)  # orbits that form a ring, in angular order
    rmsd_circle: float = float("nan")  # circle-fit residual (Å); low = clean ring
    asymmetric_unit: list[int] = field(default_factory=list)
    effective_n: int = 0
    source: str = "coords"

    def to_dict(self) -> dict:
        return asdict(self)


def _fit_plane_normal(pts: np.ndarray) -> np.ndarray:
    """Unit normal of the best-fit plane through pts (smallest-variance PCA axis)."""
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[-1] / (np.linalg.norm(vt[-1]) + 1e-12)


def _angular_order(pts: np.ndarray, normal: np.ndarray) -> tuple[list[int], float, float]:
    """
    Project pts onto the plane ⟂ normal, return (indices sorted by angle,
    circle-fit rmsd, angular-spacing regularity in [0,1]).
    """
    c = pts.mean(axis=0)
    rel = pts - c
    # Orthonormal in-plane basis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, ref)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(normal, e1)

    x = rel @ e1
    y = rel @ e2
    r = np.hypot(x, y)
    rmsd = float(np.std(r))                     # Å spread of radii about the mean
    ang = np.arctan2(y, x)
    order = list(np.argsort(ang))

    k = len(pts)
    if k < 3:
        return order, rmsd, 0.0
    sorted_ang = np.sort(ang)
    gaps = np.diff(np.concatenate([sorted_ang, [sorted_ang[0] + 2 * np.pi]]))
    expected = 2 * np.pi / k
    regularity = float(max(0.0, 1.0 - np.mean(np.abs(gaps - expected)) / expected))
    return order, rmsd, regularity


def detect_symmetry(
    pdb_id: str,
    chain_ids: Sequence[str],
    uniprot_ids: Optional[Sequence[str]] = None,
    sequences: Optional[Sequence[str]] = None,
    pdb_path: Optional[str] = None,
    ring_regularity_thresh: float = 0.75,
    ring_rmsd_frac: float = 0.25,
    cache_dir: Optional[Path] = SYM_CACHE,
) -> SymmetryInfo:
    """
    Detect the (approximate) rotational symmetry of a complex.

    Sequence/UniProt identity gives the orbits for free — that alone makes the
    evaluation well-posed. Coordinates, when present, additionally decide
    whether an orbit is a genuine *ring* (C_k) so that ring predictions can be
    scored up to rotation and reflection.
    """
    n = len(chain_ids)
    orbits = build_orbits(uniprot_ids, sequences, n)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{pdb_id}_sym.json"
        if cache_path.exists():
            try:
                d = json.loads(cache_path.read_text())
                if d.get("n_chains") == n:
                    return SymmetryInfo(**d)
            except Exception:
                pass

    info = SymmetryInfo(
        pdb_id=pdb_id,
        n_chains=n,
        orbits=orbits,
        asymmetric_unit=[orb[0] for orb in orbits],
        effective_n=len(orbits),
        source="orbits_only",
    )

    if pdb_path:
        coords = load_ca_coords(pdb_path, chain_ids)
        cents = chain_centroids(coords, chain_ids)
        if cents is not None:
            info.source = "coords"
            best_fold, best_axis, best_reg, best_rmsd = 1, None, 0.0, float("nan")
            for orb in orbits:
                if len(orb) < 3:
                    continue
                pts = cents[orb]
                normal = _fit_plane_normal(pts)
                order, rmsd, reg = _angular_order(pts, normal)
                radius = float(np.mean(np.linalg.norm(pts - pts.mean(axis=0), axis=1))) + 1e-9
                is_ring = reg >= ring_regularity_thresh and (rmsd / radius) <= ring_rmsd_frac
                if is_ring:
                    info.ring_orbits.append([orb[i] for i in order])
                    if len(orb) > best_fold:
                        best_fold, best_axis, best_reg, best_rmsd = len(orb), normal, reg, rmsd
            if best_fold > 1:
                info.fold = best_fold
                info.axis = [float(v) for v in best_axis]
                info.rmsd_circle = float(best_rmsd)
                # Two coaxial rings of the same fold ⇒ likely dihedral
                coaxial = sum(1 for r in info.ring_orbits if len(r) == best_fold)
                info.point_group = f"D{best_fold}" if coaxial >= 2 else f"C{best_fold}"

    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps(info.to_dict()))
        except Exception:
            pass
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Symmetry-aware order comparison
# ──────────────────────────────────────────────────────────────────────────────

def _tau_from_positions(pred_pos: list[int], true_pos: list[int]) -> float:
    if len(pred_pos) < 2:
        return 1.0 if len(pred_pos) == 1 else 0.0
    stat, _ = kendalltau(pred_pos, true_pos)
    return 0.0 if (stat is None or np.isnan(stat)) else float(stat)


def restrict_to_gt(pred_order: Sequence[int], gt_order: Sequence[int]) -> list[int]:
    """
    Project a full-length prediction onto the (possibly partial) GT subset,
    preserving predicted relative order. Needed because 64% of the corpus has
    only a prefix of the assembly order annotated.
    """
    gt_set = set(gt_order)
    return [x for x in pred_order if x in gt_set]


def tau_exact(pred_order: Sequence[int], gt_order: Sequence[int]) -> float:
    """Plain Kendall τ on the GT subset — the current (symmetry-blind) metric."""
    pred = restrict_to_gt(pred_order, gt_order)
    if len(pred) != len(gt_order) or len(pred) < 2:
        return 0.0
    rank = {v: i for i, v in enumerate(pred)}
    return _tau_from_positions([rank[v] for v in gt_order], list(range(len(gt_order))))


def tau_mod_symmetry(
    pred_order: Sequence[int],
    gt_order: Sequence[int],
    orbits: list[list[int]],
) -> float:
    """
    Kendall τ after optimally relabelling chains within each orbit
    (sorted-to-sorted matching — see module docstring for why this is optimal).
    """
    pred = restrict_to_gt(pred_order, gt_order)
    if len(pred) != len(gt_order) or len(pred) < 2:
        return 0.0

    oid = orbit_id_map(orbits)
    pred_rank = {v: i for i, v in enumerate(pred)}
    true_rank = {v: i for i, v in enumerate(gt_order)}

    # Bucket the GT subset by orbit, then match sorted predicted positions to
    # sorted true positions inside each bucket.
    by_orbit: dict[int, list[int]] = defaultdict(list)
    for v in gt_order:
        by_orbit[oid.get(v, -1 - v)].append(v)

    aligned_pred: list[int] = [0] * len(gt_order)
    for members in by_orbit.values():
        p_sorted = sorted(pred_rank[v] for v in members)
        t_sorted = sorted(true_rank[v] for v in members)
        for p, t in zip(p_sorted, t_sorted):
            aligned_pred[t] = p

    return _tau_from_positions(aligned_pred, list(range(len(gt_order))))


def tau_mod_ring(
    pred_order: Sequence[int],
    gt_order: Sequence[int],
    sym: SymmetryInfo,
) -> float:
    """
    τ modulo the full detected group: orbit exchange *and*, for ring orbits,
    cyclic rotation + reflection of the GT labelling.
    """
    best = tau_mod_symmetry(pred_order, gt_order, sym.orbits)
    if not sym.ring_orbits:
        return best

    # Relabel the GT by rotating/reflecting each detected ring, take the best.
    for ring in sym.ring_orbits:
        k = len(ring)
        if k < 3:
            continue
        for rev in (False, True):
            base = list(reversed(ring)) if rev else list(ring)
            for shift in range(k):
                rot = base[shift:] + base[:shift]
                relabel = {ring[i]: rot[i] for i in range(k)}
                gt_r = [relabel.get(v, v) for v in gt_order]
                if len(set(gt_r)) != len(gt_r):
                    continue
                best = max(best, tau_mod_symmetry(pred_order, gt_r, sym.orbits))
    return best


def top1_accuracy_mod_symmetry(
    step_choices: Sequence[tuple[int, int]],
    orbits: list[list[int]],
) -> float:
    """
    step_choices: [(predicted_idx, gt_idx), ...] over assembly steps.
    A prediction counts as correct if it lands in the same orbit as the GT
    choice — i.e. the model picked a symmetry-equivalent subunit.
    """
    if not step_choices:
        return 0.0
    oid = orbit_id_map(orbits)
    hits = sum(1 for p, g in step_choices if oid.get(p, -1 - p) == oid.get(g, -2 - g))
    return hits / len(step_choices)


# ──────────────────────────────────────────────────────────────────────────────
# Asymmetric-unit reduction (Lever 1: predict on the ASU, tile by symmetry)
# ──────────────────────────────────────────────────────────────────────────────

def asymmetric_unit_indices(sym: SymmetryInfo) -> list[int]:
    """One representative chain per orbit — the sub-problem to actually predict on."""
    return [orb[0] for orb in sym.orbits]


def tile_order(asu_order: Sequence[int], sym: SymmetryInfo) -> list[int]:
    """
    Expand an asymmetric-unit order back to the full complex by emitting each
    orbit's members consecutively (ring orbits in detected angular order).
    """
    ring_lookup = {tuple(sorted(r)): r for r in sym.ring_orbits}
    out: list[int] = []
    for rep in asu_order:
        orb = next((o for o in sym.orbits if rep in o), [rep])
        key = tuple(sorted(orb))
        out.extend(ring_lookup.get(key, orb))
    return out


def size_bin(n: int) -> str:
    if n <= 4:
        return "2-4"
    if n <= 10:
        return "5-10"
    if n <= 20:
        return "11-20"
    if n <= 40:
        return "21-40"
    return "41+"
