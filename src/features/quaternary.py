"""
quaternary.py — Binding-pattern and handedness descriptors for *nonstandard*
quaternary structure: rings, helices, and general homomers.

Lever 7 of PRISM_scaling_roadmap.md. Written after the scaling ablation showed
that (a) above n=21 no trained condition beats the untrained BSA heuristic and
(b) the generic pseudoscalars in `chiral_descriptors.py`, while provably
reflection-antisymmetric, bought nothing measurable.

Why the generic chiral features were not enough
-----------------------------------------------
`chiral_descriptors.py` builds handedness out of *whole-chain PCA axes*:
det[û_ij, v_i, v_j]. For a homomer that quantity is dominated by how the two
copies' inertia tensors happen to be oriented, which is a weak and noisy proxy
for the thing that actually has a handedness — **the symmetry operator that maps
one copy onto the next**. This module computes that operator directly.

The three descriptors
---------------------
1. **Screw decomposition of the copy-to-copy operator.**
   Superpose copy i onto copy j (they are the same protein, so this is exact up
   to conformational noise) to get a rigid motion x ↦ Rx + t. Every rigid motion
   with det R = +1 is a screw: a rotation by θ about an axis n̂ plus a rise
   d = t·n̂ *along* that axis. Then

       θ  — how far around the ring one step takes you (2π/k for a clean C_k)
       d  — the rise per subunit: 0 for a flat ring, ≠0 for a helix/lock-washer
       sign(d) — **the handedness of the assembly**

   `sign(d)` is a true pseudoscalar. Under a reflection M (det M = −1) the
   operator becomes (MRM, Mt); the antisymmetric part of MRM gives axis
   n̂′ = −M n̂, so d′ = (Mt)·(−M n̂) = −(t·n̂) = −d while θ′ = θ. Reflection flips
   the sign and leaves everything else alone — exactly what a handedness feature
   must do. It is also swap-symmetric: the inverse operator has axis −n̂ and
   translation −Rᵀt, whose dot product is the same d, so the feature is a
   property of the *pair*, not of the ordering of the pair.

   Crucially it is **zero for a planar ring** — which is correct. A flat C_k ring
   genuinely has no handedness, and the old det[û, v_i, v_j] feature was
   inventing one out of PCA noise. That is a plausible reason `v4_plus_chiral`
   measured as a wash: it was adding a noisy signal on the achiral majority to
   buy a weak signal on the chiral minority.

2. **Isologous vs heterologous interface type (Monod; Levy/Teichmann).**
   The single most predictive thing about a homomeric interface is whether the
   two sides use the *same* patch of surface:

       isologous    — 2-fold symmetric, self-complementary, the same residues on
                      both sides. Caps growth: an isologous contact can only ever
                      make a dimer.
       heterologous — head-to-tail, disjoint patches. Propagates: repeated
                      application builds a C_n ring, or an unbounded helix if it
                      does not close.

   This is computed as the Jaccard overlap of the two interface residue-number
   sets, which is well defined precisely because homomer copies share numbering.
   It is the feature that tells the model *whether a subunit can extend an
   existing ring at all* — the binding-pattern question, not the affinity
   question that buried area already answers.

3. **Ring topology.** Which pairs are cyclic neighbours in a detected ring, plus
   each chain's radius and axial offset about the ring axis. Ring adjacency
   turns "which of the 8 identical copies comes next" — an unanswerable question
   — into "extend the existing arc or nucleate a new one", which is well posed
   and canonical under the symmetry group.

Feature layout
--------------
Node (4d), per chain:
  0  in_ring            1.0 if the chain belongs to a detected ring orbit
  1  fold_norm          k/12, clipped — the fold of its ring
  2  ring_radius_norm   r/(r+25 Å), scale-free distance from the ring axis
  3  axial_offset_norm  signed offset along the oriented ring axis / 30 Å
                        (invariant, not chiral — it separates stacked rings)

Edge (5d), per ordered pair (i, j):
  0  isologous          Jaccard of the two interface residue sets, 0 for
                        heteromeric or non-contacting pairs
  1  screw_angle        θ/π ∈ [0, 1]
  2  screw_rise         clip(d / 30 Å, −1, 1)      — signed, PSEUDOSCALAR
  3  screw_handed       sign(d)·min(1, |d| / 2 Å)  — saturating handedness
  4  ring_adjacent      1.0 if cyclic neighbours in a detected ring

Everything is bounded in [−1, 1]. Missing coordinates ⇒ zeros (exact no-op), so
turning the block on can never break a complex that has no structure.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

QUAT_NODE_DIM = 4
QUAT_EDGE_DIM = 5

ROOT = Path(__file__).resolve().parents[2]
QUAT_CACHE = ROOT / "data" / "quaternary_cache"

# Interface definition: CA–CA within this distance counts as a residue contact.
# 10 Å on CA is the standard coarse proxy for an all-atom 5 Å heavy-atom contact.
CA_INTERFACE_CUTOFF = 10.0

_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate loading (with residue numbers — needed for isologous typing)
# ──────────────────────────────────────────────────────────────────────────────

def load_ca_with_resnums(
    pdb_path: str | Path,
    chain_ids: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Parse CA coordinates *and* residue numbers per chain.

    Returns {chain_id: (resnums (R,) int, coords (R, 3) float)}. Chains with
    fewer than 3 resolved CAs are dropped — they cannot define a frame.

    `symmetry.load_ca_coords` deliberately throws the numbering away; isologous
    typing needs it, because "the same patch of surface on both sides" is a
    statement about residue identity, not about position.
    """
    wanted = set(chain_ids)
    acc: dict[str, list] = defaultdict(list)
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
                    rn = int(line[22:26])
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                acc[ch].append((rn, xyz))
    except OSError:
        return {}

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ch, rows in acc.items():
        if len(rows) < 3:
            continue
        # Keep the first occurrence of each residue number (altloc / insertion safe).
        seen: dict[int, tuple] = {}
        for rn, xyz in rows:
            seen.setdefault(rn, xyz)
        rns = np.fromiter(seen.keys(), dtype=np.int64, count=len(seen))
        xs = np.asarray(list(seen.values()), dtype=np.float64)
        order = np.argsort(rns)
        out[ch] = (rns[order], xs[order])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 1. Screw decomposition of the copy-to-copy operator
# ──────────────────────────────────────────────────────────────────────────────

def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rigid motion taking P onto Q: returns (R, t, rmsd) with Q ≈ P @ R.T + t.

    Reflections are removed (the standard det-correction), so R is always a
    proper rotation — which is what makes the screw decomposition below valid.
    A reflected *complex* still produces a different R; it is the operator that
    carries the handedness, not this determinant.
    """
    pc, qc = P.mean(0), Q.mean(0)
    A = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    diff = (P - pc) @ R.T - (Q - qc)
    rmsd = float(np.sqrt((diff ** 2).sum(1).mean()))
    return R, t, rmsd


def screw_parameters(R: np.ndarray, t: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Chasles screw decomposition of x ↦ Rx + t.

    Returns (theta, rise, axis) with theta ∈ [0, π] measured right-handed about
    `axis`, and rise = t·axis the translation along it.

    Degenerate cases return theta=0, rise=|t| — a pure translation has no
    handedness, and reporting 0 is the honest answer rather than a random axis.
    """
    tr = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(tr))

    # Antisymmetric part of R is sin(theta) * [axis]_x, so this vector is
    # sin(theta)*axis — its direction fixes the right-handed sense of theta.
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    nw = float(np.linalg.norm(w))

    if nw < 1e-8:
        if theta < 1e-6:                       # identity / pure translation
            n = t / (np.linalg.norm(t) + 1e-12)
            return 0.0, float(np.linalg.norm(t)), n
        # theta ≈ π: the antisymmetric part vanishes. Recover the axis from the
        # symmetric part, R + I = 2 n nᵀ, whose dominant eigenvector is n. Its
        # sign is genuinely ambiguous — a 2-fold is its own inverse — so the
        # *signed* rise is not defined and 0 is the honest value. That is the
        # right answer physically too: an isologous 2-fold has no handedness.
        w2, v2 = np.linalg.eigh(R + np.eye(3))
        n = v2[:, int(np.argmax(w2))]
        n = n / (np.linalg.norm(n) + 1e-12)
        return theta, 0.0, n

    n = w / nw
    return theta, float(np.dot(t, n)), n


def pair_screw(
    ca_i: tuple[np.ndarray, np.ndarray],
    ca_j: tuple[np.ndarray, np.ndarray],
    min_common: int = 15,
) -> Optional[tuple[float, float, float]]:
    """
    Screw parameters of the operator taking chain i onto chain j, matched on
    shared residue numbers. Returns (theta, rise, rmsd), or None if the two
    chains share too few residues to superpose (i.e. they are not copies).
    """
    rn_i, x_i = ca_i
    rn_j, x_j = ca_j
    common, ii, jj = np.intersect1d(rn_i, rn_j, return_indices=True)
    if len(common) < min_common:
        return None
    R, t, rmsd = kabsch(x_i[ii], x_j[jj])
    theta, rise, _ = screw_parameters(R, t)
    return theta, rise, rmsd


# ──────────────────────────────────────────────────────────────────────────────
# 2. Isologous vs heterologous interface typing
# ──────────────────────────────────────────────────────────────────────────────

def interface_residues(
    ca_i: tuple[np.ndarray, np.ndarray],
    ca_j: tuple[np.ndarray, np.ndarray],
    cutoff: float = CA_INTERFACE_CUTOFF,
) -> tuple[set[int], set[int]]:
    """Residue numbers on each side that lie within `cutoff` of the other chain."""
    rn_i, x_i = ca_i
    rn_j, x_j = ca_j
    d2 = ((x_i[:, None, :] - x_j[None, :, :]) ** 2).sum(-1)
    hit = d2 <= cutoff * cutoff
    return set(rn_i[hit.any(1)].tolist()), set(rn_j[hit.any(0)].tolist())


def isologous_score(
    ca_i: tuple[np.ndarray, np.ndarray],
    ca_j: tuple[np.ndarray, np.ndarray],
    cutoff: float = CA_INTERFACE_CUTOFF,
) -> float:
    """
    Jaccard overlap of the two interface residue-number sets.

    ≈1  isologous  — both copies present the same surface patch (a 2-fold,
                     self-complementary contact; terminates growth at a dimer)
    ≈0  heterologous — disjoint patches, head-to-tail (propagates into a ring
                     or a helix)

    Only meaningful when i and j are copies of the same protein, so callers must
    gate on orbit membership. Returns 0.0 when the chains do not touch.
    """
    a, b = interface_residues(ca_i, ca_j, cutoff)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def classify_interface(iso: float, theta: float, rise: float) -> str:
    """
    Human-readable quaternary interface label, for diagnostics and the viewer.
    Thresholds follow the Levy/Teichmann convention: isologous contacts sit near
    a 2-fold (θ ≈ π) and reuse the same patch.
    """
    if iso >= 0.35 and theta > 2.0:
        return "isologous"
    if iso < 0.20 and theta > 0.15:
        return "helical" if abs(rise) > 2.0 else "heterologous"
    if theta <= 0.15:
        return "translational"
    return "mixed"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Ring topology and axis orientation
# ──────────────────────────────────────────────────────────────────────────────

def orient_axis(
    axis: np.ndarray,
    ring_pts: np.ndarray,
    outside_pts: Optional[np.ndarray],
    all_pts: np.ndarray,
) -> np.ndarray:
    """
    Give a ring axis a reproducible direction using a *polar* (reflection-
    invariant) rule, so that any signed quantity measured against it is a clean
    pseudoscalar rather than a coin flip.

    Rule, in order:
      1. Point from the ring centroid toward the centroid of the chains *outside*
         the ring — a difference of centroids, built only from positions, so a
         mirror image maps it to the mirrored vector consistently.
      2. If the ring *is* the whole complex, use the skewness of the projections
         of every chain centroid on the axis. Skewness is a scalar function of
         the projections; reflecting through a plane containing the axis leaves
         projections unchanged, so the orientation survives.
      3. If both are degenerate (a perfectly symmetric flat ring), any direction
         is as good as another — and the rise it would measure is 0 anyway.
    """
    if outside_pts is not None and len(outside_pts):
        v = outside_pts.mean(0) - ring_pts.mean(0)
        if np.linalg.norm(v) > 1e-3 and abs(float(np.dot(v, axis))) > 1e-3:
            return axis if float(np.dot(v, axis)) > 0 else -axis

    proj = (all_pts - all_pts.mean(0)) @ axis
    s = float(np.mean(proj ** 3))
    if abs(s) > 1e-6:
        return axis if s > 0 else -axis
    return axis


def ring_neighbour_pairs(ring: Sequence[int]) -> list[tuple[int, int]]:
    """Cyclic neighbour pairs of a ring given in angular order."""
    k = len(ring)
    if k < 3:
        return []
    return [(ring[i], ring[(i + 1) % k]) for i in range(k)]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_quaternary_features(
    pdb_id: str,
    chain_ids: Sequence[str],
    pdb_path: Optional[str],
    contact_matrix: Optional[np.ndarray] = None,
    uniprot_ids: Optional[Sequence[str]] = None,
    sequences: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = QUAT_CACHE,
    max_pairs: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (node (N, 4), edge (N, N, 5)) float32, both bounded in [−1, 1].

    `contact_matrix` (any (N, N) nonneg matrix — buried area works) restricts the
    O(N²) screw/interface work to pairs that actually touch. Without it every
    same-orbit pair is considered, which is fine up to a few dozen chains and
    capped at `max_pairs` beyond that.

    Zeros are returned whenever coordinates are unavailable, so callers can turn
    the block on unconditionally.
    """
    n = len(chain_ids)
    node = np.zeros((n, QUAT_NODE_DIM), np.float64)
    edge = np.zeros((n, n, QUAT_EDGE_DIM), np.float64)
    if not pdb_path or n == 0:
        return node.astype(np.float32), edge.astype(np.float32)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{pdb_id}_quat.npz"
        if cache_path.exists():
            try:
                d = np.load(cache_path)
                nf, ef = d["node"], d["edge"]
                if nf.shape == (n, QUAT_NODE_DIM) and ef.shape == (n, n, QUAT_EDGE_DIM):
                    return nf.astype(np.float32), ef.astype(np.float32)
            except Exception:
                pass

    from src.evaluation.symmetry import detect_symmetry, build_orbits

    ca = load_ca_with_resnums(pdb_path, chain_ids)
    if len(ca) < 2:
        return node.astype(np.float32), edge.astype(np.float32)

    cents = np.zeros((n, 3))
    have = np.zeros(n, dtype=bool)
    for i, c in enumerate(chain_ids):
        if c in ca:
            cents[i] = ca[c][1].mean(0)
            have[i] = True

    orbits = build_orbits(uniprot_ids, sequences, n)
    oid = {idx: k for k, orb in enumerate(orbits) for idx in orb}

    # ── Ring topology ────────────────────────────────────────────────────────
    sym = detect_symmetry(pdb_id, chain_ids, uniprot_ids, sequences, pdb_path)
    ring_of: dict[int, list[int]] = {}
    for ring in sym.ring_orbits:
        members = [i for i in ring if have[i]]
        if len(members) < 3:
            continue
        pts = cents[members]
        axis = np.asarray(sym.axis, dtype=np.float64) if sym.axis else None
        if axis is None or np.linalg.norm(axis) < 1e-6:
            c = pts - pts.mean(0)
            axis = np.linalg.svd(c, full_matrices=False)[2][-1]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        member_set = set(members)
        outside = [i for i in range(n) if have[i] and i not in member_set]
        axis = orient_axis(axis, pts, cents[outside] if outside else None, cents[have])

        centre = pts.mean(0)
        for i in members:
            rel = cents[i] - centre
            ax_off = float(np.dot(rel, axis))
            radius = float(np.linalg.norm(rel - ax_off * axis))
            node[i, 0] = 1.0
            node[i, 1] = min(1.0, len(ring) / 12.0)
            node[i, 2] = radius / (radius + 25.0)
            node[i, 3] = float(np.clip(ax_off / 30.0, -1.0, 1.0))
            ring_of[i] = ring

        for a, b in ring_neighbour_pairs(ring):
            if have[a] and have[b]:
                edge[a, b, 4] = 1.0
                edge[b, a, 4] = 1.0

    # ── Pairwise screw + isologous typing, same-orbit pairs only ─────────────
    if contact_matrix is not None and np.asarray(contact_matrix).shape == (n, n):
        cm = np.array(contact_matrix, dtype=np.float64, copy=True)
        np.fill_diagonal(cm, 0.0)
    else:
        cm = None

    pairs: list[tuple[int, int]] = []
    for orb in orbits:
        if len(orb) < 2:
            continue
        for a_i in range(len(orb)):
            for b_i in range(a_i + 1, len(orb)):
                i, j = orb[a_i], orb[b_i]
                if not (have[i] and have[j]):
                    continue
                if cm is not None and cm[i, j] <= 0 and edge[i, j, 4] == 0:
                    continue
                pairs.append((i, j))
    if len(pairs) > max_pairs:
        # Keep the strongest contacts — the screw of a pair that barely touches
        # is not a load-bearing descriptor anyway.
        key = (lambda p: cm[p]) if cm is not None else (lambda p: 0.0)
        pairs = sorted(pairs, key=key, reverse=True)[:max_pairs]

    for i, j in pairs:
        ci, cj = ca.get(chain_ids[i]), ca.get(chain_ids[j])
        if ci is None or cj is None:
            continue
        sc = pair_screw(ci, cj)
        if sc is None:
            continue
        theta, rise, _rmsd = sc
        iso = isologous_score(ci, cj)

        theta_n = float(np.clip(theta / np.pi, 0.0, 1.0))
        rise_n = float(np.clip(rise / 30.0, -1.0, 1.0))
        hand = float(np.sign(rise) * min(1.0, abs(rise) / 2.0))

        for a, b in ((i, j), (j, i)):
            edge[a, b, 0] = iso        # symmetric
            edge[a, b, 1] = theta_n    # symmetric
            edge[a, b, 2] = rise_n     # symmetric (see module docstring)
            edge[a, b, 3] = hand

    node = np.clip(np.nan_to_num(node), -1.0, 1.0).astype(np.float32)
    edge = np.clip(np.nan_to_num(edge), -1.0, 1.0).astype(np.float32)

    if cache_path is not None:
        try:
            np.savez_compressed(cache_path, node=node, edge=edge)
        except Exception:
            pass
    return node, edge


def mirror_probe(
    pdb_id: str,
    chain_ids: Sequence[str],
    pdb_path: str,
    uniprot_ids: Optional[Sequence[str]] = None,
    sequences: Optional[Sequence[str]] = None,
) -> dict:
    """
    Diagnostic: recompute the screw descriptors on mirrored coordinates and
    check that the handedness channels sign-flip while the scalar channels
    (θ, isologous overlap) are preserved.

    A correct implementation gives:
        mean |rise + rise_mirror|  ≈ 0     (clean sign flip)
        mean |theta − theta_mirror| ≈ 0    (rotation angle is achiral)
    """
    ca = load_ca_with_resnums(pdb_path, chain_ids)
    if len(ca) < 2:
        return {"ok": False, "reason": "missing coords"}

    from src.evaluation.symmetry import build_orbits

    orbits = build_orbits(uniprot_ids, sequences, len(chain_ids))
    mir = {c: (rn, x * np.array([1.0, 1.0, -1.0])) for c, (rn, x) in ca.items()}

    rises, rises_m, thetas, thetas_m, isos, isos_m = [], [], [], [], [], []
    for orb in orbits:
        for a_i in range(len(orb)):
            for b_i in range(a_i + 1, len(orb)):
                ci, cj = chain_ids[orb[a_i]], chain_ids[orb[b_i]]
                if ci not in ca or cj not in ca:
                    continue
                s0 = pair_screw(ca[ci], ca[cj])
                s1 = pair_screw(mir[ci], mir[cj])
                if s0 is None or s1 is None:
                    continue
                thetas.append(s0[0]); rises.append(s0[1])
                thetas_m.append(s1[0]); rises_m.append(s1[1])
                isos.append(isologous_score(ca[ci], ca[cj]))
                isos_m.append(isologous_score(mir[ci], mir[cj]))

    if not rises:
        return {"ok": False, "reason": "no same-orbit pairs"}

    r, rm = np.array(rises), np.array(rises_m)
    t, tm = np.array(thetas), np.array(thetas_m)
    i0, i1 = np.array(isos), np.array(isos_m)
    chiral = np.abs(r) > 0.5           # pairs with a real rise
    return {
        "ok": True,
        "n_pairs": int(len(r)),
        "n_chiral_pairs": int(chiral.sum()),
        "mean_abs_rise": float(np.abs(r).mean()),
        "rise_sign_flip_residual": float(np.abs(r + rm).mean()),      # ≈0 ⇒ correct
        "theta_invariance_residual": float(np.abs(t - tm).mean()),    # ≈0 ⇒ correct
        "iso_invariance_residual": float(np.abs(i0 - i1).mean()),     # ≈0 ⇒ correct
        "rise_sign_flip_fraction": (
            float((np.sign(r[chiral]) * np.sign(rm[chiral]) < 0).mean()) if chiral.any() else 1.0
        ),
    }
