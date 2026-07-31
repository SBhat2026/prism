# PRISM scaling roadmap: from small complexes to the NPC

*Scoping note for extending assembly-order prediction to n > 40 subunits.
Grounded in a PDB coverage census (this session) and the subfield map.*

---

## 0. EVIDENCE UPDATE — what the scaling ablation proved (658 train / 165 test_small / 68 test_large frozen)

*This section supersedes the priors below where they conflict. Power caveat from the
run: 97% of labels are tier-2, 64% prefix-only, test_large is only 68 complexes —
treat any difference under ~0.05 top-1 as noise.*

**OVERTURNED — "ESM-2 carries assembly-order signal" (my §5 open question, answered).**
At small n, ESM is doing almost all the work: `esm_only` 0.602 ≈ `v4_baseline` 0.610,
while `no_esm`/`interface_only` collapse to ~0.50 and the `esm_shuffled` control (0.466)
falls *below* no_esm — so it is **chain-specific embedding identity**, not extra
dimensionality. BUT this advantage is **entirely in-distribution**: ESM top-1 decays
0.769 → 0.187 across size bins, vs interface_only 0.519 → 0.279. **Above n=21 no trained
model beats the untrained BSA heuristic** (bsa_teacher 0.266 > interface_only 0.254 >
all_features 0.214 > v4_baseline 0.196 > esm_only 0.194).
→ **Consequence: do NOT invest in ESM scale-up (was implicit in Lever 1/represention work).
It buys nothing where you're headed.** The large-n regime needs structure/physics priors,
not richer sequence embeddings.

**CONFIRMED — chirality blind spot is real and architectural (my §2 claim).**
v4 distance channels are bit-identical under reflection (max Δ = 0.00e+00). The invariant-
features diagnosis was correct. BUT the interim fix isn't paying yet: `v4_plus_chiral`
0.589 vs 0.610 (small) and τ_sym 0.454 vs 0.405 (large) — mixed, within noise on 68
complexes. → Chiral descriptors are necessary-not-sufficient; likely needs the full
equivariant backbone (was Lever 1 full) or more large-n labels to show signal.

**CONFIRMED — most of the "ring failure" is a scoring artifact (my §2 rings claim).**
v4 τ = 0.080 raw → 0.444 under ring-aware scoring, BUT random also gains +0.32 from ring
scoring, so the honest **margin-over-random is only +0.07**. → Symmetry-aware scoring
(Lever 6) is *necessary* to measure anything at large n, but it does **not by itself rescue
the model** — the model really is near-random out of distribution.

**STRONGEST LEVER — hierarchical/modular assembly (Lever 2) confirmed promising.**
Module decomposition takes mean n 31.3 → **effective n 10.4 across 4.2 modules**. Caveat:
ground-truth assembly stays inside a single module only **59% of steps at n=21–40**, so
modules are only roughly assembly-aligned. The flat-vs-hierarchical payoff test was running
at report time — that result decides whether Lever 2 is *the* answer.

**RISK on Lever 4 (BSA teacher) — partially validated.** The teacher's *free-running* order
agrees with labels at only τ≈0.00 step-wise (it's fine, 0.486, on the coarser metric). So
distilling it teaches a **local BSA preference, not a global pathway**. Still useful as a
weak prior — but don't expect it to supply true pathway order.

**Reprioritization the run's data supports:** Lever 2 (hierarchical) + Lever 3 (curriculum)
first; Lever 6 (symmetry-aware scoring) as the measurement backbone; **drop ESM scale-up
entirely**; Lever 1 chirality fix continues but needs the equivariant backbone to prove out.
The biology below (evolutionary interface age, assembly factors, cotranslational timing) is
the most promising source of the *new large-n signal* the ablation shows is currently missing.

---

## 0b. EVIDENCE UPDATE II — Lever 2 resolved, and a new Lever 7

*Added after §0. Two things landed since: the flat-vs-hierarchical test that was
still running finished, and a second diagnostic pass
(`diagnose_quaternary.py` → `results/diagnostics/quaternary.json`) went after the
chirality/rings question directly. Same power caveat as §0: test_large is 68
complexes.*

**RESOLVED — Lever 2 does NOT pay off as zero-shot weight reuse.**
`results/ablation_scaling/hierarchical.json`, `v4_plus_chiral`, same 68 large
complexes, identical weights at both levels:

| | τ | τ_sym | τ_ring | seconds |
|---|---|---|---|---|
| flat | 0.132 | **0.460** | **0.491** | 9.6 |
| hierarchical | 0.135 | 0.428 | 0.461 | 11.8 |

Module decomposition really does take mean n 31.3 → 10.4, but running the same
scorer at both levels *loses* ~0.03 τ_sym and costs 23% more wall-clock. §0
called Lever 2 "the strongest lever" on the strength of the n-reduction alone;
the payoff test does not support that, and the reason is visible in the same
report — ground-truth assembly stays inside a module only 59% of steps at
n=21–40 and 50% at n=41+, so a two-level decomposition imposes a constraint the
labels violate half the time. **Lever 2 is not dead, but it is now a *training*
proposal (a model trained at the module level, with modules that are learned or
assembly-aligned rather than modularity-optimal), not the free win it looked
like.** Do not spend on it before Lever 7.

**NEW — Lever 7: the chirality fix was aimed at the wrong quantity.**
§0 recorded that `v4_plus_chiral` was a wash and concluded the interim
descriptors need the full equivariant backbone to prove out. A cheaper
explanation turns out to fit: the interim descriptors are pseudoscalars of the
*chains* (det[û_ij, v_i, v_j] over whole-chain PCA axes), not of the *assembly*.
The quantity with a physical handedness is the screw operator that maps one copy
of a subunit onto the next — rotation θ about an axis plus a rise d along it,
where sign(d) is the handedness of the ring/helix and is exactly 0 for a flat
ring. Measured over 1,826 same-orbit chain pairs:

- rise sign-flip residual under reflection = **0.00e+00** (exact pseudoscalar),
  while θ and the interface-overlap descriptor are **exactly invariant** — the
  old descriptor flipped everything, including quantities that must not flip.
- **97 of 120** complexes have a nonzero rise, mean |d| = 13.8 Å. So the corpus
  is chiral in this sense almost everywhere, and the signal is large. The old
  descriptor was manufacturing handedness out of PCA noise on the achiral
  majority, which is a plausible reason it measured as a wash.

**NEW — large-n is a different *interface regime*, not just a bigger one.**
Isologous (2-fold, self-complementary, growth-terminating) vs heterologous
(head-to-tail, ring/helix-propagating) typing across 393 complexes:

| n | homomeric contacts | isologous | heterologous | helical | mean overlap |
|---|---|---|---|---|---|
| 2-4 | 17 | 0.235 | 0.059 | 0.647 | 0.169 |
| 5-10 | 587 | 0.334 | 0.291 | 0.327 | 0.303 |
| 11-20 | 1319 | 0.130 | 0.452 | 0.302 | 0.195 |
| 21-40 | 2147 | **0.052** | **0.509** | 0.390 | 0.064 |
| 41+ | 1085 | 0.076 | 0.236 | **0.673** | 0.072 |

Small complexes are isologous; large ones are heterologous and helical. The
training distribution therefore teaches the *opposite* interface physics from
the one the target regime runs on. This is a concrete, measured mechanism for
the distribution shift §0 could only observe as a number going down.

**NEW — the loss has been fitting noise, and the amount scales with n.**
At each supervised step, count how many *interchangeable* chains the annotated
next subunit is drawn from. Exact-index cross-entropy against one of k identical
copies has an irreducible floor of log k nats that a perfect model still pays,
whose gradient is pure noise:

| n | ambiguous steps | mean k | max k | CE floor (nats) | after contact restriction |
|---|---|---|---|---|---|
| 2-4 | 3.0% | 1.03 | 3 | 0.022 | 0.018 |
| 5-10 | 23.3% | 1.27 | 5 | 0.175 | 0.145 |
| 11-20 | 57.5% | 2.27 | 11 | 0.605 | 0.531 |
| 21-40 | **73.6%** | 4.74 | 16 | **1.146** | 1.017 |
| 41+ | **73.5%** | 8.14 | 23 | **1.487** | 1.405 |

PRISM has scored with `tau_mod_symmetry` since Lever 6, but the *objective* never
got the same treatment: `train_gnn._get_orders` only quotients symmetry for
whole-complex homomers, which is the rare case — every large assembly is a
heteromer *containing* symmetric orbits. Above n=21 roughly three quarters of the
training signal is an unanswerable question. (These are lower bounds: 3–8% of
chains carry no UniProt annotation and every corpus sequence field is empty, so
`build_orbits` scores those chains as singletons.)

**REFUTED (mine) — mean-pooling over the assembled core is not the size bug.**
The scorer feeds its head `edge_feats[cand, assembled].mean(1)`, and the obvious
story was that the mean dilutes the one interface that matters. It does not
change the *ranking*: at a fixed step the divisor is the same for every
candidate, so mean and sum give identical picks (measured: they agree to the
digit), and max is very slightly worse in every size bin. Recorded because the
same measurement produced a sharper finding:

| n | mean-pool top-1 | max-pool | Δ |
|---|---|---|---|
| 11-20 | 0.337 | 0.326 | −0.010 |
| 21-40 | **0.294** | 0.264 | −0.030 |
| 41+ | 0.107 | 0.105 | −0.002 |

Buried area alone scores 0.294 at n=21–40 — matching the BSA teacher and beating
every *trained* condition in §0 (`v4_baseline` 0.210). The statistic was never
missing from the model's input. Training is driving the model *off* a rule it can
already represent, which points at the objective above, not at the architecture.

**Reprioritization this second pass supports:**
1. **Lever 7 loss** — orbit-marginal cross-entropy. Removes 1.15–1.49 nats of
   pure label noise where the model is currently near-random. Highest expected
   value, ~30 lines, no new parameters.
2. **Lever 7 features** — screw handedness, isologous/heterologous typing, ring
   adjacency. Provably correct pseudoscalars, live dynamic range on the corpus,
   and they encode the interface regime the census says large-n actually runs on.
3. **Lever 6** stays the measurement backbone.
4. **Lever 3** (curriculum) unchanged — still cheap, still untried.
5. **Lever 2** demoted from "the answer" to "a training-time proposal".
6. **Lever 1 full** (equivariant backbone) — hold. Its motivating evidence was
   `v4_plus_chiral` being a wash, and there is now a cheaper explanation for that
   which has not been tested yet.
7. **ESM scale-up** — still dropped, per §0.

---

## 1. The core tension, stated precisely

Two facts collide:

- **Targets are huge and symmetric.** The nuclear pore complex is ~552+ proteins
  built from ~30 nucleoporins in 8-fold radial symmetry; the bacterial flagellum
  is tens of thousands of copies of a handful of proteins in helical symmetry.
- **Direct training data at that scale barely exists.** Of 220,569 protein-only
  PDB entries, **92.8% have ≤4 chains** and only **1.3% have >40 chains** (~2,900
  entries). See `fig2_pdb_coverage.png`.

The good news the census also revealed: large-complex structures are **68% cryo-EM
and growing ~3× over the last decade** (725 new ≥21-chain entries in 2025 alone).
The data is scarce but not static — and it lives in cryo-EM/EMDB, not X-ray.

---

## 2. Will the current model generalize to n > 40 on its own?

**Almost certainly not — and your two known failure modes (rings, chirality) are
the reason, not a coincidence.** They are not separate bugs; they are the
small-scale shadow of the large-scale problem.

### Why rings ARE the scaling problem
A cyclic/symmetric arrangement has **no distinguished first subunit** — assembly
order is only defined *up to the symmetry group*. A model trained to emit a single
linear order is being asked to break a symmetry the data doesn't uniquely resolve,
so it thrashes. Large complexes are *dominated* by symmetric sub-assemblies (NPC
8-fold, flagellar rings are C-symmetric, proteasome, GroEL, virus capsids). The
ring failure at small n is the entire problem at large n.

### Why chirality is (likely) architectural, not data-limited
If PRISM's node/edge features are distance- or contact-based (i.e. **SE(3)-
invariant**), the network **cannot in principle distinguish an assembly from its
mirror image** — the features are identical. No amount of data fixes a
representational blind spot. Large assemblies are strongly chiral (helical
flagella, NPC spoke tilt), so this bites hardest exactly where you're headed.

### Why size extrapolation fails independently
Even setting those aside, GNNs degrade off-distribution in graph size because of:
- **Combinatorial target blow-up** — orderings scale as n!; a model good on n≤10
  has sampled a space of ~10!, and n=40 is 40! ≈ 10^47 unseen.
- **Oversmoothing / bounded receptive field** — deeper propagation to cover a
  larger graph washes out node distinctions.
- **Degree- and density-distribution shift** between small and large complexes.

---

## 3. Six levers, roughly in priority order

### Lever 1 — Make the architecture symmetry- and chirality-aware *(highest leverage; fixes both known failures)*
- Replace invariant features with an **E(3)/SE(3)-equivariant** backbone (e.g.
  EGNN, e3nn/Tensor-Field Networks, or the vector-feature track used in GVP-GNN).
  Equivariant vector features represent handedness natively → chirality solved
  at the representation level.
- Add **explicit chiral descriptors** as a cheap interim fix: signed dihedral /
  improper-torsion features, or the sign of the triple product of interface
  normal vectors. Even on an invariant backbone this injects handedness.
- **Detect symmetry first, predict on the asymmetric unit.** Run a symmetry
  detection pass (AnAnaS, or PDB's assembly symmetry annotation), predict the
  order *within one asymmetric unit*, then propagate by the symmetry operators.
  This makes rings well-posed and collapses effective n (NPC → predict 1 spoke,
  tile 8×).

### Lever 2 — Hierarchical / coarse-grained assembly *(the key to n>40 efficiency)*
Huge complexes don't assemble atom-by-atom; **preformed subcomplexes dock as
units** (NPC Y-complex, flagellar C-ring vs rod vs hook). Mirror that:
- Two-level model: predict order *within* modules, then order *of* modules.
- Effective n per prediction stays small (~5–15) even when the whole complex is
  500+. This is both an accuracy and an efficiency win — it caps the receptive
  field and the factorial output space.
- Module definitions can come from the 3D-complex hierarchy or from graph
  community detection on the interface graph.

### Lever 3 — Curriculum learning on size
Train small→large: start on 2–8-mers where labels are dense, then progressively
admit larger complexes. Standard remedy for graph-size extrapolation; cheap to try.

### Lever 4 — Weak / silver labels to beat the data wall
Assembly-*order* ground truth is far scarcer than structures. Manufacture it:
- **BSA-heuristic teacher (Marsh/Levy):** the "largest buried interface forms
  first" rule gives a weak assembly-order label for *any* structure. Pre-train /
  distill against it, then fine-tune on the rare gold labels. Turns all 220k PDB
  entries into training signal.
- **AlphaFold3 / Boltz-2 silver structures:** co-fold large complexes you lack
  atomic structures for, derive interface graphs + BSA-order labels from the
  predictions. (You already consume AF structures — this extends it to labels.)
- **Bryant et al. AF+Monte-Carlo assembly paths** as a comparator/label source.

### Lever 5 — Mine the data that actually exists at scale
Absolute counts are small but concentrated in specific resources — go where the
big complexes live:
- **EMDB / cryo-EM** — 68% of ≥21-chain structures; the growing frontier.
- **3D Complex (Levy lab)** — hierarchical quaternary-structure classification;
  *the* resource for assembly topology and symmetry.
- **Complex Portal, CORUM** — curated complex composition (edge sets / membership).
- **PDB assembly symmetry annotations** — free symmetry-group labels.

### Lever 6 — Evaluation that respects symmetry
You cannot score n>40 with exact-order accuracy — it will look like failure even
when correct-up-to-symmetry. Define:
- **Order-agreement modulo the symmetry group** (canonicalize both predicted and
  true order under the detected symmetry before comparing).
- Held-out **large-complex test split** (freeze all ≥N-chain complexes out of
  training) to measure true extrapolation, not interpolation.
- Report **module-level** and **residue/interface-level** accuracy separately.

---

### Lever 7 — Chirality and binding patterns in nonstandard quaternary structure *(added by §0b; now the top priority)*
Rings and general homomers are not a special case to be handled — above n=21
they are the whole corpus. Three parts, each independently ablatable:

- **Symmetry-quotiented objective** (`src/training/orbit_loss.py`). Replace
  exact-index cross-entropy with the marginal likelihood over the orbit of the
  true subunit, `−log Σ_{j ∈ orbit ∩ remaining} p_j`. This is the exact MLE
  under a label that is censored within the orbit, reduces *identically* to plain
  cross-entropy when the orbit is a singleton, and costs one logsumexp. Narrow
  the target set to orbit members touching the assembled core when a contact
  matrix is available — "predict on the asymmetric unit, tile by symmetry"
  (Lever 1) applied to the *loss* instead of only to inference.
- **Screw-operator descriptors** (`src/features/quaternary.py`). The handedness
  of an assembly lives in the operator that maps one copy onto the next, not in
  the copies. Superpose copy i onto copy j, decompose the rigid motion into
  (θ, axis, rise d); sign(d) flips under reflection, is symmetric under swapping
  i and j, and is exactly 0 for a flat ring — which is the correct answer, since
  a planar C_k ring genuinely has no handedness.
- **Interface-pattern typing.** Jaccard overlap of the two interface residue sets
  gives the isologous/heterologous distinction (Monod; Levy & Teichmann) that
  determines whether a contact can propagate into a ring or terminates at a
  dimer, plus ring-adjacency edges that turn "which of the 8 identical copies
  comes next" into "extend the existing arc or nucleate a new one".

Ablation conditions: `v4_plus_quaternary` (features), `v4_orbit_loss` (loss),
`v4_frontier` (architecture), `symmetry_full` (all), `symmetry_no_esm`.

## 4. Suggested sequencing

1. **Diagnose first (1–2 weeks).** Build the symmetry-aware eval + a held-out
   large-complex split. Quantify *how* current PRISM degrades with n and whether
   ring/chirality errors dominate the large-n error. This tells you which lever
   pays off.
2. **Cheap architectural fixes.** Add chiral descriptors (Lever 1 interim) +
   symmetry-detect-then-tile. Re-measure.
3. **Hierarchical reformulation (Lever 2).** The structural change most likely to
   unlock n>40 at bounded cost.
4. **Data amplification (Levers 4–5)** in parallel: BSA-teacher pre-training +
   EMDB/3D-Complex mining.
5. **Equivariant backbone (Lever 1 full)** if interim chirality fixes plateau.

---

## 5. Open questions worth resolving early
- Is PRISM's failure on rings a *symmetry-breaking* artifact (fixable by
  predicting on the asymmetric unit) or a genuine representational gap?
- Do ESM-2 sequence embeddings even carry the signal for assembly order, or is
  the order information almost entirely in the *structure/interface geometry*?
  (An ablation answers this and tells you where to invest.)
- Is the target "the single true in-vivo pathway" or "a plausible/energetically
  monotonic pathway"? The second is far better-posed and better-supplied with
  weak labels.
