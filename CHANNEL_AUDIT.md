# Why the viewer showed SASA moving and everything else pinned at 0.5

Two independent faults, with different causes and different fixes. They
compounded: the display bug hid the data bug, and the data bug made the display
bug look plausible.

Reproduce with:

```bash
python audit_channels.py --max_complexes 0            # A: data health (+ occlusion on a stored ckpt)
python audit_channels.py --train --max_complexes 0    # B: occlusion on a freshly trained model
python scripts/repair_ppi_cache.py --dry-run          # C: the PPI repair, without writing
```

---

## Fault 1 — the display bug (real, fixed)

`server/server.py` had two GNN code paths that emitted literal placeholders:

```python
"scores":   {"gnn": ..., "sasa": <real>, "esm": 0.5, "go": 0.5, "ppi": 0.5, ...},
"weighted": {"sasa": 0.0, "esm": 0.0, "go": 0.0, "ppi": 0.0},
```

Nobody had computed per-channel values for the GNN, so 0.5 was filled in as a
neutral. The frontend drew it as a half-full bar, indistinguishable from a
genuine mid-range score. SASA was the only channel with a real value, so SASA
was the only channel that appeared to move.

`hf_deploy/prism-space/app.py` — the deployed viewer — had a subtler version.
It computes all four channels for real, but `weighted` was **always**
`W_SASA·sasa`, `W_ESM·esm`, … even when `use_gnn` was true and the displayed
score was a GNN logit. The breakdown claimed to explain a score it did not
produce. On top of that, `_esm_score`, `_go_coherence` and `_ppi_score` each
return a 0.5 sentinel when their input is missing:

| channel | returns 0.5 when |
|---|---|
| `esm` | nothing assembled yet; or embeddings absent → random vectors → cos≈0 → (0+1)/2 = 0.5 |
| `go`  | nothing assembled yet; or the GO term union is empty (no annotations) |
| `ppi` | nothing assembled yet; or STRING lookup fails / has no UniProt to query |
| `sasa` | never — it is derivable from structure alone |

That last row is the whole story: SASA is the only channel that never falls
back, so complexes lacking GO annotations or STRING coverage showed exactly the
reported symptom.

**Fixed by** `src/models/attribution.py` (occlusion attribution — the honest
analogue of `w·x` for a model with no linear decomposition) and
`src/features/channel_health.py`. Payloads now carry an `available` block, and
both frontends render a hatched "no data" bar instead of a fake 0.5.

---

## Fault 2 — the data bug (real, partly fixed)

The user's hypothesis was that the model had become SASA-greedy. The data says
something stronger and different.

### Channel health, 891 usable complexes

| channel | mean | std | vs SASA | dead | verdict |
|---|---|---|---|---|---|
| `sasa` | 0.226 | 0.292 | 1.000 | 0% | ok |
| `contact` | 0.194 | 0.282 | 0.966 | 0% | ok |
| `ppi` **after fix** | 0.318 | 0.187 | **0.641** | 46% | ok where present |
| `ppi` *before fix* | 0.0003 | 0.0002 | **0.001** | 46% | numerically invisible |
| `esm_cos` | 0.894 | 0.061 | 0.210 | 0.1% | saturated but usable |
| `go_cos` | 0.000 | 0.000 | **0.000** | **100%** | dead |

**`ppi` — a unit bug.** `src/data_prep/string_api.py` did
`float(item["score"]) / 1000.0`. STRING's `/api/json/network` returns `score`
already normalised to [0, 1]; only the *request* parameter `required_score`
uses the 0-1000 integer scale. Verified against the live API — TP53/MDM2 returns
`score=0.999`. So every cached PPI value was written 1000× too small; the entire
cache clustered at 0.000999. Against SASA's 0.29 spread this channel was three
orders of magnitude too quiet to matter.

Repaired in place by `scripts/repair_ppi_cache.py` — no re-querying needed.
1941 cache files, 25,734 values, 2279 per-complex matrices. Range went from a
constant ~0.001 to [0.152, 0.999], and dynamic range from 0.001× SASA to
0.641×. The script is idempotent (bugged values land in [1e-4, 1e-3], genuine
ones are ≥ 0.15, and the ranges do not overlap).

**`go_cos` — dead, and not worth reviving as-is.** Every
`data/labeled_15k/*/go.npy` is an all-zero placeholder that was allocated but
never populated. Real GO matrices exist in `data/go_cache/` — but only for 382
complexes (85 overlapping the training corpus), and they are themselves
saturated: mean 0.985, std 0.005, 95.5% of matrices with std < 0.02. Wiring them
in would swap a zero channel for a near-constant one. Left dead and reported as
such; reviving GO needs a discriminative functional-similarity measure, not a
plumbing change.

**`contact` was a false alarm** — mine. The first audit read a `contact_matrix`
attribute that `ComplexData` does not have, so it scored 100% dead. It is
actually one of the healthiest channels (0.966× SASA). Corrected in
`audit_channels.py`.

---

## What the model actually uses

Occlusion attribution: each channel is replaced by its own off-diagonal mean
(removes variation, preserves scale — zeroing instead would conflate "no
information" with "shifted off the training distribution"). Teacher-forced, so
seed-independent. Freshly trained `all_features` model, 150 held-out complexes,
671 steps, baseline top-1 = 0.475.

| occluded | Δ top-1 | mean JSD |
|---|---|---|
| **`node:esm`** | **+0.146** | **0.1310** |
| `node:chiral` | −0.003 | 0.00415 |
| `ALL:random_edges` *(control)* | +0.003 | 0.00334 |
| `ALL:edges` *(control)* | +0.002 | 0.00256 |
| `node:plddt` | −0.005 | 0.00096 |
| `node:geom` | +0.008 | 0.00076 |
| `sasa` | +0.002 | 0.00071 |
| `contact` | +0.003 | 0.00041 |
| `chiral` | +0.006 | 0.00030 |
| `esm_cos` | 0.000 | 0.00000 |
| `ppi` | 0.000 | 0.00000 |
| `go_cos` | 0.000 | 0.00000 |

The two controls carry the finding. `ALL:random_edges` replaces the **entire**
edge tensor with uniform noise; `ALL:edges` neutralises every channel at once.
Both move the model less than 3% as much as occluding the ESM node block alone.
`ALL:random_edges` being non-zero also proves the occlusion machinery is not a
no-op — it perturbs, the model just barely responds.

**The model is not SASA-greedy. It is edge-blind.** Randomising every pairwise
interface feature — SASA included — changes top-1 by 0.003. Essentially all of
the decision comes from the per-chain ESM node embeddings.

### Re-run after the PPI repair

Same seed, same epochs, same split; only `ppi.npy` differs.

| occluded | Δ top-1 | mean JSD |
|---|---|---|
| **`node:esm`** | **+0.152** | **0.1023** |
| `ALL:random_edges` *(control)* | +0.013 | 0.00213 |
| `node:chiral` | +0.012 | 0.00154 |
| `ALL:edges` *(control)* | +0.010 | 0.00143 |
| `sasa` | +0.008 | 0.00050 |
| `contact` | +0.006 | 0.00026 |
| `ppi` | −0.002 | 0.00000 |
| `esm_cos` | +0.002 | 0.00000 |
| `go_cos` | 0.000 | 0.00000 |

Baseline top-1 rose 0.475 → 0.505. **Do not read that as the repair paying off** —
it is one seed, and it sits alongside the fact that `ppi` occlusion still moves
the model by JSD 3e-6, i.e. the trained model still does not use the channel.
The likelier explanation is that changing the input perturbed the optimisation
into a different local optimum. Needs multiple seeds before it means anything.

The structural conclusion is unchanged and is now the load-bearing one: with
`ppi` restored to 0.64× SASA's dynamic range, the edge pathway is *still* dead —
`ALL:random_edges` moves the model 48× less than the ESM node block. **Fixing
the data did not make the model use the data.** The bottleneck is architectural,
not a feature outage: the GATv2 edge pathway is not learning to condition on
edge features at all. That is the thing to investigate next, and it is upstream
of every "which interface feature matters" question.

This is consistent with the earlier scaling ablation, where `esm_only` (0.602)
nearly matched `v4_baseline` (0.610) while `interface_only` reached 0.500: the
interface graph was contributing far less than its channel count suggested.

### Consequence for the roadmap

The dead/crushed channels mean the original ablation's `interface_only` and
`no_esm` rows were run against a partly broken feature set — `ppi` at 1/1000
scale and `go_cos` identically zero. Those rows understate what interface
features can do. **They should be re-run now that PPI carries signal**, before
any conclusion about "structure vs sequence" is treated as settled. The headline
finding (ESM dominates at small n, does not extrapolate) is unaffected, since it
rests on the ESM node block, which was never broken.

---

## Files

| file | role |
|---|---|
| `audit_channels.py` | data health + occlusion attribution, with controls |
| `src/features/channel_health.py` | dead / saturated / ok classification, reusable |
| `src/models/attribution.py` | occlusion attribution for the GNN |
| `scripts/repair_ppi_cache.py` | in-place, idempotent PPI unit-bug repair |
| `src/data_prep/string_api.py` | `normalise_string_score` — tolerates both scales |
| `run_ablation.py` | `report_channel_health` gate at data load |
| `server/server.py`, `hf_deploy/prism-space/app.py` | real attribution + `available` |
| `static/index.html`, `hf_deploy/prism-space/static/index.html` | "no data" rendering |

## Known-not-fixed

- `go_cos` is still dead. Reviving it needs a discriminative GO similarity, not
  plumbing — the cached matrices are saturated at 0.985 ± 0.005.
- `hf_deploy/prism-space/` does not import standalone: it does
  `from src.models.scorer_gnn import ...` but the bundle ships no `src/models/`.
  Pre-existing, unrelated to this change, but it means that bundle only runs
  from the repo root.
- `server/server.py` expects `server/static/`, which does not exist, and loads
  `server/best_gnn.pt`, which does not exist — so it silently falls back to the
  weighted heuristic. Also pre-existing.
- `server/_build_gnn_features` derives its SASA matrix from **sequence length**
  (`min(len_i, len_j) / total_len`), not buried surface area, and builds 646/4-dim
  features against a 655/5-dim training layout.
