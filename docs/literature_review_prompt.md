# PRISM Literature Review Prompt

## Context for Claude

You are being asked to conduct a rigorous, structured literature review comparing **PRISM** (Protein Ranking by Interface and Sequence Modeling) to existing computational methods that predict or infer protein complex assembly order. The review should identify PRISM's specific novelty, its gaps relative to the field, and concrete directions for improvement.

---

## PRISM: Model Description

**PRISM** is a graph attention network (GAT) that predicts the sequential order in which protein subunits assemble into a multi-subunit complex — the "assembly order" — from structural and sequence-derived features alone, without knowledge of assembly kinetics or experimental measurements.

### Architecture

- **Model class:** Graph Attention Network (GAT), 3 layers, 4 attention heads, hidden dim = 128
- **Parameters:** ~167,000
- **Input per complex:**
  - Node features (646-dim per subunit): ESM-2 150M embeddings (640d) + AlphaFold pLDDT statistics (5d: mean, frac>90, frac<50, q10, q90) + SASA diagonal (1d)
  - Edge features (4-dim per pair): STRING PPI score, ESM-2 cosine similarity, GO-MF cosine similarity (ProtFunc logit space), normalised buried surface area (BSA/SASA)
  - Adjacency: fully connected (NxN dense graph)
- **Output:** scalar compatibility score per candidate subunit at each assembly step
- **Inference:** greedy sequential assembly — at each step, score all remaining candidates and pick highest
- **Training loss:** cross-entropy ranking loss at each assembly step (correct next subunit must outscore all others)
- **Ring homomer augmentation:** 2N cyclic rotations of ground-truth order used as training targets

### Feature computation pipeline

1. **ESM-2 150M** (`esm2_t30_150M_UR50D`): mean-pooled residue representations from layer 30, truncated at 1022 residues
2. **BSA matrix** (`freesasa`): pairwise buried surface area computed from atomic PDB coordinates — O(N²) substructure evaluations; SASA-greedy heuristic used to derive pseudo-GT for complexes without experimental assembly order
3. **pLDDT**: per-chain AlphaFold v4 model statistics via EBI API, 5 summary statistics as node features
4. **STRING PPI**: species-specific interaction scores (currently zeros for most Marsh 2013 complexes due to mixed-species problem)
5. **GO-MF**: ProtFunc MLP predictions (8M ESM-2 + 8-block ResidualMLP → 8124 GO-MF logits); pairwise cosine similarity in logit space used as GO coherence edge feature

### Evaluation metrics

- **Kendall's τ**: rank correlation between predicted and GT assembly order. Circular τ (max over 2N rotations) used for ring homomers.
- **gt_prob**: softmax probability assigned to the GT next subunit, averaged over all steps and complexes — the primary training signal
- **Top-1 accuracy**: fraction of steps where the highest-scored candidate equals GT next
- **Early-step τ**: Kendall's τ computed on only the first ⌈N/2⌉ steps (most mechanistically informative)
- **Temperature calibration**: post-training scalar temperature T* = 0.332 found by ternary search minimising NLL on eval set (NLL improved 0.2728 → 0.2326)

---

## PRISM: Current Performance

### Training set composition

| Source | Complexes | GT type | Role |
|--------|-----------|---------|------|
| Marsh 2013 | 10 | Experimental literature | Held-out eval |
| Experimental GT v2 | 13 | Experimental literature | Train + eval |
| RCSB expanded | 300 | SASA-greedy (proxy) | Training only |
| **Total training** | **323** | mixed | — |

### Benchmark results (model trained only on 10 Marsh 2013 complexes at eval time)

| Set | n | τ | Top-1 | gt_prob | early-τ |
|-----|---|---|-------|---------|---------|
| Marsh 2013 (in-distribution) | 10 | 1.000 | 0.867 | 0.856 | 1.000 |
| Exp GT v2 (held-out) | 13 | 0.974 | 0.962 | 0.966 | 0.974 |
| Combined | 23 | 0.986 | 0.920 | 0.918 | 0.986 |

**Stratified by subunit count (combined):**

| N | Complexes | Mean τ |
|---|-----------|--------|
| 2 | 14 | 1.000 |
| 3 | 2  | 1.000 |
| 4 | 3  | 0.889 |
| 5 | 1  | 1.000 |
| 6 | 1  | 1.000 |
| 7 | 2  | 1.000 |

**Notable failures:**
- 2OCJ (p53 DNA-binding domain tetramer, N=4, dimer-of-dimers): τ=0.667, top1=0.500 — model cannot capture hierarchical assembly intermediates
- 1AON, 1GRU (GroEL 7-mers, homomers): top1=0.333 — model cannot distinguish identical subunits within a ring

**Significance testing (Marsh 2013 vs Exp GT v2):**
- τ difference: MWU p=0.491 (ns), Cohen's d=0.39 (small)
- Mean confidence difference: MWU p=0.009 (**) — v2 easier due to 2-chain bias
- Conclusion: model generalises to held-out complexes; performance gap attributable to complexity difference, not structural family memorisation

### Known limitations

1. **Training scale:** 10-323 complexes is small by ML standards; risk of memorisation on Marsh 2013
2. **GT quality:** SASA-greedy pseudo-GT for 300 RCSB complexes is a proxy; may not reflect true kinetic assembly
3. **Homomer discrimination:** For N-chain homomers, all subunits are identical → model cannot distinguish sequential position from BSA/ESM/GO alone
4. **Hierarchical assembly:** Dimer-of-dimers (p53), ring closure, cooperative assembly not modelled — assumes strictly sequential addition
5. **Scalability:** O(N²) BSA computation + O(N) GNN forward passes per step; impractical for N > 20
6. **No dynamics:** Static crystal structure features only; assembly pathway may differ from equilibrium structure
7. **STRING PPI gap:** Mixed-species taxid problem means PPI edge is zero for most Marsh 2013 complexes
8. **Cryo-EM structures:** Currently restricted to X-ray structures for BSA computation; no AlphaFold-multimer integration

---

## Literature Review Task

Please conduct a structured literature review addressing the following questions. For each method you identify, provide: (a) citation, (b) approach, (c) inputs required, (d) reported performance metrics, (e) direct comparison to PRISM on each dimension.

### Primary question

**What computational methods exist that predict or infer protein complex assembly order, and how does PRISM compare to the state of the art?**

### Specific areas to cover

#### 1. Direct competitors: assembly order prediction methods

Search for methods that explicitly predict the sequential order of subunit addition during protein complex assembly. Include:
- Classic SASA/BSA greedy methods (the Marsh 2013 baseline PRISM is built on)
- Machine learning methods (if any) trained to predict assembly order
- Methods using evolutionary conservation, co-evolution, or phylogenetic analysis to infer order
- Methods using native mass spectrometry (native MS) subcomplex mapping
- Methods using pulse-chase experiments + proteomics
- Any benchmarks or competitions targeting this problem

Key papers to start from (expand from these and their citing works):
- Marsh JA et al. (2013) "Structure/function of the quaternary protein complexes" Nature 498:366–369
- Levy ED et al. (2008) "Assembly reflects evolution of protein complexes" Nature 453:1262–1265
- Ahnert SE et al. (2015) "Principles of assembly" Science 350:aaa2245
- Frappier V & Bhargava S (2015) "A coarse-grained elastic network model of protein assembly"
- Schmid AW et al. (2018) "The use of native MS to probe amyloid formation"

#### 2. Adjacent methods: complex structure prediction

Methods that predict complex structure (not assembly order) but whose output could constrain assembly:
- AlphaFold-Multimer (Evans et al. 2021)
- RoseTTAFold2 multimer
- ClusPro, HADDOCK (protein-protein docking)
- How do their predicted interface areas compare to BSA from crystal structures?

#### 3. Graph neural networks on protein complexes

Any GNN methods applied to protein complexes, regardless of task:
- Interface prediction GNNs
- Binding affinity prediction
- Complex stability prediction
- Do any use GAT with BSA/SASA edge features?

#### 4. Large complex assembly (10-15 subunits)

For large complexes (oligomers with 10-15 subunits):
- What experimental methods determine assembly order at this scale?
- Cryo-EM assembly pathway inference papers
- Native MS subcomplex mapping for large assemblies (proteasome, ribosome, ATP synthase)
- Are there computational methods that scale to this regime?
- What are typical Kendall's τ values (or equivalent metrics) reported for large-complex ordering?

#### 5. Evaluation standards

- What datasets are used to benchmark assembly order methods?
- What metrics do existing methods report? (Are τ and gt_prob standard, or does the field use other metrics?)
- What is the largest complex for which assembly order is experimentally known with step-by-step resolution?

### Output format requested

Structure your response as follows:

**Section 1: Landscape summary** (200 words)
What is the state of the field? Is assembly order prediction a well-studied problem or nascent? What are the dominant approaches?

**Section 2: Method-by-method comparison table**
For each method found, fill in:
| Method | Year | Input | Approach | Scale (max N) | Key metric | Value | PRISM comparison |

**Section 3: PRISM novelty assessment**
What is genuinely new in PRISM vs prior work? What has been done before? Be precise — avoid vague claims.

**Section 4: Critical gaps**
What does the literature say about assembly order prediction that PRISM does not address? What are the most impactful missing pieces?

**Section 5: Recommended next experiments**
Based on the literature, what experiments or datasets should be used to validate PRISM more rigorously? List 3-5 concrete, feasible experiments with specific datasets or benchmarks.

**Section 6: Suggested citations for a PRISM paper**
List 15-20 key references a PRISM methods paper should cite, with one-line justifications.

---

## Additional context for the reviewer

- PRISM's GT data source (Marsh 2013) is 10 complexes from the classic paper that introduced the SASA-greedy heuristic — the same paper PRISM benchmarks against
- The SASA-greedy baseline itself achieves τ ≈ 0.73 on Marsh 2013 (from the original paper)
- PRISM achieves τ = 1.00 on Marsh 2013 — but this may partly reflect training on those exact complexes
- PRISM's held-out τ = 0.974 on 13 experimental GT v2 complexes (mostly 2-chain)
- The model has not been tested on complexes larger than 7 subunits with known experimental GT
- No published benchmark exists for assembly order prediction at N ≥ 10 to our knowledge — but please verify this
