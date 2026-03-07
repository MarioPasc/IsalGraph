# Dataset Strategy for IsalGraph Formal Evaluation

**Authors**: Mario Pascual González, Ezequiel López Rubio
**Date**: March 2026
**Status**: Finalized — ready for implementation

---

## 1. Overview

The experimental evaluation of IsalGraph requires correlating Levenshtein distance on canonical graph strings against graph edit distance (GED), the gold-standard metric for graph structural similarity. This document defines the 5 real-world datasets selected for the evaluation, documents the GED computation strategy for each, specifies the common output format, and records the rejection rationale for excluded candidates.

All selected datasets satisfy three non-negotiable constraints imposed by IsalGraph's design:

1. **Topology-only**: IsalGraph encodes adjacency structure exclusively. Datasets must either have no node/edge attributes, or their precomputed GED must have been calculated on topology alone (uniform substitution cost = 0).
2. **Canonical string feasibility**: IsalGraph's canonical string computation uses exhaustive backtracking over neighbor choices at each V/v step, with worst-case exponential complexity. In practice, this is feasible for sparse graphs up to ~12 nodes and dense graphs up to ~8 nodes. All selected datasets must fall within these bounds.
3. **Exact GED ground truth**: Approximate GED introduces unknown error that would confound the correlation analysis. All selected datasets must provide or admit exact GED computation.

---

## 2. Selected Datasets

### 2.1 IAM Letter LOW

| Property | Value |
|----------|-------|
| **Source** | Zenodo (open access): https://zenodo.org/records/13763793 |
| **Domain** | Handwritten letter recognition |
| **N graphs** | ~750 |
| **Avg nodes** | ~4.7 |
| **Max nodes** | 9 |
| **Node features** | None (x,y coordinates stripped — topology only) |
| **Edge features** | None |
| **GED source** | Computed by us via NetworkX exact A* |
| **GED cost function** | Uniform: node ins/del = 1, edge ins/del = 1, node sub = 0 |
| **Reference** | Riesen, K. & Bunke, H. (2008). "IAM Graph Database Repository for Graph Based Pattern Recognition and Machine Learning." SSPR/SPR 2008, LNCS 5342, pp. 287–297. |

**Scientific role**: Clean baseline. LOW distortion applies minimal morphological noise to the original handwritten letter drawings before graph extraction. Within-class graphs (e.g., two drawings of the letter "A") are nearly isomorphic. This should yield high Levenshtein–GED correlation (Spearman ρ close to 1) and serves as a sanity check: if IsalGraph fails on this easy case, the method is fundamentally broken.

### 2.2 IAM Letter MED

Identical to LOW except `distortion = "MED"`. Moderate morphological operations introduce controlled structural noise. Within-class GED increases relative to LOW. Tests whether IsalGraph's correlation degrades gracefully under moderate perturbation.

### 2.3 IAM Letter HIGH

Identical to LOW except `distortion = "HIGH"`. Maximum structural variation. The hardest within-domain test case. If Levenshtein–GED correlation remains significant here, it provides strong evidence that IsalGraph captures meaningful structural similarity even under substantial noise.

**Cross-distortion analysis**: The three IAM Letter levels enable a monotonicity test via the Jonckheere-Terpstra ordered-alternative test. Specifically, for within-class pairs:

- H₀: Spearman ρ(Levenshtein, GED) is constant across LOW → MED → HIGH.
- H₁: Spearman ρ decreases monotonically as distortion increases.

Rejection of H₀ with a monotone decrease would precisely characterize how IsalGraph's proxy quality degrades under structural noise — a finding no prior work has established for any graph-to-string encoding.

### 2.4 LINUX

| Property | Value |
|----------|-------|
| **Source** | PyTorch Geometric `GEDDataset(name="LINUX")`, auto-downloaded from Google Drive |
| **Domain** | Program dependency graphs (Linux kernel) |
| **N graphs** | 1,000 (800 train + 200 test) |
| **Avg nodes** | ~7.6 |
| **Max nodes** | ~21 (rare outliers; vast majority ≤ 15) |
| **Node features** | 0 (pure topology) |
| **Edge features** | 0 |
| **GED source** | Precomputed exact A* by Bai et al. (2019) |
| **GED matrix** | Available for all train-train and train-test pairs; test-test pairs are `inf` |
| **Reference** | Bai, Y., Ding, H., Bian, S., Chen, T., Sun, Y., & Wang, W. (2019). "SimGNN: A Neural Network Approach to Fast Graph Similarity Computation." WSDM 2019. arXiv:1808.05689. |

**Scientific role**: Cross-domain validation. Software dependency graphs have fundamentally different structural properties from handwritten letters (higher density, different degree distribution). If IsalGraph correlates with GED in both domains, the claim generalizes beyond a single application area.

**Caveat**: Jain et al. (2024) found that only ~8.9% of LINUX graphs are structurally unique. Many graphs are isomorphic, meaning the GED matrix is heavily concentrated near zero. This is actually a useful test for canonical string correctness (isomorphic graphs must produce identical strings, yielding GED = 0 and Levenshtein = 0 simultaneously), but it reduces the effective sample size for non-trivial correlation analysis. We will compute and report the number of distinct canonical strings.

**Node count note**: Most LINUX graphs are well within canonical string range (≤ 12 nodes). Rare outliers with ~15–21 nodes may require greedy-only strings. The agent must compute canonical strings where feasible and fall back to greedy (min over all starting nodes) for larger graphs, clearly reporting which strategy was used.

### 2.5 ALKANE

| Property | Value |
|----------|-------|
| **Source** | PyTorch Geometric `GEDDataset(name="ALKANE")`, auto-downloaded from Google Drive |
| **Domain** | Molecular topology (alkane carbon skeletons) |
| **N graphs** | 150 |
| **Avg nodes** | ~8.9 |
| **Max nodes** | ~12 |
| **Node features** | 0 (pure topology) |
| **Edge features** | 0 |
| **GED source** | Precomputed exact A* by Bai et al. (2019) |
| **GED matrix** | Available for train-train pairs only. **Train-test GED is missing.** |
| **Reference** | Same as LINUX (Bai et al., 2019). |

**Scientific role**: Theoretical anchor. ALKANE graphs are acyclic (trees). This is significant because the only existing theoretical result connecting string edit distance to graph/tree edit distance applies specifically to trees: Akutsu, Fukagawa & Takasu (2010), "Approximating Tree Edit Distance through String Edit Distance," *Algorithmica* 57:325–348, proved an O(n^{3/4})-approximation bound for ordered rooted trees of bounded degree. IsalGraph should exhibit its strongest correlation on trees, and ALKANE lets us test this prediction directly.

**Caveat**: Only 150 graphs and train-train GED only. Statistical power is limited. We treat this as a focused validation of the tree-specific hypothesis rather than a full-power correlation study.

---

## 3. GED Computation Strategy

### 3.1 IAM Letter (LOW, MED, HIGH) — Compute via NetworkX

We compute exact GED ourselves using `networkx.graph_edit_distance()`, which implements A* search with the default uniform cost function for unlabeled graphs:

- Node insertion / deletion cost: 1
- Edge insertion / deletion cost: 1
- Node substitution cost: 0 (all nodes are identical after stripping coordinates)

**Why this is exact**: NetworkX A* explores the full search space of node mappings and returns the provably minimum-cost edit path. For unlabeled graphs, this is mathematically equivalent to what GEDLIB's A* implementation computes. There is no approximation involved.

**Why this is feasible**: Exact GED is NP-hard in general (Zeng et al., 2009), but practical for small graphs. Blumenthal & Gamper (2020) showed that A* times out at ~16 nodes. IAM Letter graphs have at most 9 nodes — far below this threshold. Empirical cost:

| Pair type | Est. time per pair | Pairs per level | Est. total time |
|-----------|--------------------|-----------------|-----------------|
| Avg (4–5 nodes) | 1–5 ms | ~281,000 | ~5–25 min |
| Worst case (9 vs 9) | 50–100 ms | ~5% of pairs | ~15 min overhead |
| **Per distortion level** | — | ~281,000 | **~25 min** |
| **All three levels** | — | ~843,000 | **~75 min** |

This is a one-time computation whose results are persisted. GEDLIB offers no advantage here.

### 3.2 LINUX — Extract from PyTorch Geometric

The precomputed exact GED matrix is accessible via:

```python
from torch_geometric.datasets import GEDDataset
dataset = GEDDataset(root="data/linux_ged", name="LINUX", train=True)
ged_matrix = dataset.ged       # torch.Tensor [N, N]
norm_ged = dataset.norm_ged    # GED / (0.5 * (|V1| + |V2|))
```

The matrix contains exact A* GED for all train-train (800×800) and train-test (800×200) pairs. Test-test pairs (200×200) are `inf` (not computed).

No computation required — we only need to extract and convert to our common format.

### 3.3 ALKANE — Extract from PyTorch Geometric

Same extraction method as LINUX. Note: only train-train GED is available. Train-test pairs are also `inf`. Total usable pairs: C(N_train, 2).

### 3.4 Common Output Format

All five datasets are persisted in a unified format under `data/ged_matrices/`:

```
data/ged_matrices/
    iam_letter_low.npz
    iam_letter_med.npz
    iam_letter_high.npz
    linux.npz
    alkane.npz
    README.md          # Documents format, provenance, cost functions
```

Each `.npz` file contains:

| Key | Type | Description |
|-----|------|-------------|
| `ged_matrix` | `float64 [N, N]` | Pairwise exact GED. `inf` for unavailable pairs. Symmetric. Diagonal = 0. |
| `node_counts` | `int32 [N]` | Number of nodes per graph. |
| `edge_counts` | `int32 [N]` | Number of edges per graph. |
| `graph_ids` | `str [N]` | Unique identifier per graph (e.g., `"AP1_0001.gxl"` for IAM, `"train_042"` for LINUX). |
| `labels` | `str [N]` | Class label (e.g., `"A"` for IAM Letter, `""` for LINUX/ALKANE). |
| `metadata` | `dict` | `{"dataset", "ged_method", "ged_cost_function", "source", "n_graphs", "n_valid_pairs"}` |

This format enables all downstream benchmarks to load GED matrices without knowing the original dataset format or recomputing anything.

---

## 4. Rejected Datasets

### 4.1 AIDS700nef — Rejected (label-contaminated GED)

| Property | Value |
|----------|-------|
| **N graphs** | 700 |
| **Avg nodes** | ~8.9 |
| **Node features** | **29 atom types** (O, S, C, N, Cl, Br, B, Si, Hg, I, Bi, P, F, ...) |
| **GED computation** | Exact A*, **with node relabeling costs** |

**Rejection reason**: The precomputed GED in AIDS700nef incorporates **node substitution costs** driven by atom-type differences. From the SimGNN paper: edit operations include "a node relabeling." From the Noah-GED repository: "Note that AIDS dataset is with node labels, while the other two datasets are not."

IsalGraph encodes topology only — it is blind to node labels. Correlating a topology-only distance (Levenshtein on canonical strings) against a topology-plus-labels distance (the precomputed GED) creates a systematic confound:

- Two graphs with identical topology but different atom types → GED > 0, Levenshtein = 0.
- Two graphs with different topology but matching atoms → GED is deflated by the label agreement.

The measured correlation would be **artificially suppressed** by the label component that IsalGraph cannot capture. A reviewer could correctly argue that any weak correlation is an artifact of the measurement mismatch rather than a limitation of IsalGraph's encoding. Recomputing GED ourselves with uniform node costs would solve this, but then the 29 node features would be discarded — and at avg ~8.9 nodes, we already have IAM Letter covering a similar size range with cleaner provenance.

### 4.2 IMDBMulti — Rejected (approximate GED + infeasible canonical strings)

| Property | Value |
|----------|-------|
| **N graphs** | 1,500 |
| **Avg nodes** | ~13.0 |
| **Avg edges** | ~131.9 (density ≈ 0.85) |
| **GED computation** | **Approximate**: minimum of HED, Hungarian, and VJ heuristics |

**Rejection reason 1 — Approximate GED**: Unlike LINUX, AIDS, and ALKANE where exact A* was used, the IMDBMulti ground truth was computed as the minimum of three heuristic upper bounds (HED, Hungarian, VJ). These are not guaranteed to equal the true GED. Correlating Levenshtein distance against an approximation with unknown error would weaken the scientific claim. We could not distinguish between "IsalGraph is a poor proxy" and "the ground truth itself is inaccurate."

**Rejection reason 2 — Canonical string infeasibility**: With ~13 avg nodes and density ~0.85, these graphs are both too large and too dense for IsalGraph's exhaustive canonical search. Our empirically validated limits are N ≤ 8 for dense graphs (`CANONICAL_LIMIT_DEFAULT = 8` in the codebase) and N ≤ 12 for sparse graphs (`CANONICAL_LIMIT_SPARSE = 12`). At 13 nodes and 85% density, canonical strings would require intractable computation time. Falling back to greedy strings (which depend on the starting node and are not isomorphism-invariant) would fundamentally undermine the core claim that IsalGraph provides a canonical representation.

Either rejection reason alone would suffice. Together, they make IMDBMulti unsuitable for our evaluation.

### 4.3 IAM GREC — Rejected (edge labels essential to the dataset's purpose)

| Property | Value |
|----------|-------|
| **Avg nodes** | ~11.5 |
| **Edge features** | Line types (labeled) |

The GREC dataset represents electronic circuit symbols where edge labels encode line types (line, arc, etc.). Stripping these labels to obtain topology-only graphs would lose the dataset's structural identity — two GREC graphs that differ only in edge labels (semantically very different circuits) would appear identical to IsalGraph. The resulting correlation analysis would be meaningless.

### 4.4 IAM Mutagenicity / Protein — Rejected (supervisor instruction)

These molecular/protein datasets overlap with the published JCIM paper. Per the supervisor's explicit instruction, molecular datasets are excluded to avoid redundancy with the prior publication.

### 4.5 TUDatasets social graphs (COLLAB, REDDIT-B) — Rejected (size)

| Dataset | Avg nodes |
|---------|-----------|
| COLLAB | ~74.5 |
| REDDIT-B | ~430 |

Far beyond the feasible range for canonical strings. Even greedy strings at this scale would require benchmarking that falls outside the scope of the correlation analysis.

---

## 5. Dataset Summary Table (for the Paper)

| Dataset | Domain | N | Avg |V| | Max |V| | Topology-only | GED method | GED source |
|---------|--------|---|---------|---------|---------------|------------|------------|
| IAM Letter LOW | Handwriting | ~750 | 4.7 | 9 | Yes | Exact A* (NX) | Computed |
| IAM Letter MED | Handwriting | ~750 | 4.7 | 9 | Yes | Exact A* (NX) | Computed |
| IAM Letter HIGH | Handwriting | ~750 | 4.7 | 9 | Yes | Exact A* (NX) | Computed |
| LINUX | Software deps | 1,000 | 7.6 | ~21 | Yes | Exact A* | Precomputed (Bai et al., 2019) |
| ALKANE | Molecular trees | 150 | 8.9 | ~12 | Yes | Exact A* | Precomputed (Bai et al., 2019) |

**Total**: 5 real-world datasets, 3 domains, ~3,400 graphs, all with exact GED ground truth.

This exceeds the minimum of 3–5 datasets recommended by Demšar (2006) for Q1 journal standards and provides both within-domain gradients (IAM Letter LOW/MED/HIGH) and cross-domain validation (handwriting vs. software vs. molecular).

---

## 6. Implementation Plan

### Phase 1: GED Matrix Computation and Extraction

This is the immediate next step before any benchmark agent is launched.

**Task 1**: Compute IAM Letter GED matrices (LOW, MED, HIGH).
- Load graphs via `iam_letter.py` (topology only, coordinates stripped).
- Compute all-pairs exact GED via `networkx.graph_edit_distance()`.
- Persist as `iam_letter_{low,med,high}.npz` in common format.
- Estimated time: ~75 minutes total (single-threaded). Parallelizable.

**Task 2**: Extract LINUX GED matrix from PyTorch Geometric.
- Load via `GEDDataset(name="LINUX")`.
- Convert PyG Data objects to NetworkX graphs (topology only).
- Extract `dataset.ged` and `dataset.norm_ged` tensors.
- Persist as `linux.npz` in common format.
- Estimated time: < 5 minutes (download + conversion).

**Task 3**: Extract ALKANE GED matrix from PyTorch Geometric.
- Same procedure as LINUX.
- Note: only train-train GED is available.
- Persist as `alkane.npz` in common format.
- Estimated time: < 5 minutes.

**Task 4**: Validate all matrices.
- Symmetry: `ged_matrix[i,j] == ged_matrix[j,i]` for all valid pairs.
- Diagonal: `ged_matrix[i,i] == 0` for all i.
- Non-negativity: `ged_matrix[i,j] >= 0` for all valid pairs.
- Triangle inequality spot checks: `ged_matrix[i,k] <= ged_matrix[i,j] + ged_matrix[j,k]` on random triples.
- Concordance: for LINUX/ALKANE, verify our extracted matrix matches PyG's `dataset.ged` tensor exactly.

### Phase 2: Benchmark Execution

Only after Phase 1 is validated. All 5 benchmark agents read from `data/ged_matrices/*.npz` as their ground truth. No benchmark agent computes GED.
