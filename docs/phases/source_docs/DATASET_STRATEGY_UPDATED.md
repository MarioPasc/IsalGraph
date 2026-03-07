# Dataset Strategy for IsalGraph Formal Evaluation

**Authors**: Mario Pascual González, Ezequiel López Rubio
**Date**: March 2026
**Status**: Finalized — ready for implementation
**Revision**: v2 — March 2026 (data source migration; see §7 Changelog)

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
| **Source** | GraphEdX `no_attr_data/linux/` — Jain et al. (NeurIPS 2024) |
| **Download** | https://rebrand.ly/graph-edit-distance → `data/no_attr_data/linux/` |
| **Domain** | Program dependency graphs (Linux kernel) |
| **N graphs** | ~800–1,000 (exact count depends on isomorphism deduplication; see below) |
| **Avg nodes** | ~7.6 |
| **Max nodes** | ~21 (rare outliers; vast majority ≤ 15) |
| **Node features** | 0 (pure topology) |
| **Edge features** | 0 |
| **GED source** | Precomputed exact A* with uniform costs (Cn = (0,0,0), Ce = (1,2,0) for the `no_attr_data` setting per Jain et al., 2024) |
| **GED matrix** | Available as PyTorch tensors in `train_result.pt`, `test_result.pt`, `val_result.pt` |
| **Isomorphism deduplication** | Yes — GraphEdX removes isomorphic duplicates (only 8.9% of original LINUX graphs were structurally unique per Jain et al., 2024) |
| **Format** | PyTorch `.pt` files: `{split}_graphs.pt` (graph structures), `{split}_result.pt` (GED ground truth), `{split}.pt` (PyG Data list), `{split}_features.pt` (node features, empty for topology-only) |
| **Reference (data)** | Bai, Y., Ding, H., Bian, S., Chen, T., Sun, Y., & Wang, W. (2019). "SimGNN: A Neural Network Approach to Fast Graph Similarity Computation." WSDM 2019. arXiv:1808.05689. |
| **Reference (curation)** | Jain, E., Roy, I., Meher, S., Chakrabarti, S., & De, A. (2024). "Graph Edit Distance with General Costs Using Neural Set Divergence." NeurIPS 2024. arXiv:2409.17687. |

**Scientific role**: Cross-domain validation. Software dependency graphs have fundamentally different structural properties from handwritten letters (higher density, different degree distribution). If IsalGraph correlates with GED in both domains, the claim generalizes beyond a single application area.

**Provenance note**: The original LINUX dataset from Bai et al. (2019), distributed via PyTorch Geometric's `GEDDataset`, became inaccessible in early 2026 due to Google Drive link rot affecting the hosting files (IDs `1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI` for data, `14FDm3NSnrBvB7eNpLeGy5Bz6FjuCSF5v` for the GED pickle). We use the GraphEdX redistribution, which sources from the same original data but applies isomorphism deduplication. This deduplication is scientifically beneficial: Jain et al. (2024) demonstrated that only 8.9% of the original LINUX graphs were structurally unique, causing severe train-test leakage in the standard benchmark. The deduplicated version provides a more rigorous evaluation.

**Isomorphism deduplication and H₃**: Because GraphEdX removes isomorphic duplicates, many trivial GED = 0 pairs are eliminated. This reduces the inflated correlation that isomorphic pairs would produce (both GED and Levenshtein are zero for isomorphic graphs, which trivially satisfies the proxy relationship). The resulting correlation estimates are more conservative and scientifically honest.

**Tree subset analysis (H₃)**: LINUX program dependency graphs include a subset of acyclic graphs (trees). We identify these via `nx.is_tree()` after loading and compute Levenshtein–GED correlation separately on the tree subset versus the non-tree subset. This tests the prediction from Akutsu et al. (2010) that string edit distance should be a stronger structural proxy for trees than for general graphs, without requiring a dedicated tree-only dataset.

**Node count note**: Most LINUX graphs are well within canonical string range (≤ 12 nodes). Rare outliers with ~15–21 nodes are excluded by the node-count filter (N_max = 12). The exact filtering impact is reported in `filtering_report.json`.

### 2.5 AIDS (Uniform-Cost GED)

| Property | Value |
|----------|-------|
| **Source** | GraphEdX `no_attr_data/aids/` — Jain et al. (NeurIPS 2024) |
| **Download** | https://rebrand.ly/graph-edit-distance → `data/no_attr_data/aids/` |
| **Domain** | Antiviral compound molecular topology (NCI/NIH DTP) |
| **N graphs** | ~700 (after isomorphism deduplication; original AIDS700nef had 700) |
| **Avg nodes** | ~8.9 |
| **Max nodes** | ~10 |
| **Node features** | 0 — **stripped by GraphEdX** (original AIDS700nef had 29 atom types) |
| **Edge features** | 0 |
| **GED source** | Precomputed exact A* with **uniform topology-only costs** (Cn = (0,0,0), Ce = (1,2,0) per Jain et al., 2024) |
| **GED matrix** | Available as PyTorch tensors in `train_result.pt`, `test_result.pt`, `val_result.pt` |
| **Isomorphism deduplication** | Yes |
| **Format** | Same as LINUX (PyTorch `.pt` files) |
| **Reference (data)** | Same as LINUX (Bai et al., 2019). |
| **Reference (curation)** | Same as LINUX (Jain et al., 2024). |

**Scientific role**: Molecular topology cross-domain validation. AIDS compound graphs have structural properties distinct from both handwritten letters (IAM) and software dependency graphs (LINUX): short cycles, branching carbon chains, and heteroatom-induced degree variation. This provides the third domain required for robust cross-domain generalization claims (Demšar, 2006).

**Critical distinction from AIDS700nef**: The original AIDS700nef dataset distributed via PyTorch Geometric's `GEDDataset` was rejected from this evaluation (see §4.1) because its precomputed GED incorporated **node substitution costs** driven by atom-type differences. The GraphEdX `no_attr_data/aids/` version solves this problem: it strips all node attributes and computes GED with purely topology-based uniform costs. This eliminates the systematic confound where IsalGraph (topology-only) would be evaluated against a label-sensitive distance metric. The two versions of AIDS contain the same underlying molecular graphs but with fundamentally different GED ground truth — the GraphEdX version is the correct one for our evaluation.

**Advantage over ALKANE**: The original dataset strategy included ALKANE (150 tree-structured alkane graphs, Bai et al. 2019) as the molecular topology representative. AIDS-uniform is superior on every dimension: (a) 700 vs 150 graphs (4.7× more statistical power), (b) full train/val/test GED available vs train-train only, (c) mixed molecular topology (cycles, branches, heterogeneous degree) vs trees only, and (d) available via a permanent, version-controlled source (GraphEdX) vs inaccessible Google Drive files.

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

### 3.2 LINUX — Extract from GraphEdX `.pt` Files

The precomputed exact GED matrix is distributed as PyTorch tensors in the GraphEdX `no_attr_data/linux/` folder:

```
no_attr_data/linux/
    train.pt              # List of PyG Data objects (train split)
    train_graphs.pt       # Graph structure tensors
    train_features.pt     # Node features (empty for topology-only)
    train_result.pt       # GED ground truth for train pairs
    val.pt / val_*.pt     # Validation split
    test.pt / test_*.pt   # Test split
```

**Extraction procedure**:

```python
import torch

# Load graph structures and GED ground truth
train_data = torch.load("no_attr_data/linux/train.pt")
train_results = torch.load("no_attr_data/linux/train_result.pt")

# Convert PyG Data objects to NetworkX (topology only)
# Extract edge_index from each Data object, build nx.Graph
# Extract GED matrix from train_results
```

**Important**: The exact tensor schema of `train_result.pt` must be inspected at load time. It may contain a dense GED matrix, a sparse list of (i, j, ged) triples, or a normalized distance tensor. The loader must handle all cases and document the observed format.

**Merging splits**: For maximum statistical power, we merge train + val + test graphs into a single pool and use all available GED pairs. If GED is only available within-split (train-train, val-val, etc.), we restrict to the largest available block. The filtering report documents exactly which pairs are available.

After loading, apply `filter_by_node_count(n_max=12)` and extract the GED submatrix for kept graphs.

### 3.3 AIDS (Uniform Cost) — Extract from GraphEdX `.pt` Files

Same extraction procedure as LINUX, from `no_attr_data/aids/`. The GED in this folder was computed with uniform topology-only costs (no node substitution penalty), making it directly compatible with IsalGraph's topology-only encoding.

After loading, apply `filter_by_node_count(n_max=12)`. Given that AIDS graphs have max ~10 nodes, we expect no filtering losses.

### 3.4 Common Output Format

All five datasets are persisted in a unified format under `data/eval/ged_matrices/`:

```
data/eval/ged_matrices/
    iam_letter_low.npz
    iam_letter_med.npz
    iam_letter_high.npz
    linux.npz
    aids.npz
    README.md          # Documents format, provenance, cost functions
```

Each `.npz` file contains:

| Key | Type | Description |
|-----|------|-------------|
| `ged_matrix` | `float64 [N, N]` | Pairwise exact GED. `inf` for unavailable pairs. Symmetric. Diagonal = 0. |
| `node_counts` | `int32 [N]` | Number of nodes per graph. |
| `edge_counts` | `int32 [N]` | Number of edges per graph. |
| `graph_ids` | `str [N]` | Unique identifier per graph (e.g., `"AP1_0001.gxl"` for IAM, `"linux_train_042"` for LINUX, `"aids_train_015"` for AIDS). |
| `labels` | `str [N]` | Class label (e.g., `"A"` for IAM Letter, `""` for LINUX/AIDS). |
| `metadata` | `dict` | `{"dataset", "ged_method", "ged_cost_function", "source", "n_graphs", "n_valid_pairs"}` |

This format enables all downstream benchmarks to load GED matrices without knowing the original dataset format or recomputing anything.

---

## 4. Rejected Datasets

### 4.1 AIDS700nef (Label-Contaminated GED) — Rejected

| Property | Value |
|----------|-------|
| **N graphs** | 700 |
| **Avg nodes** | ~8.9 |
| **Node features** | **29 atom types** (O, S, C, N, Cl, Br, B, Si, Hg, I, Bi, P, F, ...) |
| **GED computation** | Exact A*, **with node relabeling costs** |

**Rejection reason**: The precomputed GED in AIDS700nef (as distributed via PyTorch Geometric's `GEDDataset`) incorporates **node substitution costs** driven by atom-type differences. From the SimGNN paper: edit operations include "a node relabeling." From the Noah-GED repository: "Note that AIDS dataset is with node labels, while the other two datasets are not."

IsalGraph encodes topology only — it is blind to node labels. Correlating a topology-only distance (Levenshtein on canonical strings) against a topology-plus-labels distance (the precomputed GED) creates a systematic confound:

- Two graphs with identical topology but different atom types → GED > 0, Levenshtein = 0.
- Two graphs with different topology but matching atoms → GED is deflated by the label agreement.

The measured correlation would be **artificially suppressed** by the label component that IsalGraph cannot capture.

**Resolution**: The GraphEdX dataset suite (Jain et al., NeurIPS 2024) provides a **topology-only** version of AIDS under `no_attr_data/aids/`, where GED is computed with uniform costs and no node substitution penalty. This version is accepted as dataset §2.5 above. The rejection applies specifically to the label-contaminated PyG version.

### 4.2 IMDBMulti — Rejected (approximate GED + infeasible canonical strings)

| Property | Value |
|----------|-------|
| **N graphs** | 1,500 |
| **Avg nodes** | ~13.0 |
| **Avg edges** | ~131.9 (density ≈ 0.85) |
| **GED computation** | **Approximate**: minimum of HED, Hungarian, and VJ heuristics |

**Rejection reason 1 — Approximate GED**: Unlike LINUX, AIDS, and ALKANE where exact A* was used, the IMDBMulti ground truth was computed as the minimum of three heuristic upper bounds (HED, Hungarian, VJ). These are not guaranteed to equal the true GED. Correlating Levenshtein distance against an approximation with unknown error would weaken the scientific claim. We could not distinguish between "IsalGraph is a poor proxy" and "the ground truth itself is inaccurate."

**Rejection reason 2 — Canonical string infeasibility**: With ~13 avg nodes and density ~0.85, these graphs are both too large and too dense for IsalGraph's exhaustive canonical search. Our empirically validated limits are N ≤ 8 for dense graphs (`CANONICAL_LIMIT_DEFAULT = 8` in the codebase) and N ≤ 12 for sparse graphs (`CANONICAL_LIMIT_SPARSE = 12`). At 13 nodes and 85% density, canonical strings would require intractable computation time.

Either rejection reason alone would suffice. Together, they make IMDBMulti unsuitable for our evaluation.

### 4.3 IAM GREC — Rejected (edge labels essential to the dataset's purpose)

| Property | Value |
|----------|-------|
| **Avg nodes** | ~11.5 |
| **Edge features** | Line types (labeled) |

The GREC dataset represents electronic circuit symbols where edge labels encode line types (line, arc, etc.). Stripping these labels to obtain topology-only graphs would lose the dataset's structural identity — two GREC graphs that differ only in edge labels (semantically very different circuits) would appear identical to IsalGraph. The resulting correlation analysis would be meaningless.

### 4.4 IAM Mutagenicity / Protein — Rejected (supervisor instruction)

These molecular/protein datasets overlap with the published JCIM paper. Per the supervisor's explicit instruction, molecular datasets from the IAM database are excluded to avoid redundancy with the prior publication. Note: AIDS (§2.5) is sourced from GraphEdX, not IAM, and represents a different molecular domain (antiviral compounds vs. mutagens), so this exclusion does not apply to it.

### 4.5 TUDatasets social graphs (COLLAB, REDDIT-B) — Rejected (size)

| Dataset | Avg nodes |
|---------|-----------|
| COLLAB | ~74.5 |
| REDDIT-B | ~430 |

Far beyond the feasible range for canonical strings. Even greedy strings at this scale would require benchmarking that falls outside the scope of the correlation analysis.

### 4.6 ALKANE — Rejected (data unavailability + limited statistical power)

| Property | Value |
|----------|-------|
| **N graphs** | 150 |
| **Avg nodes** | ~8.9 |
| **Max nodes** | ~12 |
| **GED matrix** | Train-train pairs only. **Train-test GED missing.** |
| **Source** | PyTorch Geometric `GEDDataset(name="ALKANE")` — **inaccessible since early 2026** |

**Rejection reason 1 — Data inaccessibility**: The ALKANE dataset was distributed via Google Drive files hosted by Bai et al. (2019). As of March 2026, both the data archive (Google Drive ID `1-LmxaWW3KulLh00YqscVEflbqr0g4cXt`) and the GED pickle file (ID `15BpvMuHx77-yUGYgM27_sQett02HQNYu`) return HTTP 404. The original SimGNN Google Drive folder (`1lY3pqpnUAK0H9Tgjyh7tlMVYy0gYPthC`) has been reorganized and no longer contains the ALKANE archives. This is a known instance of Google Drive link rot affecting multiple PyTorch Geometric datasets (cf. PyG issues #8787, #8797). GraphEdX (Jain et al., 2024) does not redistribute ALKANE.

**Rejection reason 2 — Limited statistical power**: Even when accessible, ALKANE contained only 150 graphs with train-train GED only (~4,000 valid pairs). This yields marginal statistical power for permutation-based tests (Mantel test) and bootstrap confidence intervals. By contrast, AIDS-uniform (§2.5) provides ~700 graphs with full train/val/test GED.

**Rejection reason 3 — Superseded by tree subset analysis**: The original scientific role of ALKANE was to test the tree-specific hypothesis H₃, grounded in Akutsu et al. (2010). This hypothesis is now tested via the tree subset of LINUX (see §2.4), which avoids the need for a dedicated tree-only dataset and provides the test within a larger, more diverse graph population.

### 4.7 GraphEdX `ogbg-*` datasets — Rejected (graph size)

The GraphEdX suite includes `ogbg-code2`, `ogbg-molhiv`, and `ogbg-molpcba` under `no_attr_data/`. These are sourced from the Open Graph Benchmark (Hu et al., NeurIPS 2020) and contain graphs with tens to hundreds of nodes on average, far exceeding the feasible range for exhaustive canonical strings.

### 4.8 GraphEdX `no_attr_asymm_data/` and `label_symm_data/` — Rejected (cost function mismatch)

GraphEdX provides datasets under asymmetric costs (Cn = (1,3,0), Ce = (1,2,0)) and label-substitution costs (Cn = (1,1,1), Ce = (1,1,0)). These violate our topology-only, uniform-cost requirement. Only `no_attr_data/` (Cn = (0,0,0), Ce = (1,2,0)) is compatible with IsalGraph's design.

---

## 5. Dataset Summary Table (for the Paper)

| Dataset | Domain | N | Avg |V| | Max |V| | Topology-only | GED method | GED source |
|---------|--------|---|---------|---------|---------------|------------|------------|
| IAM Letter LOW | Handwriting | ~750 | 4.7 | 9 | Yes | Exact A* (NX) | Computed |
| IAM Letter MED | Handwriting | ~750 | 4.7 | 9 | Yes | Exact A* (NX) | Computed |
| IAM Letter HIGH | Handwriting | ~750 | 4.7 | 9 | Yes | Exact A* (NX) | Computed |
| LINUX | Software deps | ~800–1,000 | 7.6 | ~21 | Yes | Exact A* | Precomputed (Bai et al., 2019; curated by Jain et al., 2024) |
| AIDS | Molecular topology | ~700 | 8.9 | ~10 | Yes | Exact A* (uniform cost) | Precomputed (Bai et al., 2019; curated by Jain et al., 2024) |

**Total**: 5 real-world datasets, 3 domains, ~3,750 graphs, all with exact GED ground truth.

This exceeds the minimum of 3–5 datasets recommended by Demšar (2006) for Q1 journal standards and provides both within-domain gradients (IAM Letter LOW/MED/HIGH) and cross-domain validation (handwriting vs. software vs. molecular).

---

## 6. Implementation Plan

### Phase 1: GED Matrix Computation and Extraction

This is the immediate next step before any benchmark agent is launched.

**Task 1**: Compute IAM Letter GED matrices (LOW, MED, HIGH).
- Load graphs via `iam_letter_loader.py` (topology only, coordinates stripped).
- Compute all-pairs exact GED via `networkx.graph_edit_distance()`.
- Persist as `iam_letter_{low,med,high}.npz` in common format.
- Estimated time: ~75 minutes total (single-threaded). Parallelizable.

**Task 2**: Extract LINUX GED matrix from GraphEdX `.pt` files.
- Load `no_attr_data/linux/{train,val,test}.pt` and `*_result.pt` via `torch.load()`.
- Inspect tensor schema of `*_result.pt` to determine GED matrix layout.
- Convert PyG Data objects to NetworkX graphs (topology only via `edge_index`).
- Merge splits, apply `filter_by_node_count(n_max=12)`, extract GED submatrix.
- Persist as `linux.npz` in common format.
- Estimated time: < 5 minutes (load + conversion).

**Task 3**: Extract AIDS GED matrix from GraphEdX `.pt` files.
- Same procedure as LINUX, from `no_attr_data/aids/`.
- No node-count filtering expected (max ~10 nodes < N_max = 12).
- Persist as `aids.npz` in common format.
- Estimated time: < 5 minutes.

**Task 4**: Validate all matrices.
- Symmetry: `ged_matrix[i,j] == ged_matrix[j,i]` for all valid pairs.
- Diagonal: `ged_matrix[i,i] == 0` for all i.
- Non-negativity: `ged_matrix[i,j] >= 0` for all valid pairs.
- Triangle inequality spot checks: `ged_matrix[i,k] <= ged_matrix[i,j] + ged_matrix[j,k]` on random triples.
- Cross-validation: for LINUX, compare graph statistics (node count distribution, edge count distribution) against published values in Bai et al. (2019) Table 1 and Jain et al. (2024) to confirm data integrity.

### Phase 2: Benchmark Execution

Only after Phase 1 is validated. All 5 benchmark agents read from `data/eval/ged_matrices/*.npz` as their ground truth. No benchmark agent computes GED.

---

## 7. Changelog (v2, March 2026)

### Data Source Migration

The original dataset strategy (v1) relied on PyTorch Geometric's `GEDDataset` for LINUX and ALKANE, which downloads precomputed data from Google Drive files hosted by Bai et al. (2019). As of March 2026, these Google Drive links are permanently broken:

| Dataset | Google Drive ID (data) | Google Drive ID (GED pickle) | Status |
|---------|----------------------|------------------------------|--------|
| LINUX | `1nw0RRVgyLpit4V4XFQyDy0pI6wUEXSOI` | `14FDm3NSnrBvB7eNpLeGy5Bz6FjuCSF5v` | **404** |
| ALKANE | `1-LmxaWW3KulLh00YqscVEflbqr0g4cXt` | `15BpvMuHx77-yUGYgM27_sQett02HQNYu` | **404** |

The original SimGNN data folder has been reorganized by the owner. This affects every downstream project that depends on `GEDDataset` (Extended-SimGNN, EGSC, etc.).

### Resolution

We migrated to GraphEdX (Jain et al., NeurIPS 2024, arXiv:2409.17687), which provides a curated redistribution of the GED benchmark datasets with two improvements over the original:

1. **Isomorphism deduplication**: Removes structurally redundant graphs that cause train-test leakage (only 8.9% of original LINUX graphs were structurally unique).
2. **Uniform-cost topology-only GED**: The `no_attr_data/` folder provides GED computed without node/edge attribute costs, directly compatible with IsalGraph's topology-only encoding.

### Specific Changes

| Change | From (v1) | To (v2) | Rationale |
|--------|-----------|---------|-----------|
| LINUX source | PyG `GEDDataset` (Google Drive) | GraphEdX `no_attr_data/linux/` | Data availability + isomorphism deduplication |
| ALKANE | PyG `GEDDataset` (Google Drive) | **Removed** | Data unavailable + limited stats power + superseded by tree subset analysis |
| AIDS (uniform) | Rejected (§4.1 in v1) | **Accepted** (§2.5) | GraphEdX provides topology-only GED, resolving the label contamination confound |
| H₃ test method | Dedicated ALKANE dataset | Tree subset of LINUX | More robust within a larger graph population |
| Dataset count | 5 (IAM×3 + LINUX + ALKANE) | 5 (IAM×3 + LINUX + AIDS) | Maintained |
| Total graphs | ~3,400 | ~3,750 | Increased (AIDS has ~700 vs ALKANE's 150) |
| Domains | 3 (handwriting, software, molecular) | 3 (handwriting, software, molecular) | Unchanged |

### Impact on Downstream Agents

Agents 0–5 require the following updates (see `CASCADING_CHANGES.md` for details):

- **Agent 0 (Setup)**: Replace `pyg_ged_extractor.py` with `graphedx_loader.py`. Remove ALKANE references. Add AIDS loading. Update `DATASETS` list.
- **Agent 1 (Correlation)**: Replace `alkane` with `aids` in all dataset loops. Update H₃ to use LINUX tree subset. Add tree/non-tree stratification to the analysis.
- **Agents 2–5**: Replace `alkane` → `aids` in dataset references. No structural changes.

### Backup Data Source

As additional insurance against future data loss, the LINUX and AIDS graph structures (without GED matrices) are available in GEXF format from the GEDGNN repository (Piao et al., PVLDB 2023): https://github.com/ChengzhiPiao/GEDGNN — committed directly to git, not hosted on external file storage. These can serve as a cross-validation reference for graph structure integrity.
