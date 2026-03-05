# IsalGraph Implementation Progress

## Overview

Migration of original flat-layout code to src/ package layout with bug fixes,
proper typing, and comprehensive test coverage.

## Implementation Order

### Core Modules (zero external deps)

| # | Module | Status | Notes |
|---|--------|--------|-------|
| 1 | `errors.py` | DONE | Custom exception hierarchy |
| 2 | `types.py` | DONE | Shared type aliases |
| 3 | `core/cdll.py` | DONE | Clean migration, no bugs, `__slots__`, type annotations |
| 4 | `core/sparse_graph.py` | DONE | Fix B1 (edge_count=0), Fix B9 (duplicate edge guard), `logical_edge_count()` |
| 5 | `core/string_to_graph.py` | DONE | Fix B6 (use `cdll.get_value(ptr)` in V/v/C/c handlers) |
| 6 | `core/graph_to_string.py` | DONE | Fix B2-B5, B7, B8. Added `_check_reachability()` |
| 7 | `core/canonical.py` | DONE | Exhaustive backtracking search for labeling-independent canonical string |

### Tests

| # | Test file | Status | Notes |
|---|-----------|--------|-------|
| 8 | `tests/unit/test_cdll.py` | DONE | 14 tests (incl. repr) |
| 9 | `tests/unit/test_sparse_graph.py` | DONE | 26 tests (incl. has_edge errors, add_edge errors, isomorphism edge cases, repr) |
| 10 | `tests/unit/test_string_to_graph.py` | DONE | 17 tests (incl. property accessors) |
| 11 | `tests/unit/test_graph_to_string.py` | DONE | 13 tests (incl. reachability, B2-B4 regression) |
| 12 | `tests/unit/test_roundtrip.py` | DONE | Phase 1 (19 short strings x 2 dirs) + Phase 2 mini (20 seeds x 2 dirs) + specific graphs |
| 13 | `tests/unit/test_canonical.py` | DONE | 18 tests: Levenshtein, basics, invariance, discrimination, distance, directed c-branch, _is_reachable |
| 14 | `tests/unit/test_errors.py` | DONE | 5 tests: error hierarchy |

### Adapters

| # | Module | Status | Notes |
|---|--------|--------|-------|
| 15 | `adapters/base.py` | DONE | Abstract `GraphAdapter[T]` with bridge pattern |
| 16 | `adapters/networkx_adapter.py` | DONE | Node label mapping, `TypeAlias` for NxGraph |
| 17 | `adapters/igraph_adapter.py` | DONE | |
| 18 | `adapters/pyg_adapter.py` | DONE | edge_index tensor conversion, CUDA support |

### Integration / Property Tests

| # | Test file | Status | Notes |
|---|-----------|--------|-------|
| 19 | `tests/integration/test_networkx_adapter.py` | DONE | 58 tests: adapter + cross-validation + random GNP + all-starting-nodes |
| 20 | `tests/integration/test_igraph_adapter.py` | DONE | 8 tests: adapter + round-trip |
| 21 | `tests/integration/test_canonical_networkx.py` | DONE | 24 tests: invariance (trees, cycles, complete, Petersen) + discrimination |
| 22 | `tests/integration/test_pyg_adapter.py` | DONE | 30 tests: from_external, to_external, round-trip, CUDA tensors |
| 23 | `tests/property/test_roundtrip_property.py` | DONE | Hypothesis-based, 500 examples each, directed + undirected |

### Benchmarks

| # | Script | Status | Notes |
|---|--------|--------|-------|
| 24 | `benchmarks/random_roundtrip.py` | DONE | 945/945 pass (100%), 12 graph families, 3.28s |
| 25 | `benchmarks/canonical_invariance.py` | DONE | 71/71 pass (100%), invariance + discrimination, 175s |
| 26 | `benchmarks/string_length_analysis.py` | DONE | 105 graphs, compression analysis, canonical vs greedy |
| 27 | `benchmarks/levenshtein_vs_ged.py` | DONE | 85 pairs, Pearson r=0.83, locality confirmed |

## Code Quality

| Check | Status |
|-------|--------|
| `ruff check src/ tests/` | PASS (0 errors) |
| `mypy src/isalgraph/` | PASS (0 errors, strict mode) |
| `pytest tests/ -v` | PASS (308/308) |
| Coverage | 99% (642 stmts, 8 miss — all defensive ImportError/RuntimeError guards) |

## Benchmark Results

### Phase 2: Random Round-Trip (1000 tests, seed=42)

- **945 tests executed, 945 passed (100.0%)**
- Graph families: random_string (400), GNP (100), trees (100), Barabasi-Albert (100),
  Watts-Strogatz (100), directed GNP (100), cycle (9), complete (9), star (9),
  wheel (9), ladder (8), Petersen (1)
- Max nodes: 20, max string length: 50
- Total time: 3.28s (avg 3.5ms/test)
- Cross-validated with `nx.is_isomorphic`

### Phase 3: Canonical String Invariance (100 tests, seed=42, max_nodes=8)

- **71 tests executed, 71 passed (100.0%)**
- 49 invariance tests (isomorphic pairs via random relabeling): 100%
- 22 discrimination tests (non-isomorphic pairs): 100%
- Graph families: trees, cycles, complete, star, wheel, GNP, Barabasi-Albert, ladder
- Total time: 175.25s (avg 2.47s/test — exhaustive backtracking is expensive)

### String Length Analysis (105 graphs, seed=42, max_nodes=50)

- **105 graphs analyzed, 0 errors**
- Compression ratio |w|/N²: mean=0.66, min=0.04 (star-50), max=1.95 (complete-20)
- Sparse graphs compress well: trees 4-8% of N², BA-m1 7-10%
- Dense graphs can exceed N²: complete at ~1.9×, dense GNP at ~1.5×
- Stars achieve optimal compression (N-1 chars for N nodes)
- Canonical vs greedy: nearly identical lengths (only 3 chars saved across all small graphs)
- Theoretical Eq. 9 from preprint overestimates string length for our CDLL-based algorithm
  on sparse graphs, and underestimates on dense graphs (expected: different instruction set)

### Levenshtein vs Graph Edit Distance (85 pairs, seed=42, max_nodes=7)

- **85 pairs tested, 0 errors**
- Overall correlation: Pearson r=0.83 (p<10⁻²²), Spearman r=0.59 (p<10⁻⁹)
- Family pairs: Pearson r=0.88 — excellent structural correlation
- Random GNP pairs: Pearson r=0.84 — strong
- **Locality confirmed**: k=1 edge edit → mean Levenshtein 4.2; k=4 → 6.1 (monotonically increasing)
- Validates preprint Section 2.3: "small changes in graph → small changes in string"

## Bug Tracker

### Original Code Bugs (fixed during migration)

| # | File | Bug | Fix | Status |
|---|------|-----|-----|--------|
| B1 | sparse_graph.py | `_edge_count = 1` | Init to 0 | FIXED |
| B2 | graphtostring.py | `sort by a+b` | Sort by `abs(a)+abs(b)` | FIXED |
| B3 | graphtostring.py | `while ... and ...` | Change to `or` | FIXED |
| B4 | graphtostring.py | Pointers not updated | Update after each operation | FIXED |
| B5 | graphtostring.py | Debug `print()` | Remove | FIXED |
| B6 | stringtograph.py | CDLL idx passed to add_edge | Use `cdll.get_value(ptr)` | FIXED |
| B7 | graphtostring.py | Graph idx passed to insert_after | Use CDLL node idx | FIXED |
| B8 | graphtostring.py | V/v creates nodes for already-mapped inputs | Filter by `_i2o` | FIXED |

### Bugs Found During Testing

| # | File | Bug | Fix | Status |
|---|------|-----|-----|--------|
| B9 | sparse_graph.py | `add_edge` increments `_edge_count` for duplicate edges | Guard with `if target not in self._adjacency[source]` | FIXED |

## Architectural Notes

- **Directed graph limitation**: V/v only creates `existing -> new` edges. Nodes reachable
  only via incoming edges from unplaced nodes cannot be inserted. GraphToString raises
  `ValueError` if not all nodes are reachable from `initial_node` via outgoing edges.
  This is inherent to the instruction set, not a bug.

- **Canonical string uses exhaustive search**: The greedy GraphToString is not
  isomorphism-equivariant because Python set iteration order depends on integer
  node IDs. The canonical string computation uses backtracking search over all
  valid neighbor choices at V/v steps, with in-place mutation and undo for
  performance. This produces a true complete graph invariant.

- **Canonical search complexity**: For graphs up to ~8 nodes, the exhaustive search
  completes in seconds. For 10-node graphs (e.g., Petersen), it can take ~10s per
  starting node. This is acceptable for correctness verification but would need
  pruning/heuristics for production use on larger graphs.

## Picasso HPC Scaling

| # | Component | Status | Notes |
|---|-----------|--------|-------|
| 28 | `slurm/config.yaml` | DONE | Central config: paths, SLURM defaults, per-benchmark params |
| 29 | `slurm/launch.sh` | DONE | Master executor: reads config, dispatches sbatch, --dry-run |
| 30 | `slurm/workers/*_login.sh` (x4) | DONE | Login-node prep: validate env, install deps |
| 31 | `slurm/workers/*_slurm.sh` (x4) | DONE | Compute workers: module load, run benchmark |
| 32 | Benchmark --mode/--csv/--plot/--table | DONE | All 4 benchmarks support Picasso mode |
| 33 | ProcessPoolExecutor parallelization | DONE | 3 benchmarks (roundtrip, canonical, levenshtein) |
| 34 | `plotting_styles.py` extensions | DONE | INSTRUCTION_COLORS, FAMILY_COLORS, binomial_ci, bootstrap_ci, save_figure, save_latex_table |
| 35 | Benchmark README.md (x4) | DONE | Scientific claim, method, figure/table description, running instructions |
| 36 | `docs/DEVELOPMENT.md` | DONE | Updated with Picasso coding standards |

### Per-Benchmark Outputs

Each benchmark now generates (with --csv --plot --table flags):
- **JSON**: Raw results (always)
- **CSV**: Tabular data for downstream analysis
- **PDF + PNG**: Publication-quality IEEE figure (Paul Tol colorblind-safe palette)
- **LaTeX .tex**: Table ready for paper inclusion

### SLURM Resource Allocation

| Benchmark | Time | CPUs | RAM | Notes |
|-----------|------|------|-----|-------|
| random_roundtrip | 4h | 32 | 32G | 10K tests, max_nodes=50 |
| canonical_invariance | 2d | 64 | 64G | 2K tests, max_nodes=8 |
| string_length_analysis | 8h | 16 | 16G | max_nodes=200 |
| levenshtein_vs_ged | 2d | 64 | 64G | 500 pairs, max_nodes=8 |

## Environment

- Python 3.11 (conda env: isalgraph)
- PyTorch 2.6.0+cu124 (CUDA 12.4, RTX 4060 8GB)
- PyTorch Geometric 2.7.0
- NetworkX, python-igraph, hypothesis, pandas, scipy, matplotlib, pyyaml installed
