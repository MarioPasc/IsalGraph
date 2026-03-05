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
| 8 | `tests/unit/test_cdll.py` | DONE | 12 tests |
| 9 | `tests/unit/test_sparse_graph.py` | DONE | 15 tests (incl. B1 regression) |
| 10 | `tests/unit/test_string_to_graph.py` | DONE | 13 tests |
| 11 | `tests/unit/test_graph_to_string.py` | DONE | 10 tests (incl. B2, B3, B4 regression) |
| 12 | `tests/unit/test_roundtrip.py` | DONE | Phase 1 (19 short strings x 2 dirs) + Phase 2 mini (20 seeds x 2 dirs) + specific graphs |
| 13 | `tests/unit/test_canonical.py` | DONE | 18 tests: Levenshtein, basics, invariance, discrimination, distance |

### Adapters

| # | Module | Status | Notes |
|---|--------|--------|-------|
| 14 | `adapters/base.py` | DONE | Abstract `GraphAdapter[T]` with bridge pattern |
| 15 | `adapters/networkx_adapter.py` | DONE | Node label mapping, `TypeAlias` for NxGraph |
| 16 | `adapters/igraph_adapter.py` | DONE | |
| 17 | `adapters/pyg_adapter.py` | DONE | edge_index tensor conversion (no integration tests yet -- PyTorch not in env) |

### Integration / Property Tests

| # | Test file | Status | Notes |
|---|-----------|--------|-------|
| 18 | `tests/integration/test_networkx_adapter.py` | DONE | 58 tests: adapter + cross-validation + random GNP + all-starting-nodes |
| 19 | `tests/integration/test_igraph_adapter.py` | DONE | 8 tests: adapter + round-trip |
| 20 | `tests/integration/test_canonical_networkx.py` | DONE | 24 tests: invariance (trees, cycles, complete, Petersen) + discrimination |
| 21 | `tests/property/test_roundtrip_property.py` | DONE | Hypothesis-based, 500 examples each, directed + undirected |

### Benchmarks

| # | Script | Status | Notes |
|---|--------|--------|-------|
| 22 | `benchmarks/random_roundtrip.py` | NOT STARTED | Phase 2 at scale |
| 23 | `benchmarks/canonical_invariance.py` | NOT STARTED | Phase 3 at scale |

## Code Quality

| Check | Status |
|-------|--------|
| `ruff check src/ tests/` | PASS (0 errors) |
| `mypy src/isalgraph/` | PASS (0 errors, strict mode) |
| `pytest tests/ -v` | PASS (246/246) |
| Coverage | 90% |

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
