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

### Tests

| # | Test file | Status | Notes |
|---|-----------|--------|-------|
| 7 | `tests/unit/test_cdll.py` | DONE | 12 tests |
| 8 | `tests/unit/test_sparse_graph.py` | DONE | 15 tests (incl. B1 regression) |
| 9 | `tests/unit/test_string_to_graph.py` | DONE | 13 tests |
| 10 | `tests/unit/test_graph_to_string.py` | DONE | 10 tests (incl. B2, B3, B4 regression) |
| 11 | `tests/unit/test_roundtrip.py` | DONE | Phase 1 (19 short strings x 2 dirs) + Phase 2 mini (20 seeds x 2 dirs) + specific graphs |

### Adapters

| # | Module | Status | Notes |
|---|--------|--------|-------|
| 12 | `adapters/base.py` | DONE | Abstract `GraphAdapter[T]` with bridge pattern |
| 13 | `adapters/networkx_adapter.py` | DONE | Node label mapping, `TypeAlias` for NxGraph |
| 14 | `adapters/igraph_adapter.py` | DONE | |
| 15 | `adapters/pyg_adapter.py` | DONE | edge_index tensor conversion (no integration tests yet -- PyTorch not in env) |

### Integration / Property Tests

| # | Test file | Status | Notes |
|---|-----------|--------|-------|
| 16 | `tests/integration/test_networkx_adapter.py` | DONE | 58 tests: adapter + cross-validation + random GNP + all-starting-nodes |
| 17 | `tests/integration/test_igraph_adapter.py` | DONE | 8 tests: adapter + round-trip |
| 18 | `tests/property/test_roundtrip_property.py` | DONE | Hypothesis-based, 500 examples each, directed + undirected |

### Phase 3: Canonical String

| # | Module | Status | Notes |
|---|--------|--------|-------|
| 19 | `core/canonical.py` | NOT STARTED | `w*_G = lexmin{w in argmin_{v in V} |G2S(G,v)|}` |
| 20 | `benchmarks/canonical_invariance.py` | NOT STARTED | Verify invariant for isomorphic graph pairs |

### Benchmarks

| # | Script | Status | Notes |
|---|--------|--------|-------|
| 21 | `benchmarks/random_roundtrip.py` | NOT STARTED | Phase 2 at scale |
| 22 | `benchmarks/canonical_invariance.py` | NOT STARTED | Phase 3 |

## Code Quality

| Check | Status |
|-------|--------|
| `ruff check src/ tests/` | PASS (0 errors) |
| `mypy src/isalgraph/` | PASS (0 errors, strict mode) |
| `pytest tests/ -v` | PASS (204/204) |
| Coverage | 88% |

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
