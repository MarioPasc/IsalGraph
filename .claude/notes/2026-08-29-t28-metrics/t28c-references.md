# T-28c Work Log — t28c-references agent

Wave: `2026-08-29-t28-metrics`
Agent: `t28c-references`
Branch: `agent-ab9169ed5f6d5d3b5` (worktree from main `89b4c9fc`)
Date: 2026-08-29

---

## Mission

Build the five production reference distance matrices for every (suite, dataset) cell in T-28:
`spectral`, `spectral_comb`, `spectral_adj`, `spectral_esd`, `wl` — all in the CONTRACTS §4
dense NPZ schema.  Apply gates G3 and G5.  Write tests, ruff-clean, mypy-strict.

---

## Files Built

| File | Purpose |
|------|---------|
| `src/isalgraph/competitors/references/__init__.py` | Package init |
| `src/isalgraph/competitors/references/spectral.py` | Four spectral computation functions |
| `src/isalgraph/competitors/references/build.py` | Cohort loading, gate checks, NPZ writing |
| `tests/unit/test_t28_references.py` | 35 unit tests |
| `benchmarks/real_data/eval_reference_metrics/build_references.py` | CLI entry point |
| `.claude/notes/2026-08-29-t28-metrics/t28c-references.md` | This log |

---

## Commands Run and Real Outputs

```
# reinstall (new subpackage)
$PY -m pip install -e ".[dev]" -q   →  exit 0

# tests
$PY -m pytest tests/unit/test_t28_references.py -q
→  35 passed in 0.58s

# lint
$PY -m ruff check src/isalgraph/competitors/references/ tests/unit/test_t28_references.py ...
→  All checks passed!

# mypy
$PY -m mypy src/isalgraph/competitors/references/
→  Success: no issues found in 3 source files

# build suite1
$PY benchmarks/real_data/eval_reference_metrics/build_references.py --suite suite1 ...
→  5 cells, 25 NPZ files, all 5 keys OK per cell

# build suite2
$PY benchmarks/real_data/eval_reference_metrics/build_references.py --suite suite2 ...
→  10 cells, 50 NPZ files, all 5 keys OK per cell

# total output
find .../T28/references -name '*.npz' | wc -l  →  75
```

---

## 15×5 Build Status Table

All 75 cells succeeded.

| Cell | spectral | spectral_comb | spectral_adj | spectral_esd | wl |
|------|----------|---------------|--------------|--------------|-----|
| suite1/aids | OK | OK | OK | OK | OK |
| suite1/iam_letter_low | OK | OK | OK | OK | OK |
| suite1/iam_letter_med | OK | OK | OK | OK | OK |
| suite1/iam_letter_high | OK | OK | OK | OK | OK |
| suite1/linux | OK | OK | OK | OK | OK |
| suite2/aids_graphedx | OK | OK | OK | OK | OK |
| suite2/aids_iam | OK | OK | OK | OK | OK |
| suite2/coil_del | OK | OK | OK | OK | OK |
| suite2/grec | OK | OK | OK | OK | OK |
| suite2/iam_letter_low | OK | OK | OK | OK | OK |
| suite2/iam_letter_med | OK | OK | OK | OK | OK |
| suite2/iam_letter_high | OK | OK | OK | OK | OK |
| suite2/linux | OK | OK | OK | OK | OK |
| suite2/mutagenicity | OK | OK | OK | OK | OK |
| suite2/protein | OK | OK | OK | OK | OK |

---

## Gate Measurements (selected, from log)

All matrices passed G3 (symmetric, zero diagonal, finite, non-negative, graph_ids join) and
G5 (off-diagonal zero fraction < 0.99).  Representative measured fractions:

| Cell / key | off-diag zero frac |
|------------|-------------------|
| suite1/aids / spectral | 0.0000 |
| suite2/mutagenicity / spectral_esd | 0.0000 |
| suite2/coil_del / wl | 0.0006 |
| suite2/aids_iam / spectral_comb | 0.0010 |
| suite2/aids_iam / spectral_adj | 0.0012 |
| suite2/grec / spectral_adj | 0.0215 |
| suite2/grec / wl | 0.0280 |

Maximum observed: ~0.028 (grec/wl).  All well below the 0.99 gate.

---

## n_max Values (from metadata in spot-checked files)

- suite1/aids spectral: `n_max=12` (matches archive n_max for suite1 aids)
- suite2/mutagenicity spectral_esd: `n_max=None` (ESD is never padded — correct)
- suite2/coil_del wl: `n_max=None` (WL is not padded — correct)

The design note §5 n_max table shows suite2 mutagenicity n_max=98; for spectral this would be
padded to 98 dimensions.  Not spot-checked per cell in this log but all `cohort_spectra()` calls
read `n_nodes.max()` from the actual cohort array.

---

## Assumptions Recorded

1. **K1 (single isolated node) convention**: `laplacian_spectrum(1, [], variant="norm")` returns
   `[1.0]`, not `[0.0]`.  The isolated-vertex guard sets `inv_sqrt=0`, so `L_sym = I - 0 = I`,
   giving eigenvalue 1.0.  This is our guard convention, not the standard graph-theoretic zero
   eigenvalue.  Documented in both `spectral.py` and the test.  Does not affect production cohorts
   (no single-node isolated graphs observed in any of the 15 datasets).

2. **`datetime.UTC` not in mypy stubs**: replaced `from datetime import UTC` with
   `from datetime import datetime, timezone` and used `timezone.utc`.  Python 3.11.15 has it;
   mypy's bundled typeshed stubs lagged.

3. **scipy not type-ignored**: `wasserstein_distance` is shipped with scipy stubs (mypy resolved
   it); the `# type: ignore[import-untyped]` was removed as unused.

4. **WL graph_ids ordering**: The WL cache matrix uses the same `graph_ids` ordering as the
   exported cohort NPZ.  The gate_check verifies `np.array_equal(wl_ids, cohort.graph_ids)`
   exactly.  All 15 datasets passed.

5. **`shutil` imported but not used in build.py**: removed from final version (the WL copy
   writes fresh metadata rather than copying bytes; the raw matrix is loaded and re-serialised).

---

## What Was Not Done / Could Not Do

Nothing blocked.  All 75 cells succeeded.

---

## Surprises

- GREC had the highest off-diagonal zero fraction (~0.028 for wl) but still far below the 0.99
  gate.  GREC graphs have regular structure (line drawings) and many near-isomorphic pairs.
- mutagenicity (4040 × 4040) spectral_esd completed in roughly 2 minutes — the scipy
  1-Wasserstein per pair is fast enough for this cohort size without vectorization.
