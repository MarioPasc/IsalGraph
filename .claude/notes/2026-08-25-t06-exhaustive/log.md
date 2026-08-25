# T-06-exhaustive — work log

Re-execution of the T-06 IsalGraph arm with the **exhaustive** canonical form
(`canonical_string`) in place of the length-suboptimal `pruned_canonical_string`.

Started 2026-08-25. Base commit `855e4adc63dd20850b406b136785117d97a1145a` (`main`).
Output tree: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/T06_exhaustive/`

**The original `data/source/T06/` and `results/reports/T-06-full-recompute/` are
read-only for this ticket.** Nothing here writes into either.

---

## Status board

| step | state | note |
|---|---|---|
| 0. read plan + contracts | DONE | |
| 1. `isalgraph_exhaustive` backend + tests | DONE | 27 new tests, all green |
| 2. commit code | DONE | `a691b57` backend, `4cbc6cd` stats |
| 3. encode 15 cells @ 30 s | RUNNING | Suite 1 done: 5,350 graphs, 0 censored |
| 4. distances (new arm only) | pending | GED matrices reused verbatim |
| 5. statistics + reduced view | pending | |
| 6. SUMMARY.md / manifest.json | pending | |

Base commit at launch: `6befc1a` (another session moved HEAD from `855e4ad`
mid-work; re-checked before committing, per the standing rule).

---

## Architecture findings that shaped the design

Read before changing anything.

### F1. D14's fallback is a DRIVER policy, not a backend one

`t06_encode_worker.py` docstring, verbatim: *"The worker never decides D14. It
reports what happened -- `ok`, or `error` with the exception class name -- and
the driver applies the fallback policy. Keeping the policy in one place is what
stops a censored graph being silently dropped in one code path and retained in
another."*

The mechanism is a **second pass**: `t06_encode._apply_d14` collects every record
whose `error_family(error_kind) == "wall_clock"` and re-runs those indices with
`mode="fallback"`; `_stamp_fallback` then rewrites the record to
`status="censored", fallback_used=True`.

**Consequence for this ticket.** Putting the fallback inside
`_IsalGraphBackend.encode()` would be wrong twice over:

1. The worker would never learn a fallback happened, so the row would land as
   `status="ok", fallback_used=False`. A censored graph laundered into a
   completed one is precisely the bias D14 exists to expose.
2. The budget is enforced by the parent **killing the process**
   (`t06_encode._consume`). A backend-internal `except CanonicalizationTimeoutError`
   never runs for a killed graph, so the graphs most in need of a fallback would
   get none.

So the fallback is wired through the existing driver machinery, and the backend
only *names* its fallback variant. The backend change is: encode with
`canonical_string`, drop `SUITE1_ONLY`, declare `fallback_variant = "pruned"`.

### F2. The fallback target differs from the pruned arm's

D14's fallback for `isalgraph_pruned` is the **greedy-min** string
(`GreedyMinG2S`). For `isalgraph_exhaustive` the task specifies
`pruned_canonical_string`, which is strictly better: it is still a canonical
form, so the row stays inside the completeness theorem, whereas a greedy-min row
does not.

**Cascade, to honour "do not drop a graph":**

    canonical_string(30 s)  ->  pruned_canonical_string(30 s)  ->  greedy_min_string

`pruned` has its own ceiling (T-06 measured 24/400 on Mutagenicity, 4/400 on
Protein at a 2 s budget), so a two-tier cascade would still drop rows. Greedy-min
is O(n) greedy encodes and always terminates, so the third tier closes it.

The parent's per-line deadline in fallback mode is `budget_s + LINE_GRACE_S`
= 45 s, and greedy-min at n~100 is milliseconds, so the cascade fits inside it.

### F3. `\x1f` separator does not apply to this arm

`CONTRACTS.md` §3.1: `symbol_sep` is `"\x1f"` only for `min_dfs`, `size_null`,
`wl_subtree`. IsalGraph symbols are single characters from `{N,n,P,p,V,v,C,c,W}`,
so `symbol_sep = ""` and `length == len(encoding)`. The new arm inherits this by
**not** being added to `SYMBOL_SEP`, and `_join_symbols` asserts it per encode.

### F4. Fifteen cells, not fifteen datasets

10 Suite-2 (`linux grec protein aids_graphedx iam_letter_low iam_letter_med
aids_iam iam_letter_high coil_del mutagenicity`) + 5 Suite-1 (`linux aids
iam_letter_low iam_letter_med iam_letter_high`).

### F5. Distances reuse — symlink, never copy

The stats need every comparator's distance matrix, not just the new arm's.
`data/source/T06/distances/` is 518 MB / 190 files. It is **symlinked** into
`T06_exhaustive/distances/`, not copied and never moved: a symlink is read-only
consumption, costs no disk, and cannot mutate the pre-registered record.
`run_distances.sh`'s `[ -s "$target" ]` guard follows symlinks, so the reused
cells are skipped rather than recomputed.

---

## Decisions

| id | decision | rationale |
|---|---|---|
| X1 | fallback wired through driver `_apply_d14`, not inside `encode()` | F1 — a backend-internal fallback reports `ok/False` and never fires for a killed graph |
| X2 | 3-tier cascade exhaustive -> pruned -> greedy-min | F2 — pruned alone still drops rows, and "do not drop a graph" is the hard constraint |
| X3 | budget 30 s, recorded in every cell's `metadata.encode_budget_s` | a censoring rate is a property of its budget |
| X4 | every competitor stays in the data; reduced view is a flag | dropping from the campaign changes a pre-registered family's cardinality (`N_actual = 79`) |
| X5 | distances symlinked from T06 | F5 |

---

## Test suite

| when | passed | skipped | note |
|---|---|---|---|
| reference (T-09 close) | 2,583 | 321 | the floor in `.claude/CLAUDE.md` |
| this ticket | **2,610** | 321 | +27, all from `tests/unit/test_t06_exhaustive.py` |

Measured in 9 min 26 s. The one failure the first full run reported
(`test_admissibility_e2::test_quick_run_classifies_every_representation`) is
fixed and re-verified 15/15; that run had started before the fix landed.

**Why E2 failed, and why the fix is a test fix and not a grid regeneration.**
Part C classifies a representation from its F3 record in T-04a's **frozen**
admissibility grid (`/media/.../T-04a/grid_200.json`). A backend registered
after that grid was frozen has no record, so `e2_completeness` reports
`class = None` with reason *"no admissible distance and no F3 record; not
classified"* — which is the correct answer. The test asserted every complete
invariant is class III, a premise that is now false. Regenerating the grid to
absorb the new arm would move a pre-registered artifact, so the assertion is
scoped to what the grid covers and the uncovered arms are asserted to be
unclassified *for the stated reason* rather than skipped.

---

## Timeline

- `2026-08-25` — read plan, contracts, drivers; wrote this log.
- `2026-08-25` — backend + 27 tests; ruff clean, `mypy --strict` clean.
- `2026-08-25` — committed `a691b57` (backend) and `4cbc6cd` (stats knobs).
- `2026-08-25` — launched the encoding campaign. Suite 1 finished in seconds:
  5,350 graphs, **0 censored, 0 error** at the 30 s budget.

---

## Independent verification of a written cell

`suite1/linux`, all 89 graphs, re-derived from the cohort without going through
the campaign code path:

| check | mismatches |
|---|---|
| `encoding` equals `canonical_string(to_sparse_graph(G))` | **0** / 89 |
| `length` equals `len(string)` (§3.1, empty separator) | **0** / 89 |
| `entropy_bits` equals `L·log2(9)` | **0** / 89 |
| `S2G(encoding)` isomorphic to the cohort graph | **0** / 89 |

Provenance in the file: budget 30.0 s, engine `cpp`, build hash
`298fc1188bf1b051`, seed 42, `src_commit = 4cbc6cd`.

This is the gate for building statistics on the new arm. It passed before any
distance was computed.
