# Wave summary — `2026-08-13-t05-bounds`

**Date**: 2026-08-13 · **Ticket**: T-05, Bounded GED via GEDLIB over Suite 2
**Base commit**: `885d98d8e6b37dfeb98c4df741510fc28d4a8615`
**Integration branch**: `integration/2026-08-13-t05-bounds`
**Design note**: `.claude/notes/review/tasks/T-05-design.md` · **Contracts**: `CONTRACTS.md` (this dir)

## User's original prompt, verbatim

> Load /review-ticket and complete the ticket T-05 in .claude/notes/review/plan/tickets.md ; Remember
> to read and understand the results of Ticket T-27, which are detailed in
> /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/results/reports/T-27-ged-bound-bakeoff ;
> You must iterate until the jobs for computing the approx GED (upper and lower bounds) are completed
> and the results are correctly organized and in the expected format in
> /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED ; Regarding the
> expected format, i'd say that the best would be to use the same format as the exact ged
> (/media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/GED_PRECOMPUTED/extended_merged_exact_ged)
> so that the loader is the same when we have to code the experiments. Datasets are in
> /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED/datasets/IAM_Database/extracted
> Save the lower bound results in
> /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED/LB and upper
> bound in /media/mpascual/Sandisk2TB/research/ISAL/completed/isalgraph/data/source/APPROX_GED/UB.You
> must re-format the IAM Database datasets in the format you need them to be and copy them to Picasso
> in picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/isalgraph ; SInce we have little
> to no room in fscratch for high filecount, you should take inspiration from the way T03 (exact GED
> computation) approached this problem and also prioritize having larger files with less filecount (I
> think the agent took note in .claude/notes/review/tasks/T-03-design.md) You must plan the parallel
> jobs correctly. Save enough metadata of the runs across datasets so that we can afterwards analyze
> our data for the experiments. Only launch subagents for disentabled and parallel implementations
> which are long enough to be worth it. Think step by step, reason.

## Agents

| Agent | Model | Branch | Head | Verdict | Log |
|---|---|---|---|---|---|
| `wave-t05-export` | Opus 5, worktree | `worktree-agent-aa518b70750701b10` | `68149ee` | **ACCEPT** | [t05-export.md](t05-export.md) |
| `wave-t05-runner` | Opus 5, worktree | `worktree-agent-ab10166d8f9bb07a0` | `3b7597c` | **ACCEPT** | [t05-runner.md](t05-runner.md) |
| `wave-t05-slurm` | Opus 5, worktree | `worktree-agent-a60da166efbbe5eac` | `abe49ae` | **ACCEPT** | [t05-slurm.md](t05-slurm.md) |

Definition files in `.claude/agents/` were **not discovered by the watcher**, so all three were spawned
as `general-purpose` with explicit `isolation: worktree` and the definition body folded into the
prompt — the documented fallback. Archived copies in `agents/`.

## Merge record

Merged from a clean checkout at `BASE_SHA`, contracts first then by increasing diff size, fast suite
after each. **Zero merge conflicts** — the ownership partition held.

| Step | Suite (`tests/unit/`, GEDLIB on path) |
|---|---|
| baseline at `885d98d` | 8 failed · **907** passed · 1 skipped |
| `7ad157c` orchestrator amendments | — |
| merge `t05-export` | 8 failed · 1045 passed · 1 skipped |
| merge `t05-runner` | 8 failed · 1136 passed · 1 skipped |
| merge `t05-slurm` | 8 failed · 1179 passed · 1 skipped |
| `1bce903` `fix(integration)` | 8 failed · **1182** passed · 1 skipped |

Full suite `tests/ -q`: **8 failed · 1570 passed · 272 skipped**. `mypy --strict` clean.
`ruff`: 28 errors, **identical count at base**, all in visualization/synthetic files this wave never
touched.

**The 8 failures pre-date the wave** — `test_export_graphs.py`'s real-data tests, red because of the
path defect below. Verified at `885d98d` by the orchestrator and independently by two agents.

> ⚠ **Hold `PYTHONPATH` fixed when comparing suites.** 43 tests skip when the in-place GEDLIB build is
> off the path (44 skipped without, 1 with). Two tracks reported skip counts of 44 and 1 for the same
> suite; a branch measured one way against a baseline measured the other appears to invent 43 tests.

## Verification the orchestrator personally re-ran

Nothing below is taken from an agent's log.

| Check | Result |
|---|---|
| All ten Suite-2 cohort rows, read off the written `.npz` | exact; **16,370 graphs / 21,710,892 pairs** |
| `graph_ids` vs the Suite-1 census, 4 datasets | identical element-wise |
| COIL-DEL split-index enumeration | 3,900 indexed / 7,200 on disk / **100 classes × 39** |
| Subsample | 28,000 pairs, 14 bins at the 2,000 ceiling, `dataset_key` `<U15` |
| Probe ∩ subsample | **0** |
| `bin_table.json` | edges match §5; totals **21,710,892**; every row sums to its own `C(N,2)` |
| **T-27 reproduction, running the runner myself** | `BRANCH_FAST` sha `e95b44c7edad1369`, `BIPARTITE` sha `2528fd19b98accb0` — **byte-identical, max diff 0.0** |
| Same with `--record-orientations` | `ub` unchanged, `ub == min(fwd,rev)` on all pairs |
| Suite-1 `aids` ⊂ Suite-2 `aids_graphedx` | **True**, overlap exactly 769 of 819 |
| **Local smoke, LINUX end to end** | export → 3 campaigns → 3 merges → cross-fill → gates |
| Cross-fill | 89 graphs, 3,916 pairs, **79 certified (2.02 %)**, 0 inverted |
| **Gates G2 / G3 / G4 / lb-consistency** | **all PASS** after the integration fix |

## Interventions

| # | What | Outcome |
|---|---|---|
| 1 | `graph_ids` contract wrong (`{key}_{split}_{id}`) | **Mine.** Amended to the loader's native id |
| 2 | Class counts asserted in §2 were **raw**, not post-filter | **Mine.** No count asserted; measured counts ship in the manifest |
| 3 | Subsample needed two files, not one | Adopted as amendment 3 |
| 4 | Merge CLI could not express the flat `ubt` output | **Mine.** Ruled a separate entry point; runner built it |
| 5 | Probe pair list was unowned | Resumed `wave-t05-export`; its fallback would have biased the rate **low** |
| 6 | `wave-t05-slurm` proposed skipping G3's AIDS arm | **Overridden in its favour** — the id join recovers 295,296 pairs |
| 7 | It reported censored pairs as NaN | **Corrected**: they are `inf`; an `isnan` filter passes all 92 |
| 8 | CONTRACTS §6.1's lazy-guard rationale | **Mine, and wrong.** See below |
| 9 | "`--compute` halves the work" | **Mine, and wrong.** 1.81× / 1.28× |

## Findings that outlive T-05 — for `review-close` to propagate

1. **Decision 22's reproduction script cannot run on today's tree.** `export_graphs.py:430` and
   `cohort_audit.py:254` resolve GraphEdX as `<source>/GED_PRECOMPUTED/<NAME>`; the real path is
   `.../datasets/<NAME>`. Because IAM sits under `APPROX_GED/datasets/` and GraphEdX under
   `GED_PRECOMPUTED/datasets/`, **no single `--source` resolves both**. T-01's tracked cohort
   reproduction is **red on this machine** for the LINUX and AIDS-GraphEdX rows. Not patched — frozen
   artifacts, and the fix is a two-root refactor, which is a PI call. **Owner: T-01/T-06.**
2. **Labels do not survive the connectivity filter.** Letter LOW retains **9 of 15** classes (loses
   A, E, F, H, K, T), GREC **17 of 22** (loses 5, 6, 9, 15, 21); LINUX and AIDS-GraphEdX carry **no
   class label at all**. "Letter, 15 classes" is false of the filtered cohort. **Owner: T-18, T-06.**
3. **Size and provenance are confounded across the size bins.** Bins 0–2 ~90 % Letter; bins 8–13
   50–97 % Mutagenicity + COIL-DEL; bin 13 is **97.1 % Mutagenicity**. Density moves with provenance
   (0.607 → 0.094). **The within-dataset slope is now the primary AE.1 measurement**; the pooled curve
   is a descriptive overlay. Frozen in design §7 before any production pair. **Owner: T-05, T-06.**
4. **`decisions.md` §6's orientation figure does not describe the production method.** §6 says 33.2 %
   / 1.15 from our BP on 400 LINUX pairs. `BIPARTITE` measures **22.8 %** on all 3,916 LINUX pairs and
   **11.2 %** on Mutagenicity — the rate falls with `n` while the magnitude does not (3.24 → 3.00).
   Rewrite from the `ubt` subsample. **Owner: T-05 close.**
5. **CLAUDE.md's "assert `0 < value < inf` on every read" is wrong per pair.** GED is legitimately 0
   for isomorphic graphs — 15.5 % of Letter LOW. Not edited; it is the user's file. **Owner: user.**
6. **The lazy `zero_ok` guard buys nothing and my stated reason for it was false.** The `(n,m)`
   precheck short-circuits before VF2 on **99.4–99.5 %** of large-cohort pairs; the guard is **0.1 %**
   of per-pair cost. Measured 0.998×–1.005×. The change stays; the rationale must not be repeated.

## What the decomposition got wrong

**Two cross-track integration defects that neither track could have caught alone**, both found by the
local smoke and both of which would have failed on Picasso *after* the compute:

- `int(None)` on Suite 2's `"n_max": null`. The exporter was right, the merge assumed absent-or-numeric,
  and every merge fixture hardcodes `12`.
- Seven CONTRACTS §4 metadata keys missing from all three role files — caught by the **independent**
  gate, which is precisely why `wave-t05-slurm` was told to code against the contract and not against
  the runner. Track B's own tests were green.

The lesson is that the disjoint-ownership rule guarantees no *merge* conflict but says nothing about
*interface* conflict, and only end-to-end execution on real data surfaces it. The smoke was cheap
(~90 s) and would have cost a full campaign to skip.

## Open

- Picasso submission of the four campaigns — orchestrator, pending user sign-off.
- The calibration ladder (design §6) — a later wave.
- Analysis deliverables (design §7) — a later wave.
